// Decision tree induction on the GPU, with information gain, gain ratio or
// the Gini index as the attribute selection measure. It grows exactly the
// tree that dtree_cpu grows (same splits, same thresholds, same leaves); the
// nodes are only numbered level by level instead of depth first.
//
// The CPU builder expands one node at a time. Here a whole level of the tree
// is expanded at once, and the unit of parallel work is not the node but the
// record and the candidate split:
//
//   1. histogram   Every record of the level adds itself to the
//                  (attribute value, class) histograms of its node, one
//                  atomic increment per attribute.
//   2. score       One thread per (node, attribute) scans that histogram and
//                  finds the best split of the attribute: the multiway split
//                  or the best threshold (gain, gain ratio), the best
//                  threshold or value subset (Gini).
//   3. choose      One thread per node picks the best attribute.
//   4. assign      Every record looks up the split of its node and moves to
//                  one of the new children.
//   5. regroup     The class counts of all new children are counted, the
//                  host decides which of them are leaves, and the records of
//                  the others are sorted by node (radix sort) to form the
//                  next level. Records in leaves drop out.
//
// The host only keeps the tree: per level it downloads the chosen split and
// the class counts of every node. Because the records of a level are sorted
// by node, a level whose histograms do not fit in the memory budget is
// simply processed in several ranges of nodes.
//
// The scores are computed from the same integer counts with the same
// formulas, in double precision, as on the CPU (see src/dtree_cpu.cpp for
// the formulas and include/dtree.h for the tie rules).
//
// usage: dtree_gpu <table.csv> [--measure gain|gain_ratio|gini] [options]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "cuda_compat.h"
#include "dataset.h"
#include "dtree.h"
#include "dtree_app.h"

namespace {

using namespace ml;

constexpr uint32_t kNone = 0xFFFFFFFFu;   // record that is in a finished leaf

// Memory budget for the histograms of one range of nodes (--hist-mb).
size_t g_hist_budget = size_t(256) << 20;

struct Shape {
    size_t n;          // training records
    int32_t d, C;      // attributes, classes
    int32_t B;         // bins of all attributes together
    int32_t measure;   // 0 gain, 1 gain ratio, 2 Gini
};

// Best split of one attribute at one node (the Candidate of the CPU builder).
struct DevCand {
    double gain, split_info, gain_ratio;
    double gini_split, gini_reduction;
    int32_t valid;
    int32_t kind;
    int32_t left_bin, right_bin;   // kThreshold: last bin on the left, first bin on the right
    uint32_t best_set;             // kSubset, exhaustive search: bit k = k-th present value goes left
    int32_t best_prefix;           // kSubset, ordered search: the first best_prefix values go left
    uint32_t best_left;            // kSubset: records in the left child
    int32_t absent_left;           // kSubset: values the node does not contain go left
    int32_t t;                     // values of the attribute present in the node
    int32_t attr;                  // set by k_choose
};

struct MaskItem {
    int32_t cand;   // index of the candidate, node * d + attribute
    int32_t off;    // first word of its mask
};

__host__ __device__ inline DevCand empty_cand() {
    DevCand c;
    c.gain = c.split_info = c.gain_ratio = c.gini_split = c.gini_reduction = 0;
    c.valid = 0;
    c.kind = kLeaf;
    c.left_bin = c.right_bin = 0;
    c.best_set = 0;
    c.best_prefix = 0;
    c.best_left = 0;
    c.absent_left = 0;
    c.t = 0;
    c.attr = -1;
    return c;
}

// ------------------------------------------------------------------ kernels

// Sort key of a numeric cell (Binning::sort_key).
__global__ void k_bin_keys(const float* x, float* key, size_t n, float infinity) {
    GRID_STRIDE(i, n) key[i] = x[i] != x[i] ? infinity : x[i] + 0.0f;
}

// Bin of every cell (Binning::transform): the value code of a categorical
// attribute; for a numeric one the first bin whose largest value is not
// below the cell's key (binary search), or the last bin.
__global__ void k_bin_cells(const float* x, const int32_t* attr_numeric, const int32_t* attr_bins,
                            const int32_t* upper_off, const float* upper, uint16_t* bins, size_t n, int d,
                            float infinity) {
    GRID_STRIDE(i, n * d) {
        const size_t a = i / n;
        if (!attr_numeric[a]) {
            bins[i] = static_cast<uint16_t>(x[i]);
            continue;
        }
        const float key = x[i] != x[i] ? infinity : x[i] + 0.0f;
        const float* u = upper + upper_off[a];
        int lo = 0, hi = attr_bins[a];
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (u[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        bins[i] = static_cast<uint16_t>(lo < attr_bins[a] ? lo : attr_bins[a] - 1);
    }
}

// 1. Histograms of the nodes [f0, ...) whose records are rows[0, len):
//    hist[((node - f0) * B + attr_off[a] + bin) * C + class]. Neighboring
//    threads handle different attributes of one record, which spreads their
//    increments over different cells.
__global__ void k_histogram(Shape s, const uint16_t* bins, const int32_t* y, const int32_t* attr_off,
                            const uint32_t* rows, const uint32_t* slot, uint32_t f0, size_t len, uint32_t* hist) {
    GRID_STRIDE(i, len * s.d) {
        const size_t p = i / s.d, a = i - p * s.d;
        const uint32_t row = rows[p];
        const size_t cell =
            (static_cast<size_t>(slot[p] - f0) * s.B + attr_off[a] + bins[a * s.n + row]) * s.C + y[row];
        atomicAdd(&hist[cell], 1u);
    }
}

__device__ inline double sum_xlog2x(const double* T, const uint32_t* counts, int C) {
    double s = 0;
    for (int c = 0; c < C; ++c) s += T[counts[c]];
    return s;
}

__device__ inline double gini_of_node(const uint32_t* nc, int C, uint32_t N) {
    double sq = 0;
    for (int j = 0; j < C; ++j) sq += static_cast<double>(nc[j]) * nc[j];
    return 1.0 - sq / (static_cast<double>(N) * N);
}

// Gini index of the split that puts L[] (n_left records) on one side.
__device__ inline double gini_split(const uint32_t* L, const uint32_t* nc, int C, uint32_t N, uint32_t n_left) {
    double sq_left = 0, sq_right = 0;
    for (int j = 0; j < C; ++j) {
        const double l = L[j], r = nc[j] - L[j];
        sq_left += l * l;
        sq_right += r * r;
    }
    const double n_right = N - n_left;
    return ((n_left - sq_left / n_left) + (n_right - sq_right / n_right)) / N;
}

// 2. Best split of every (node, attribute). T[k] = k log2 k. Scratch per
//    (node, attribute): tot (records per bin), order (present bins; for an
//    ordered subset search, in the order searched) and left (class counts of
//    the left child being scored).
__global__ void k_score(Shape s, const int32_t* attr_off, const int32_t* attr_bins, const int32_t* attr_numeric,
                        const double* T, const uint32_t* node_cnt, const uint32_t* hist, uint32_t* btot,
                        uint16_t* order, uint32_t* left, DevCand* cand, size_t n_nodes) {
    GRID_STRIDE(i, n_nodes * s.d) {
        const size_t f = i / s.d;
        const int a = static_cast<int>(i - f * s.d);
        const int C = s.C, nb = attr_bins[a];
        const uint32_t* nc = node_cnt + f * C;
        const uint32_t* H = hist + (f * s.B + attr_off[a]) * C;
        uint32_t* tot = btot + f * s.B + attr_off[a];
        uint16_t* ord = order + f * s.B + attr_off[a];
        uint32_t* L = left + i * C;

        uint32_t N = 0;
        for (int j = 0; j < C; ++j) N += nc[j];
        int t = 0;
        for (int b = 0; b < nb; ++b) {
            uint32_t sum = 0;
            for (int j = 0; j < C; ++j) sum += H[static_cast<size_t>(b) * C + j];
            tot[b] = sum;
            if (sum) ord[t++] = static_cast<uint16_t>(b);
        }
        DevCand c = empty_cand();
        c.t = t;
        if (t < 2) {
            cand[i] = c;
            continue;
        }
        for (int j = 0; j < C; ++j) L[j] = 0;

        if (s.measure != 2 && !attr_numeric[a]) {
            // ---- gain / gain ratio, categorical: the multiway split
            double parts = 0, sizes = 0;
            for (int k = 0; k < t; ++k) {
                const int b = ord[k];
                parts += T[tot[b]] - sum_xlog2x(T, H + static_cast<size_t>(b) * C, C);
                sizes += T[tot[b]];
            }
            c.valid = 1;
            c.kind = kMultiway;
            c.gain = (T[N] - sum_xlog2x(T, nc, C) - parts) / N;
            c.split_info = (T[N] - sizes) / N;
            c.gain_ratio = c.gain / c.split_info;
        } else if (s.measure != 2) {
            // ---- gain / gain ratio, numeric: the threshold with the highest gain
            const double parent = T[N] - sum_xlog2x(T, nc, C);
            uint32_t n_left = 0;
            for (int k = 0; k + 1 < t; ++k) {
                const int b = ord[k];
                const uint32_t* h = H + static_cast<size_t>(b) * C;
                double s_left = 0, s_right = 0;
                for (int j = 0; j < C; ++j) {
                    L[j] += h[j];
                    s_left += T[L[j]];
                    s_right += T[nc[j] - L[j]];
                }
                n_left += tot[b];
                const uint32_t n_right = N - n_left;
                const double gain = (parent - (T[n_left] - s_left) - (T[n_right] - s_right)) / N;
                if (!c.valid || gain > c.gain + kScoreEps) {
                    c.valid = 1;
                    c.kind = kThreshold;
                    c.gain = gain;
                    c.split_info = (T[N] - T[n_left] - T[n_right]) / N;
                    c.left_bin = b;
                    c.right_bin = ord[k + 1];
                }
            }
            c.gain_ratio = c.gain / c.split_info;
        } else if (attr_numeric[a]) {
            // ---- Gini, numeric: the threshold with the largest reduction
            const double parent = gini_of_node(nc, C, N);
            uint32_t n_left = 0;
            for (int k = 0; k + 1 < t; ++k) {
                const int b = ord[k];
                const uint32_t* h = H + static_cast<size_t>(b) * C;
                for (int j = 0; j < C; ++j) L[j] += h[j];
                n_left += tot[b];
                const double g = gini_split(L, nc, C, N, n_left);
                if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                    c.valid = 1;
                    c.kind = kThreshold;
                    c.gini_split = g;
                    c.gini_reduction = parent - g;
                    c.left_bin = b;
                    c.right_bin = ord[k + 1];
                }
            }
        } else {
            // ---- Gini, categorical: the best subset of the values
            const double parent = gini_of_node(nc, C, N);
            int classes_present = 0, m = 0;
            for (int j = 0; j < C; ++j) {
                classes_present += nc[j] > 0;
                if (nc[j] > nc[m]) m = j;
            }
            if (classes_present > 2 && t <= kExhaustiveSubsetValues) {
                // Every split once: the last value always stays on the right.
                const uint32_t n_sets = 1u << (t - 1);
                for (uint32_t set = 1; set < n_sets; ++set) {
                    for (int j = 0; j < C; ++j) L[j] = 0;
                    uint32_t n_left = 0;
                    for (int k = 0; k + 1 < t; ++k) {
                        if (!((set >> k) & 1u)) continue;
                        const uint32_t* h = H + static_cast<size_t>(ord[k]) * C;
                        for (int j = 0; j < C; ++j) L[j] += h[j];
                        n_left += tot[ord[k]];
                    }
                    const double g = gini_split(L, nc, C, N, n_left);
                    if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                        c.valid = 1;
                        c.gini_split = g;
                        c.gini_reduction = parent - g;
                        c.best_set = set;
                        c.best_left = n_left;
                    }
                }
            } else {
                // Order the values by the fraction of the node's majority
                // class (exact integer comparison; ties by value code), by
                // insertion sort, then try every prefix of that order.
                for (int k = 1; k < t; ++k) {
                    const uint16_t x = ord[k];
                    const uint64_t hx = H[static_cast<size_t>(x) * C + m], tx = tot[x];
                    int p = k;
                    while (p > 0) {
                        const uint16_t w = ord[p - 1];
                        const uint64_t lhs = hx * tot[w];                               // x before w ?
                        const uint64_t rhs = static_cast<uint64_t>(H[static_cast<size_t>(w) * C + m]) * tx;
                        if (!(lhs != rhs ? lhs < rhs : x < w)) break;
                        ord[p] = w;
                        --p;
                    }
                    ord[p] = x;
                }
                uint32_t n_left = 0;
                for (int k = 0; k + 1 < t; ++k) {
                    const uint32_t* h = H + static_cast<size_t>(ord[k]) * C;
                    for (int j = 0; j < C; ++j) L[j] += h[j];
                    n_left += tot[ord[k]];
                    const double g = gini_split(L, nc, C, N, n_left);
                    if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                        c.valid = 1;
                        c.gini_split = g;
                        c.gini_reduction = parent - g;
                        c.best_prefix = k + 1;
                        c.best_left = n_left;
                    }
                }
            }
            c.kind = kSubset;
            // Values that do not occur in the node follow the larger child.
            c.absent_left = 2 * static_cast<uint64_t>(c.best_left) >= N;
        }
        cand[i] = c;
    }
}

// 3. Best attribute of every node (attr = -1: no attribute separates the records).
__global__ void k_choose(Shape s, const DevCand* cand, DevCand* chosen, size_t n_nodes) {
    GRID_STRIDE(f, n_nodes) {
        const DevCand* c = cand + f * s.d;
        int attr = -1;
        if (s.measure == 1) {
            // C4.5: the highest gain ratio among the attributes whose gain is
            // at least the average gain.
            double sum = 0;
            int valid = 0;
            for (int a = 0; a < s.d; ++a)
                if (c[a].valid) sum += c[a].gain, ++valid;
            const double average = valid ? sum / valid : 0;
            for (int a = 0; a < s.d; ++a) {
                if (!c[a].valid || c[a].gain < average - kScoreEps) continue;
                if (attr < 0 || c[a].gain_ratio > c[attr].gain_ratio + kScoreEps) attr = a;
            }
        } else {
            for (int a = 0; a < s.d; ++a) {
                if (!c[a].valid) continue;
                const double v = s.measure == 0 ? c[a].gain : c[a].gini_reduction;
                const double b = attr < 0 ? 0 : s.measure == 0 ? c[attr].gain : c[attr].gini_reduction;
                if (attr < 0 || v > b + kScoreEps) attr = a;
            }
        }
        DevCand out = attr < 0 ? empty_cand() : c[attr];
        out.attr = attr;
        chosen[f] = out;
    }
}

// Value masks (bit set = left child) of subset splits.
__global__ void k_masks(Shape s, const int32_t* attr_off, const int32_t* attr_bins, const DevCand* cand,
                        const uint16_t* order, const MaskItem* items, uint64_t* masks, size_t n_items) {
    GRID_STRIDE(i, n_items) {
        const size_t f = items[i].cand / s.d;
        const int a = items[i].cand % s.d;
        const DevCand c = cand[items[i].cand];
        const uint16_t* ord = order + f * s.B + attr_off[a];
        uint64_t* M = masks + items[i].off;
        const int n_values = attr_bins[a], words = (n_values + 63) / 64;
        for (int w = 0; w < words; ++w) M[w] = c.absent_left ? ~uint64_t(0) : uint64_t(0);
        if (c.absent_left && (n_values & 63)) M[words - 1] = (uint64_t(1) << (n_values & 63)) - 1;
        for (int k = 0; k < c.t; ++k) {
            const bool is_left = c.best_prefix > 0 ? k < c.best_prefix : ((c.best_set >> k) & 1u) != 0;
            const int v = ord[k];
            if (is_left) M[v >> 6] |= uint64_t(1) << (v & 63);
            else M[v >> 6] &= ~(uint64_t(1) << (v & 63));
        }
    }
}

// 4. New node of every record: child_base[node] + child index, or kNone if
//    the node was not split.
__global__ void k_assign(Shape s, const uint16_t* bins, const uint32_t* rows, const uint32_t* slot, uint32_t f0,
                         const DevCand* chosen, const int32_t* child_base, const int32_t* mask_off,
                         const uint64_t* masks, uint32_t* new_slot, size_t len) {
    GRID_STRIDE(p, len) {
        const uint32_t f = slot[p] - f0;
        const int32_t base = child_base[f];
        if (base < 0) {
            new_slot[p] = kNone;
            continue;
        }
        const DevCand& c = chosen[f];
        const int b = bins[static_cast<size_t>(c.attr) * s.n + rows[p]];
        int child;
        if (c.kind == kMultiway) child = b;
        else if (c.kind == kThreshold) child = b <= c.left_bin ? 0 : 1;
        else child = (masks[mask_off[f] + (b >> 6)] >> (b & 63)) & 1u ? 0 : 1;
        new_slot[p] = static_cast<uint32_t>(base + child);
    }
}

// 5. Class counts of the new nodes.
__global__ void k_count(const int32_t* y, const uint32_t* rows, const uint32_t* new_slot, uint32_t* counts, int C,
                        size_t m) {
    GRID_STRIDE(p, m) {
        if (new_slot[p] != kNone) atomicAdd(&counts[static_cast<size_t>(new_slot[p]) * C + y[rows[p]]], 1u);
    }
}

// Sort key of every record: its node in the next level, or `none` (sorts last).
__global__ void k_sort_keys(const uint32_t* new_slot, const uint32_t* remap, uint32_t none, uint32_t* key, size_t m) {
    GRID_STRIDE(p, m) key[p] = new_slot[p] == kNone ? none : remap[new_slot[p]];
}

__global__ void k_iota(uint32_t* rows, uint32_t* slot, size_t n) {
    GRID_STRIDE(i, n) {
        rows[i] = static_cast<uint32_t>(i);
        slot[i] = 0;
    }
}

// Class of every test record; x is column-major like Dataset::x.
__global__ void k_predict(const TreeNode* nodes, const uint64_t* masks, const float* x, int32_t* out, size_t n) {
    GRID_STRIDE(i, n) {
        const TreeNode* node = nodes;
        while (node->kind != kLeaf) {
            const float v = x[static_cast<size_t>(node->attr) * n + i];
            if (node->kind == kThreshold) {
                node = nodes + node->child + (v <= node->threshold ? 0 : 1);
            } else {
                const int code = static_cast<int>(v);
                if (code < 0 || code >= node->n_values) break;   // value unknown to the tree
                if (node->kind == kMultiway) node = nodes + node->child + code;
                else node = nodes + node->child + ((masks[node->mask + (code >> 6)] >> (code & 63)) & 1u ? 0 : 1);
            }
        }
        out[i] = node->label;
    }
}

// --------------------------------------------------------------------- host

class GpuTreeBuilder {
public:
    GpuTreeBuilder(const Dataset& train, const TreeOptions& opt) : data_(train), opt_(opt) {
        const size_t n = train.num_rows();
        const int d = train.num_attrs();
        shape_.n = n;
        shape_.d = d;
        shape_.C = train.num_classes();
        shape_.measure = opt.measure == Measure::Gain ? 0 : opt.measure == Measure::GainRatio ? 1 : 2;
        d_y_.upload(train.y);
        bin_attributes();

        std::vector<double> xlog2x(n + 1);
        xlog2x[0] = 0;
        for (size_t k = 1; k <= n; ++k) xlog2x[k] = static_cast<double>(k) * std::log2(static_cast<double>(k));
        d_xlog2x_.upload(xlog2x);
    }

    DecisionTree build(std::vector<AttrScore>* root_scores) {
        const size_t n = shape_.n;
        const int d = shape_.d, C = shape_.C, B = shape_.B;
        DecisionTree tree;
        tree.n_classes = C;
        add_nodes(tree, 1, 0);

        // The current level: tree node, class counts and record range of
        // every node that may still be split.
        std::vector<int32_t> f_node{0};
        std::vector<uint32_t> f_cnt(C, 0);
        std::vector<size_t> f_begin{0, n};
        for (int32_t c : data_.y) ++f_cnt[c];
        bool root_stop = !finish_node(tree, 0, f_cnt.data(), 0);
        if (root_stop && !root_scores) return tree;

        size_t m = n;   // records of the level
        d_rows_.reserve(n);
        d_slot_.reserve(n);
        d_new_slot_.reserve(n);
        d_key_.reserve(n);
        d_rows2_.reserve(n);
        d_slot2_.reserve(n);
        gpu::launch(k_iota, n, d_rows_.get(), d_slot_.get(), n);

        // Nodes whose histograms fit in the budget together.
        const size_t node_bytes = static_cast<size_t>(B) * C * sizeof(uint32_t);
        const size_t max_chunk = std::max<size_t>(1, g_hist_budget / node_bytes);

        std::vector<DevCand> chosen;
        std::vector<int32_t> child_base, mask_off;
        std::vector<MaskItem> items;
        std::vector<uint64_t> mask_words;
        std::vector<int32_t> child_node, child_parent_label;
        std::vector<uint32_t> child_cnt, remap;

        while (!f_node.empty()) {
            const size_t F = f_node.size();
            d_node_cnt_.upload(f_cnt);
            child_node.clear();
            child_parent_label.clear();

            for (size_t f0 = 0; f0 < F; f0 += max_chunk) {
                const size_t fc = std::min(max_chunk, F - f0);
                const size_t p0 = f_begin[f0], len = f_begin[f0 + fc] - p0;

                // 1-3: histograms, best split per attribute, best attribute.
                d_hist_.zero(fc * B * C);
                d_btot_.reserve(fc * B);
                d_order_.reserve(fc * B);
                d_left_.reserve(fc * d * C);
                d_cand_.reserve(fc * d);
                d_chosen_.reserve(fc);
                gpu::launch(k_histogram, len * d, shape_, d_bins_.get(), d_y_.get(), d_attr_off_.get(),
                            d_rows_.get() + p0, d_slot_.get() + p0, static_cast<uint32_t>(f0), len, d_hist_.get());
                gpu::launch(k_score, fc * d, shape_, d_attr_off_.get(), d_attr_bins_.get(), d_attr_numeric_.get(),
                            d_xlog2x_.get(), d_node_cnt_.get() + f0 * C, d_hist_.get(), d_btot_.get(),
                            d_order_.get(), d_left_.get(), d_cand_.get(), fc);
                gpu::launch(k_choose, fc, shape_, d_cand_.get(), d_chosen_.get(), fc);
                d_chosen_.download(chosen, fc);

                if (root_scores) {
                    report_root(*root_scores);
                    root_scores = nullptr;
                    if (root_stop) return tree;
                }

                // The host adds the children to the tree.
                child_base.assign(fc, -1);
                mask_off.assign(fc, 0);
                items.clear();
                int32_t n_mask_words = 0;
                const int32_t masks_before = static_cast<int32_t>(tree.masks.size());
                for (size_t i = 0; i < fc; ++i) {
                    const DevCand& c = chosen[i];
                    if (c.attr < 0) continue;   // no attribute separates the records: the node stays a leaf
                    const int32_t parent = f_node[f0 + i];
                    const int n_children = c.kind == kMultiway ? binning_.n_bins[c.attr] : 2;
                    const int32_t first = add_nodes(tree, n_children, tree.nodes[parent].depth + 1);
                    TreeNode& node = tree.nodes[parent];
                    node.kind = static_cast<uint8_t>(c.kind);
                    node.attr = c.attr;
                    node.child = first;
                    if (c.kind == kThreshold) {
                        node.threshold = binning_.threshold(c.attr, c.left_bin, c.right_bin);
                    } else {
                        node.n_values = binning_.n_bins[c.attr];
                        if (c.kind == kSubset) {
                            mask_off[i] = n_mask_words;
                            node.mask = masks_before + n_mask_words;
                            items.push_back({static_cast<int32_t>(i * d + c.attr), n_mask_words});
                            n_mask_words += DecisionTree::mask_words(node.n_values);
                        }
                    }
                    child_base[i] = static_cast<int32_t>(child_node.size());
                    for (int k = 0; k < n_children; ++k) {
                        child_node.push_back(first + k);
                        child_parent_label.push_back(node.label);
                    }
                }
                if (!items.empty()) {
                    download_masks(items, n_mask_words, mask_words);
                    tree.masks.insert(tree.masks.end(), mask_words.begin(), mask_words.end());
                }

                // 4: every record moves to a child.
                d_child_base_.upload(child_base);
                d_mask_off_.upload(mask_off);
                gpu::launch(k_assign, len, shape_, d_bins_.get(), d_rows_.get() + p0, d_slot_.get() + p0,
                            static_cast<uint32_t>(f0), d_chosen_.get(), d_child_base_.get(), d_mask_off_.get(),
                            d_masks_.get(), d_new_slot_.get() + p0, len);
            }

            // 5: class counts of the children; the host closes the leaves.
            const size_t K = child_node.size();
            if (K == 0) break;
            d_child_cnt_.zero(K * C);
            gpu::launch(k_count, m, d_y_.get(), d_rows_.get(), d_new_slot_.get(), d_child_cnt_.get(), C, m);
            d_child_cnt_.download(child_cnt, K * C);

            f_node.clear();
            f_cnt.clear();
            f_begin.assign(1, 0);
            remap.assign(K, kNone);
            for (size_t k = 0; k < K; ++k) {
                const uint32_t* cnt = &child_cnt[k * C];
                if (!finish_node(tree, child_node[k], cnt, child_parent_label[k])) continue;
                remap[k] = static_cast<uint32_t>(f_node.size());
                f_node.push_back(child_node[k]);
                f_cnt.insert(f_cnt.end(), cnt, cnt + C);
                f_begin.push_back(f_begin.back() + tree.nodes[child_node[k]].n);
            }
            if (f_node.empty()) break;

            // Sort the records by their node in the next level; the records
            // of finished leaves sort to the end and are dropped.
            const uint32_t none = static_cast<uint32_t>(f_node.size());
            for (uint32_t& r : remap)
                if (r == kNone) r = none;
            int bits = 1;
            while ((uint64_t(1) << bits) <= none) ++bits;
            d_remap_.upload(remap);
            gpu::launch(k_sort_keys, m, d_new_slot_.get(), d_remap_.get(), none, d_key_.get(), m);
            prim_.sort_pairs(d_key_.get(), d_slot2_.get(), d_rows_.get(), d_rows2_.get(), m, bits);
            d_slot_.swap(d_slot2_);
            d_rows_.swap(d_rows2_);
            m = f_begin.back();
        }
        gpu::synchronize();
        return tree;
    }

private:
    // Binning on the device (the same bins as Binning::fit): every numeric
    // column is sorted and reduced to its distinct values and their counts,
    // from which the host lays out the bins; then every cell is mapped to
    // its bin.
    void bin_attributes() {
        const size_t n = shape_.n;
        const int d = shape_.d;
        gpu::DevBuf<float> d_x, d_key(n), d_sorted(n), d_values(n), d_upper;
        gpu::DevBuf<uint32_t> d_counts(n);
        std::vector<float> values;
        std::vector<uint32_t> counts;
        d_x.upload(data_.x);
        binning_.init(data_);
        for (int a = 0; a < d; ++a) {
            if (!data_.attrs[a].numeric) continue;
            gpu::launch(k_bin_keys, n, d_x.get() + static_cast<size_t>(a) * n, d_key.get(), n,
                        std::numeric_limits<float>::infinity());
            prim_.sort_keys(d_key.get(), d_sorted.get(), n);
            const size_t distinct = prim_.run_length_encode(d_sorted.get(), d_values.get(), d_counts.get(), n);
            d_values.download(values, distinct);
            d_counts.download(counts, distinct);
            binning_.set_numeric(a, values.data(), counts.data(), distinct, n, opt_.max_bins);
        }

        attr_off_.resize(d);
        attr_numeric_.resize(d);
        std::vector<int32_t> upper_off(d, 0);
        std::vector<float> upper;   // largest value of every bin, all numeric attributes
        int total = 0;
        for (int a = 0; a < d; ++a) {
            attr_off_[a] = total;
            attr_numeric_[a] = data_.attrs[a].numeric;
            total += binning_.n_bins[a];
            upper_off[a] = static_cast<int32_t>(upper.size());
            upper.insert(upper.end(), binning_.hi[a].begin(), binning_.hi[a].end());
        }
        shape_.B = std::max(total, 1);
        d_attr_off_.upload(attr_off_);
        d_attr_bins_.upload(binning_.n_bins);
        d_attr_numeric_.upload(attr_numeric_);
        d_upper.upload(upper);
        gpu::DevBuf<int32_t> d_upper_off;
        d_upper_off.upload(upper_off);
        d_bins_.reserve(n * d);
        gpu::launch(k_bin_cells, n * d, d_x.get(), d_attr_numeric_.get(), d_attr_bins_.get(), d_upper_off.get(),
                    d_upper.get(), d_bins_.get(), n, d, std::numeric_limits<float>::infinity());
    }

    int32_t add_nodes(DecisionTree& tree, int count, int depth) const {
        const int32_t first = static_cast<int32_t>(tree.nodes.size());
        TreeNode node;
        node.depth = depth;
        tree.nodes.resize(tree.nodes.size() + count, node);
        tree.counts.resize(tree.nodes.size() * shape_.C, 0);
        return first;
    }

    // Records the class counts and the label of a node; returns whether the
    // node may be split. A node without records (an empty branch of a
    // multiway split) gets the majority class of its parent.
    bool finish_node(DecisionTree& tree, int32_t index, const uint32_t* cnt, int32_t parent_label) const {
        const int C = shape_.C;
        TreeNode& node = tree.nodes[index];
        uint32_t total = 0;
        int32_t label = 0;
        for (int c = 0; c < C; ++c) {
            total += cnt[c];
            if (cnt[c] > cnt[label]) label = c;
        }
        std::copy(cnt, cnt + C, tree.counts.begin() + static_cast<size_t>(index) * C);
        node.n = total;
        node.label = total ? label : parent_label;
        if (total == 0) return false;
        return !(cnt[label] == total || total < opt_.min_split ||
                 (opt_.max_depth > 0 && node.depth >= opt_.max_depth));
    }

    // Builds the masks of `items` on the device and copies them to the host.
    void download_masks(const std::vector<MaskItem>& items, int32_t n_words, std::vector<uint64_t>& out) {
        d_items_.upload(items);
        d_masks_.reserve(n_words);
        gpu::launch(k_masks, items.size(), shape_, d_attr_off_.get(), d_attr_bins_.get(), d_cand_.get(),
                    d_order_.get(), d_items_.get(), d_masks_.get(), items.size());
        d_masks_.download(out, n_words);
    }

    // Best split of every attribute at the root, from the candidates of node 0.
    void report_root(std::vector<AttrScore>& scores) {
        const int d = shape_.d;
        std::vector<DevCand> cand;
        d_cand_.download(cand, d);
        std::vector<MaskItem> items;
        std::vector<int32_t> off(d, 0);
        int32_t n_words = 0;
        for (int a = 0; a < d; ++a) {
            if (!cand[a].valid || cand[a].kind != kSubset) continue;
            off[a] = n_words;
            items.push_back({a, n_words});
            n_words += DecisionTree::mask_words(binning_.n_bins[a]);
        }
        std::vector<uint64_t> masks;
        if (!items.empty()) download_masks(items, n_words, masks);
        scores.clear();
        for (int a = 0; a < d; ++a) {
            const DevCand& c = cand[a];
            AttrScore s;
            s.attr = a;
            s.valid = c.valid != 0;
            s.gain = c.gain;
            s.split_info = c.split_info;
            s.gain_ratio = c.gain_ratio;
            s.gini_split = c.gini_split;
            s.gini_reduction = c.gini_reduction;
            if (s.valid) {
                const float t = c.kind == kThreshold ? binning_.threshold(a, c.left_bin, c.right_bin) : 0.0f;
                s.split = describe_split(data_.attrs[a], static_cast<uint8_t>(c.kind), t,
                                         c.kind == kSubset ? masks.data() + off[a] : nullptr);
            }
            scores.push_back(std::move(s));
        }
    }

    const Dataset& data_;
    const TreeOptions opt_;
    Shape shape_;
    Binning binning_;
    std::vector<int32_t> attr_off_, attr_numeric_;
    gpu::Primitives prim_;

    // Training table.
    gpu::DevBuf<uint16_t> d_bins_;          // [a * n + record]
    gpu::DevBuf<int32_t> d_y_;
    gpu::DevBuf<int32_t> d_attr_off_, d_attr_bins_, d_attr_numeric_;
    gpu::DevBuf<double> d_xlog2x_;          // k log2 k for k = 0 .. n
    // Records of the level, grouped by node.
    gpu::DevBuf<uint32_t> d_rows_, d_slot_, d_rows2_, d_slot2_, d_new_slot_, d_key_;
    // Per range of nodes.
    gpu::DevBuf<uint32_t> d_node_cnt_, d_hist_, d_btot_, d_left_;
    gpu::DevBuf<uint16_t> d_order_;
    gpu::DevBuf<DevCand> d_cand_, d_chosen_;
    gpu::DevBuf<int32_t> d_child_base_, d_mask_off_;
    gpu::DevBuf<MaskItem> d_items_;
    gpu::DevBuf<uint64_t> d_masks_;
    // Per level.
    gpu::DevBuf<uint32_t> d_child_cnt_, d_remap_;
};

std::vector<int32_t> predict(const DecisionTree& tree, const Dataset& test) {
    const size_t n = test.num_rows();
    std::vector<int32_t> out(n);
    if (n == 0) return out;
    gpu::DevBuf<TreeNode> d_nodes;
    gpu::DevBuf<uint64_t> d_masks;
    gpu::DevBuf<float> d_x;
    gpu::DevBuf<int32_t> d_out(n);
    d_nodes.upload(tree.nodes);
    d_masks.upload(tree.masks);
    d_x.upload(test.x);
    gpu::launch(k_predict, n, d_nodes.get(), d_masks.get(), d_x.get(), d_out.get(), n);
    d_out.download(out.data(), n);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    TreeBackend backend;
    backend.device.id = gpu::device_id();
    backend.device.name = gpu::device_name();
    backend.device.peak_bytes = &gpu::g_peak_bytes;
    backend.build = [](const Dataset& train, const TreeOptions& opt, std::vector<AttrScore>* root_scores) {
        return GpuTreeBuilder(train, opt).build(root_scores);
    };
    backend.predict = predict;
    backend.parse_options = [](Args& args) {
        double mb = 256;
        args.value("--hist-mb", mb);
        if (!(mb >= 0 && mb <= 65536)) throw std::invalid_argument("--hist-mb out of range");
        g_hist_budget = static_cast<size_t>(mb * 1048576.0);
    };
    backend.options_help =
        "  --hist-mb M          device memory for the histograms of one pass (default 256);\n"
        "                       a level that needs more is processed in several passes\n";
    gpu::warm_up();
    return dtree_main(argc, argv, backend);
}
