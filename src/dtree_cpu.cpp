// Decision tree induction on the CPU (Han & Kamber, "Generate_decision_tree"),
// with information gain, gain ratio or the Gini index as the attribute
// selection measure.
//
// The tree is grown depth first. The records of a node are a contiguous
// range of one index array, which is partitioned in place when the node is
// split. For every attribute the node's records are counted into a
// (value, class) histogram, and every candidate split of the attribute is
// scored from that histogram alone:
//
//   gain / gain ratio
//     categorical  the multiway split on all values
//     numeric      the threshold with the highest gain
//   Gini index
//     categorical  the best subset split (see kExhaustiveSubsetValues)
//     numeric      the threshold with the largest reduction in impurity
//
// A node becomes a leaf labeled with its majority class when it is pure, when
// no attribute separates its records, or when a stopping option applies. A
// multiway branch that receives no records becomes a leaf labeled with the
// majority class of its parent.
//
// usage: dtree_cpu <table.csv> [--measure gain|gain_ratio|gini] [options]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "dtree.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

namespace {

using namespace ml;

// Best split of one attribute at one node.
struct Candidate {
    bool valid = false;
    uint8_t kind = kLeaf;
    double gain = 0, split_info = 0, gain_ratio = 0;
    double gini_split = 0, gini_reduction = 0;
    int left_bin = 0, right_bin = 0;    // kThreshold: last bin on the left, first bin on the right
    std::vector<uint64_t> mask;         // kSubset: values of the left child
    bool absent_left = false;           // kSubset: values the node does not contain go left
};

class TreeBuilder {
public:
    TreeBuilder(const Dataset& train, const TreeOptions& opt)
        : data_(train), opt_(opt), n_(train.num_rows()), n_classes_(train.num_classes()) {
        binning_.fit(train, opt.max_bins);
        bins_ = binning_.transform(train);
        xlog2x_.resize(n_ + 1);
        xlog2x_[0] = 0;
        for (size_t k = 1; k <= n_; ++k) xlog2x_[k] = static_cast<double>(k) * std::log2(static_cast<double>(k));
        const size_t max_bins = static_cast<size_t>(std::max(1, binning_.max_n_bins));
        hist_.assign(max_bins * n_classes_, 0);
        bin_total_.assign(max_bins, 0);
        rows_.resize(n_);
        for (size_t i = 0; i < n_; ++i) rows_[i] = static_cast<uint32_t>(i);
        scratch_.resize(n_);
        left_.resize(n_classes_);
        node_counts_.resize(n_classes_);
    }

    // root_scores (optional) receives the best split of every attribute at the root.
    DecisionTree build(std::vector<AttrScore>* root_scores = nullptr) {
        DecisionTree tree;
        tree.n_classes = n_classes_;
        struct Work {
            int32_t node;
            uint32_t begin, end;
        };
        std::vector<Work> stack;
        add_nodes(tree, 1, 0);
        stack.push_back({0, 0, static_cast<uint32_t>(n_)});
        std::vector<uint32_t> child_size;

        while (!stack.empty()) {
            const Work w = stack.back();
            stack.pop_back();
            const uint32_t m = w.end - w.begin;

            std::fill(node_counts_.begin(), node_counts_.end(), 0u);
            for (uint32_t i = w.begin; i < w.end; ++i) ++node_counts_[data_.y[rows_[i]]];
            const int32_t label = majority(node_counts_.data());
            {
                TreeNode& node = tree.nodes[w.node];
                node.n = m;
                node.label = label;
                std::copy(node_counts_.begin(), node_counts_.end(), tree.counts.begin() + w.node * n_classes_);
            }
            const int depth = tree.nodes[w.node].depth;
            const bool is_root = w.node == 0;
            const bool stop = node_counts_[label] == m || m < opt_.min_split ||
                              (opt_.max_depth > 0 && depth >= opt_.max_depth);
            if (stop && !(is_root && root_scores)) continue;

            int best_attr = -1;
            Candidate best = choose_split(w.begin, w.end, best_attr, is_root ? root_scores : nullptr);
            if (stop || best_attr < 0) continue;

            // ---- split the node
            const uint16_t* col = column(best_attr);
            const int n_children = best.kind == kMultiway ? binning_.n_bins[best_attr] : 2;
            const int32_t first = add_nodes(tree, n_children, depth + 1);
            {
                TreeNode& node = tree.nodes[w.node];
                node.kind = best.kind;
                node.attr = best_attr;
                node.child = first;
                if (best.kind == kThreshold) {
                    node.threshold = binning_.threshold(best_attr, best.left_bin, best.right_bin);
                } else {
                    node.n_values = binning_.n_bins[best_attr];
                    if (best.kind == kSubset) {
                        node.mask = static_cast<int32_t>(tree.masks.size());
                        tree.masks.insert(tree.masks.end(), best.mask.begin(), best.mask.end());
                    }
                }
            }
            auto child_of = [&](uint32_t row) -> int {
                const int b = col[row];
                if (best.kind == kMultiway) return b;
                if (best.kind == kThreshold) return b <= best.left_bin ? 0 : 1;
                return (best.mask[b >> 6] >> (b & 63)) & 1u ? 0 : 1;
            };
            // Stable counting sort of the node's records by child.
            child_size.assign(n_children, 0);
            for (uint32_t i = w.begin; i < w.end; ++i) ++child_size[child_of(rows_[i])];
            std::vector<uint32_t>& offset = offsets_;
            offset.assign(n_children + 1, w.begin);
            for (int c = 0; c < n_children; ++c) offset[c + 1] = offset[c] + child_size[c];
            {
                std::vector<uint32_t>& cursor = cursor_;
                cursor.assign(offset.begin(), offset.end() - 1);
                for (uint32_t i = w.begin; i < w.end; ++i) {
                    const uint32_t row = rows_[i];
                    scratch_[cursor[child_of(row)]++] = row;
                }
                std::copy(scratch_.begin() + w.begin, scratch_.begin() + w.end, rows_.begin() + w.begin);
            }
            // Children are pushed in reverse so that they are expanded in order.
            for (int c = n_children - 1; c >= 0; --c) {
                if (child_size[c] == 0) {
                    tree.nodes[first + c].label = label;   // empty branch: majority class of the parent
                    continue;
                }
                stack.push_back({first + c, offset[c], offset[c + 1]});
            }
        }
        return tree;
    }

    std::string describe_split(int attr, const Candidate& c) const {
        const Attribute& a = data_.attrs[attr];
        if (c.kind == kMultiway) return "multiway";
        if (c.kind == kThreshold) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "<= %.9g", static_cast<double>(binning_.threshold(attr, c.left_bin, c.right_bin)));
            return buf;
        }
        std::string s;
        for (int v = 0; v < a.num_values(); ++v)
            if ((c.mask[v >> 6] >> (v & 63)) & 1u) s += (s.empty() ? "" : ", ") + a.values[v];
        return "in {" + s + "}";
    }

private:
    const uint16_t* column(int a) const { return bins_.data() + static_cast<size_t>(a) * n_; }

    int32_t majority(const uint32_t* counts) const {
        int32_t best = 0;
        for (int c = 1; c < n_classes_; ++c)
            if (counts[c] > counts[best]) best = c;
        return best;
    }

    int32_t add_nodes(DecisionTree& tree, int count, int depth) const {
        const int32_t first = static_cast<int32_t>(tree.nodes.size());
        TreeNode node;
        node.depth = depth;
        tree.nodes.resize(tree.nodes.size() + count, node);
        tree.counts.resize(tree.nodes.size() * n_classes_, 0);
        return first;
    }

    double sum_xlog2x(const uint32_t* counts) const {
        double s = 0;
        for (int c = 0; c < n_classes_; ++c) s += xlog2x_[counts[c]];
        return s;
    }

    // Best split of the node rows_[begin, end) over all attributes; attr = -1
    // if no attribute separates the records. Uses node_counts_.
    Candidate choose_split(uint32_t begin, uint32_t end, int& attr, std::vector<AttrScore>* scores) {
        const int d = data_.num_attrs();
        std::vector<Candidate> cand(d);
        for (int a = 0; a < d; ++a) cand[a] = score_attribute(a, begin, end);

        attr = -1;
        if (opt_.measure == Measure::GainRatio) {
            // C4.5: the highest gain ratio among the attributes whose gain is
            // at least the average gain, which keeps a split with a tiny
            // split information from winning on the ratio alone.
            double sum = 0;
            int valid = 0;
            for (int a = 0; a < d; ++a)
                if (cand[a].valid) sum += cand[a].gain, ++valid;
            const double average = valid ? sum / valid : 0;
            for (int a = 0; a < d; ++a) {
                if (!cand[a].valid || cand[a].gain < average - kScoreEps) continue;
                if (attr < 0 || cand[a].gain_ratio > cand[attr].gain_ratio + kScoreEps) attr = a;
            }
        } else {
            for (int a = 0; a < d; ++a) {
                if (!cand[a].valid) continue;
                const double s = opt_.measure == Measure::Gain ? cand[a].gain : cand[a].gini_reduction;
                const double b = attr < 0 ? 0 : opt_.measure == Measure::Gain ? cand[attr].gain : cand[attr].gini_reduction;
                if (attr < 0 || s > b + kScoreEps) attr = a;
            }
        }
        if (scores) {
            scores->clear();
            for (int a = 0; a < d; ++a) {
                AttrScore s;
                s.attr = a;
                s.valid = cand[a].valid;
                s.gain = cand[a].gain;
                s.split_info = cand[a].split_info;
                s.gain_ratio = cand[a].gain_ratio;
                s.gini_split = cand[a].gini_split;
                s.gini_reduction = cand[a].gini_reduction;
                if (s.valid) s.split = describe_split(a, cand[a]);
                scores->push_back(std::move(s));
            }
        }
        return attr < 0 ? Candidate() : std::move(cand[attr]);
    }

    Candidate score_attribute(int a, uint32_t begin, uint32_t end) {
        const uint16_t* col = column(a);
        const int C = n_classes_;
        // (value, class) histogram of the node; `present_` lists the bins that occur.
        present_.clear();
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t row = rows_[i];
            const uint32_t b = col[row];
            if (bin_total_[b]++ == 0) present_.push_back(static_cast<uint16_t>(b));
            ++hist_[static_cast<size_t>(b) * C + data_.y[row]];
        }
        Candidate c;
        if (present_.size() >= 2) {
            std::sort(present_.begin(), present_.end());
            const uint32_t N = end - begin;
            const bool numeric = data_.attrs[a].numeric;
            if (opt_.measure == Measure::Gini) {
                if (numeric) gini_threshold(N, c);
                else gini_subset(a, N, c);
            } else {
                if (numeric) entropy_threshold(N, c);
                else entropy_multiway(N, c);
            }
        }
        for (uint16_t b : present_) {
            bin_total_[b] = 0;
            std::fill_n(hist_.begin() + static_cast<size_t>(b) * C, C, 0u);
        }
        return c;
    }

    // ---- information gain / gain ratio.  With T(k) = k log2 k:
    //   N * Info(D)            = T(N) - sum_c T(n_c)
    //   N * Info_A(D)          = sum_j [ T(N_j) - sum_c T(n_jc) ]
    //   N * SplitInfo_A(D)     = T(N) - sum_j T(N_j)

    void entropy_multiway(uint32_t N, Candidate& c) const {
        double parts = 0, sizes = 0;
        for (uint16_t b : present_) {
            parts += xlog2x_[bin_total_[b]] - sum_xlog2x(&hist_[static_cast<size_t>(b) * n_classes_]);
            sizes += xlog2x_[bin_total_[b]];
        }
        c.valid = true;
        c.kind = kMultiway;
        c.gain = (xlog2x_[N] - sum_xlog2x(node_counts_.data()) - parts) / N;
        c.split_info = (xlog2x_[N] - sizes) / N;
        c.gain_ratio = c.gain / c.split_info;
    }

    void entropy_threshold(uint32_t N, Candidate& c) {
        const int C = n_classes_;
        const double parent = xlog2x_[N] - sum_xlog2x(node_counts_.data());
        std::fill(left_.begin(), left_.end(), 0u);
        uint32_t n_left = 0;
        for (size_t k = 0; k + 1 < present_.size(); ++k) {
            const uint16_t b = present_[k];
            const uint32_t* h = &hist_[static_cast<size_t>(b) * C];
            double s_left = 0, s_right = 0;
            for (int j = 0; j < C; ++j) {
                left_[j] += h[j];
                s_left += xlog2x_[left_[j]];
                s_right += xlog2x_[node_counts_[j] - left_[j]];
            }
            n_left += bin_total_[b];
            const uint32_t n_right = N - n_left;
            const double gain = (parent - (xlog2x_[n_left] - s_left) - (xlog2x_[n_right] - s_right)) / N;
            if (!c.valid || gain > c.gain + kScoreEps) {
                c.valid = true;
                c.kind = kThreshold;
                c.gain = gain;
                c.split_info = (xlog2x_[N] - xlog2x_[n_left] - xlog2x_[n_right]) / N;
                c.left_bin = b;
                c.right_bin = present_[k + 1];
            }
        }
        c.gain_ratio = c.gain / c.split_info;
    }

    // ---- Gini index.  Gini(D) = 1 - sum_c (n_c / N)^2, and for a binary split
    //   N * Gini_A(D) = [N_L - sum_c l_c^2 / N_L] + [N_R - sum_c r_c^2 / N_R]

    double gini_of_node(uint32_t N) const {
        double sq = 0;
        for (int j = 0; j < n_classes_; ++j) sq += static_cast<double>(node_counts_[j]) * node_counts_[j];
        return 1.0 - sq / (static_cast<double>(N) * N);
    }

    // Gini index of the split that puts left_[] (n_left records) on one side.
    double gini_split(uint32_t N, uint32_t n_left) const {
        double sq_left = 0, sq_right = 0;
        for (int j = 0; j < n_classes_; ++j) {
            const double l = left_[j], r = node_counts_[j] - left_[j];
            sq_left += l * l;
            sq_right += r * r;
        }
        const double n_right = N - n_left;
        return ((n_left - sq_left / n_left) + (n_right - sq_right / n_right)) / N;
    }

    void gini_threshold(uint32_t N, Candidate& c) {
        const int C = n_classes_;
        const double parent = gini_of_node(N);
        std::fill(left_.begin(), left_.end(), 0u);
        uint32_t n_left = 0;
        for (size_t k = 0; k + 1 < present_.size(); ++k) {
            const uint16_t b = present_[k];
            const uint32_t* h = &hist_[static_cast<size_t>(b) * C];
            for (int j = 0; j < C; ++j) left_[j] += h[j];
            n_left += bin_total_[b];
            const double g = gini_split(N, n_left);
            if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                c.valid = true;
                c.kind = kThreshold;
                c.gini_split = g;
                c.gini_reduction = parent - g;
                c.left_bin = b;
                c.right_bin = present_[k + 1];
            }
        }
    }

    void gini_subset(int a, uint32_t N, Candidate& c) {
        const int C = n_classes_;
        const double parent = gini_of_node(N);
        const int t = static_cast<int>(present_.size());
        int classes_present = 0;
        for (int j = 0; j < C; ++j) classes_present += node_counts_[j] > 0;

        // `order_` lists the present values; the left child is order_[k] for
        // every k with bit k of `best_set` set (exhaustive search), or the
        // first `best_prefix` values (ordered search).
        order_.assign(present_.begin(), present_.end());
        uint32_t best_set = 0;
        int best_prefix = 0;
        uint32_t best_left = 0;
        if (classes_present > 2 && t <= kExhaustiveSubsetValues) {
            // Every split once: the last value always stays on the right.
            const uint32_t n_sets = 1u << (t - 1);
            for (uint32_t set = 1; set < n_sets; ++set) {
                std::fill(left_.begin(), left_.end(), 0u);
                uint32_t n_left = 0;
                for (int k = 0; k + 1 < t; ++k) {
                    if (!((set >> k) & 1u)) continue;
                    const uint32_t* h = &hist_[static_cast<size_t>(order_[k]) * C];
                    for (int j = 0; j < C; ++j) left_[j] += h[j];
                    n_left += bin_total_[order_[k]];
                }
                const double g = gini_split(N, n_left);
                if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                    c.valid = true;
                    c.gini_split = g;
                    c.gini_reduction = parent - g;
                    best_set = set;
                    best_left = n_left;
                }
            }
        } else {
            // Order the values by the fraction of the node's majority class
            // (exact integer comparison; ties by value code).
            const int32_t m = majority(node_counts_.data());
            std::sort(order_.begin(), order_.end(), [&](uint16_t x, uint16_t y) {
                const uint64_t lhs = static_cast<uint64_t>(hist_[static_cast<size_t>(x) * C + m]) * bin_total_[y];
                const uint64_t rhs = static_cast<uint64_t>(hist_[static_cast<size_t>(y) * C + m]) * bin_total_[x];
                return lhs != rhs ? lhs < rhs : x < y;
            });
            std::fill(left_.begin(), left_.end(), 0u);
            uint32_t n_left = 0;
            for (int k = 0; k + 1 < t; ++k) {
                const uint32_t* h = &hist_[static_cast<size_t>(order_[k]) * C];
                for (int j = 0; j < C; ++j) left_[j] += h[j];
                n_left += bin_total_[order_[k]];
                const double g = gini_split(N, n_left);
                if (!c.valid || parent - g > c.gini_reduction + kScoreEps) {
                    c.valid = true;
                    c.gini_split = g;
                    c.gini_reduction = parent - g;
                    best_prefix = k + 1;
                    best_left = n_left;
                }
            }
        }

        c.kind = kSubset;
        const int n_values = binning_.n_bins[a];
        // Values that do not occur in the node follow the larger child.
        c.absent_left = 2 * static_cast<uint64_t>(best_left) >= N;
        c.mask.assign(DecisionTree::mask_words(n_values), c.absent_left ? ~uint64_t(0) : 0);
        if (c.absent_left && (n_values & 63)) c.mask.back() = (uint64_t(1) << (n_values & 63)) - 1;
        for (int k = 0; k < t; ++k) {
            const bool left = best_prefix > 0 ? k < best_prefix : ((best_set >> k) & 1u) != 0;
            const uint16_t v = order_[k];
            if (left) c.mask[v >> 6] |= uint64_t(1) << (v & 63);
            else c.mask[v >> 6] &= ~(uint64_t(1) << (v & 63));
        }
    }

    const Dataset& data_;
    const TreeOptions opt_;
    const size_t n_;
    const int n_classes_;
    Binning binning_;
    std::vector<uint16_t> bins_;          // bins_[a * n + i]
    std::vector<double> xlog2x_;          // k log2 k for k = 0 .. n
    std::vector<uint32_t> rows_;          // record indices, grouped by node
    std::vector<uint32_t> scratch_;
    std::vector<uint32_t> hist_;          // hist_[bin * n_classes + class], zero between attributes
    std::vector<uint32_t> bin_total_;     // records per bin, zero between attributes
    std::vector<uint16_t> present_;       // bins that occur in the node, ascending
    std::vector<uint16_t> order_;
    std::vector<uint32_t> left_;          // class counts of the left child being scored
    std::vector<uint32_t> node_counts_;   // class counts of the node being split
    std::vector<uint32_t> offsets_, cursor_;
};

DecisionTree build_tree(const Dataset& train, const TreeOptions& opt, std::vector<AttrScore>* root_scores = nullptr) {
    TreeBuilder builder(train, opt);
    DecisionTree tree = builder.build(root_scores);
    if (opt.prune_cf > 0) tree.prune(opt.prune_cf);
    return tree;
}

class DecisionTreeCpu : public Classifier {
public:
    explicit DecisionTreeCpu(const TreeOptions& opt) : opt_(opt) {}
    std::string name() const override { return std::string("dtree(") + measure_name(opt_.measure) + ")"; }
    void fit(const Dataset& train) override { tree_ = build_tree(train, opt_); }
    std::vector<int32_t> predict(const Dataset& test) override { return tree_.predict(test); }
    std::vector<std::pair<std::string, double>> info() const override {
        const TreeStats s = tree_.stats();
        return {{"tree_nodes", static_cast<double>(s.nodes)},
                {"tree_leaves", static_cast<double>(s.leaves)},
                {"tree_depth", static_cast<double>(s.depth)}};
    }

private:
    TreeOptions opt_;
    DecisionTree tree_;
};

[[noreturn]] void usage(const char* prog, const std::string& problem = "") {
    if (!problem.empty()) std::fprintf(stderr, "error: %s\n", problem.c_str());
    std::fprintf(stderr,
                 "usage: %s <table.csv> [options]\n"
                 "  --measure M          gain (default), gain_ratio or gini\n"
                 "  --max-depth D        stop splitting at depth D (default: no limit)\n"
                 "  --min-split N        do not split nodes with fewer than N records (default 2)\n"
                 "  --max-bins B         candidate thresholds per numeric attribute: one per distinct\n"
                 "                       value up to B values, B equal-frequency bins beyond that\n"
                 "                       (default 256; 0 = always one per distinct value)\n"
                 "  --prune              pessimistic post-pruning with confidence 0.25\n"
                 "  --prune-cf CF        pessimistic post-pruning with confidence CF\n"
                 "  --print-tree         train on the whole table and print the tree\n"
                 "  --scores             train on the whole table and print the score of every\n"
                 "                       attribute's best split at the root\n"
                 "%s",
                 prog, run_options_help());
    std::exit(2);
}

void print_scores(const Dataset& d, Measure measure, const std::vector<AttrScore>& scores) {
    std::printf("root scores (%s)\n", measure_name(measure));
    for (const AttrScore& s : scores) {
        const char* name = d.attrs[s.attr].name.c_str();
        if (!s.valid) std::printf("  %-28s cannot split\n", name);
        else if (measure == Measure::Gini)
            std::printf("  %-28s gini_split %.6f  reduction %.6f  %s\n", name, s.gini_split, s.gini_reduction,
                        s.split.c_str());
        else
            std::printf("  %-28s gain %.6f  split_info %.6f  gain_ratio %.6f  %s\n", name, s.gain, s.split_info,
                        s.gain_ratio, s.split.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    TreeOptions tree_opt;
    RunOptions run_opt;
    bool print_tree = false, scores = false;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage(argv[0]);
        parse_run_options(args, run_opt);
        std::string measure;
        if (args.value("--measure", measure) && !parse_measure(measure, tree_opt.measure))
            usage(argv[0], "unknown measure '" + measure + "'");
        int min_split = 2;
        args.value("--max-depth", tree_opt.max_depth);
        args.value("--min-split", min_split);
        args.value("--max-bins", tree_opt.max_bins);
        if (args.flag("--prune")) tree_opt.prune_cf = 0.25;
        args.value("--prune-cf", tree_opt.prune_cf);
        print_tree = args.flag("--print-tree");
        scores = args.flag("--scores");
        if (tree_opt.max_depth < 0 || min_split < 1 || tree_opt.max_bins < 0 || tree_opt.prune_cf < 0 ||
            tree_opt.prune_cf >= 1)
            usage(argv[0], "option value out of range");
        tree_opt.min_split = static_cast<uint32_t>(min_split);
        if (args.rest().size() != 1 || args.rest()[0][0] == '-')
            usage(argv[0], args.rest().empty() ? "no input table" : "unexpected argument '" + args.rest().back() + "'");
        run_opt.input = args.rest()[0];
    } catch (const std::invalid_argument& e) {
        usage(argv[0], e.what());
    }

    try {
        Timer timer;
        Dataset data = load_table(run_opt.input);
        RunInfo info;
        info.algorithm = "dtree";
        info.device = "cpu";
        info.options = tree_opt.describe();
        info.read_ms = timer.ms();

        if (print_tree || scores) {
            std::vector<AttrScore> root;
            DecisionTree tree = build_tree(data, tree_opt, &root);
            if (scores) print_scores(data, tree_opt.measure, root);
            if (print_tree) tree.print(stdout, data);
        }
        run_evaluation(data, run_opt, info,
                       [&] { return std::unique_ptr<Classifier>(new DecisionTreeCpu(tree_opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
