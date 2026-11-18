#include "dtree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ml {

const char* measure_name(Measure m) {
    switch (m) {
        case Measure::Gain: return "gain";
        case Measure::GainRatio: return "gain_ratio";
        default: return "gini";
    }
}

bool parse_measure(const std::string& s, Measure& out) {
    if (s == "gain") out = Measure::Gain;
    else if (s == "gain_ratio") out = Measure::GainRatio;
    else if (s == "gini") out = Measure::Gini;
    else return false;
    return true;
}

std::string TreeOptions::describe() const {
    std::string s = std::string("measure=") + measure_name(measure);
    if (max_depth > 0) s += " max_depth=" + std::to_string(max_depth);
    if (min_split > 2) s += " min_split=" + std::to_string(min_split);
    s += " max_bins=" + std::to_string(max_bins);
    if (prune_cf > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " prune_cf=%g", prune_cf);
        s += buf;
    }
    return s;
}

// ----------------------------------------------------------------- binning

namespace {

constexpr int kMaxBinId = 65535;   // bins are stored as uint16_t

// A missing numeric value sorts after every real value.
inline float numeric_key(float v) { return std::isnan(v) ? std::numeric_limits<float>::infinity() : v; }

}  // namespace

void Binning::fit(const Dataset& train, int max_bins) {
    const int d = train.num_attrs();
    const size_t n = train.num_rows();
    const int limit = max_bins <= 0 ? kMaxBinId : std::min(max_bins, kMaxBinId);
    n_bins.assign(d, 0);
    lo.assign(d, {});
    hi.assign(d, {});
    max_n_bins = 0;
    std::vector<float> sorted;
    for (int a = 0; a < d; ++a) {
        const Attribute& attr = train.attrs[a];
        if (!attr.numeric) {
            if (attr.num_values() > kMaxBinId) throw std::runtime_error("too many values in attribute " + attr.name);
            n_bins[a] = attr.num_values();
        } else {
            const float* col = train.column(a);
            sorted.resize(n);
            for (size_t i = 0; i < n; ++i) sorted[i] = numeric_key(col[i]);
            std::sort(sorted.begin(), sorted.end());
            size_t distinct = 0;
            for (size_t i = 0; i < n; ++i) distinct += i == 0 || sorted[i] != sorted[i - 1];
            // One bin per distinct value, or (too many values) the values are
            // taken in order and bin k is closed once it holds the k-th of
            // `limit` equal shares of the records.
            const bool merge = distinct > static_cast<size_t>(limit);
            uint64_t closed = 0;
            size_t i = 0;
            while (i < n) {
                size_t j = i;
                while (j < n && sorted[j] == sorted[i]) ++j;
                if (lo[a].size() == closed) {      // first value of a new bin
                    lo[a].push_back(sorted[i]);
                    hi[a].push_back(sorted[i]);
                } else {
                    hi[a].back() = sorted[i];
                }
                if (!merge || static_cast<uint64_t>(j) * limit >= (closed + 1) * static_cast<uint64_t>(n)) ++closed;
                i = j;
            }
            n_bins[a] = static_cast<int>(lo[a].size());
        }
        max_n_bins = std::max(max_n_bins, n_bins[a]);
    }
}

std::vector<uint16_t> Binning::transform(const Dataset& data) const {
    const int d = data.num_attrs();
    const size_t n = data.num_rows();
    std::vector<uint16_t> bins(static_cast<size_t>(d) * n);
    for (int a = 0; a < d; ++a) {
        const float* col = data.column(a);
        uint16_t* out = bins.data() + static_cast<size_t>(a) * n;
        if (!data.attrs[a].numeric) {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint16_t>(col[i]);
        } else {
            const std::vector<float>& upper = hi[a];
            for (size_t i = 0; i < n; ++i) {
                size_t b = std::lower_bound(upper.begin(), upper.end(), numeric_key(col[i])) - upper.begin();
                out[i] = static_cast<uint16_t>(std::min(b, upper.size() - 1));
            }
        }
    }
    return bins;
}

float Binning::threshold(int attr, int left, int right) const {
    const float a = hi[attr][left], b = lo[attr][right];
    float t = static_cast<float>(0.5 * (static_cast<double>(a) + static_cast<double>(b)));
    // The midpoint must still send b to the right (it can round up to b, or
    // be infinite when b stands for a missing value).
    if (!(t < b)) t = a;
    return t;
}

// -------------------------------------------------------------------- tree

int32_t DecisionTree::predict_row(const Dataset& data, size_t row) const {
    const TreeNode* node = &nodes[0];
    while (node->kind != kLeaf) {
        const float v = data.value(row, node->attr);
        if (node->kind == kThreshold) {
            node = &nodes[node->child + (v <= node->threshold ? 0 : 1)];
        } else {
            const int code = static_cast<int>(v);
            if (code < 0 || code >= node->n_values) break;   // value unknown to the tree: majority of this node
            if (node->kind == kMultiway) node = &nodes[node->child + code];
            else node = &nodes[node->child + (in_left(*node, code) ? 0 : 1)];
        }
    }
    return node->label;
}

std::vector<int32_t> DecisionTree::predict(const Dataset& data) const {
    std::vector<int32_t> out(data.num_rows());
    for (size_t i = 0; i < out.size(); ++i) out[i] = predict_row(data, i);
    return out;
}

TreeStats DecisionTree::stats() const {
    TreeStats s;
    s.nodes = nodes.size();
    for (const TreeNode& node : nodes) {
        s.leaves += node.kind == kLeaf;
        s.depth = std::max(s.depth, node.depth);
    }
    return s;
}

namespace {

// C4.5's pessimistic estimate of the number of errors a leaf will make on
// unseen data, given that it misclassified e of its n training records: n
// times the upper limit of the confidence interval (confidence cf) of the
// binomial error rate. Exact for e = 0, the continuity-corrected normal
// approximation otherwise (Quinlan, C4.5, function AddErrs).
double pessimistic_errors(double n, double e, double cf) {
    if (n <= 0) return 0;
    if (e < 0.5) return n * (1.0 - std::pow(cf, 1.0 / n));
    if (e + 0.5 >= n) return e + 0.67 * (n - e);
    // Squared normal deviate for cf, interpolated in C4.5's table.
    static const double val[] = {0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00};
    static const double dev[] = {4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00};
    int i = 1;
    while (cf > val[i]) ++i;
    const double z = dev[i - 1] + (dev[i] - dev[i - 1]) * (cf - val[i - 1]) / (val[i] - val[i - 1]);
    const double z2 = z * z, f = e + 0.5;
    return n * (f + z2 / 2 + std::sqrt(z2 * (f * (1 - f / n) + z2 / 4))) / (n + z2);
}

}  // namespace

void DecisionTree::prune(double confidence) {
    if (!(confidence > 0 && confidence < 1)) throw std::runtime_error("pruning confidence must be in (0, 1)");
    const size_t n_nodes = nodes.size();
    std::vector<double> estimate(n_nodes, 0);
    // Children always follow their parent, so a reverse scan is bottom-up.
    for (size_t i = n_nodes; i-- > 0;) {
        TreeNode& node = nodes[i];
        const double as_leaf = pessimistic_errors(node.n, errors(i), confidence);
        if (node.kind == kLeaf) {
            estimate[i] = as_leaf;
            continue;
        }
        double subtree = 0;
        for (int c = 0; c < node.num_children(); ++c) subtree += estimate[node.child + c];
        if (as_leaf <= subtree + kPruneSlack) {
            node.kind = kLeaf;
            estimate[i] = as_leaf;
        } else {
            estimate[i] = subtree;
        }
    }

    // Rebuild without the removed subtrees, keeping children consecutive.
    DecisionTree out;
    out.n_classes = n_classes;
    std::vector<size_t> source{0};   // source[new index] = old index
    out.nodes.push_back(nodes[0]);
    for (size_t j = 0; j < out.nodes.size(); ++j) {
        const TreeNode old = nodes[source[j]];
        TreeNode& node = out.nodes[j];
        if (old.kind == kLeaf) {
            node.attr = -1;
            node.child = -1;
            node.mask = -1;
            node.n_values = 0;
            continue;
        }
        if (old.kind == kSubset) {
            node.mask = static_cast<int32_t>(out.masks.size());
            out.masks.insert(out.masks.end(), masks.begin() + old.mask,
                             masks.begin() + old.mask + mask_words(old.n_values));
        }
        const int32_t first = static_cast<int32_t>(out.nodes.size());
        out.nodes[j].child = first;
        for (int c = 0; c < old.num_children(); ++c) {
            source.push_back(old.child + c);
            out.nodes.push_back(nodes[old.child + c]);
        }
    }
    out.counts.resize(out.nodes.size() * n_classes);
    for (size_t j = 0; j < out.nodes.size(); ++j)
        std::copy(counts.begin() + source[j] * n_classes, counts.begin() + (source[j] + 1) * n_classes,
                  out.counts.begin() + j * n_classes);
    *this = std::move(out);
}

namespace {

void print_node(const DecisionTree& t, const Dataset& schema, FILE* out, size_t i, const std::string& edge,
                int indent) {
    const TreeNode& node = t.nodes[i];
    std::fprintf(out, "%*s%s", 2 * indent, "", edge.c_str());
    if (node.kind == kLeaf) std::fprintf(out, "leaf %s", schema.classes[node.label].c_str());
    else std::fprintf(out, "[%s]", schema.attrs[node.attr].name.c_str());
    std::fprintf(out, " n=%u {", node.n);
    for (int c = 0; c < t.n_classes; ++c)
        std::fprintf(out, "%s%s:%u", c ? ", " : "", schema.classes[c].c_str(), t.counts[i * t.n_classes + c]);
    std::fprintf(out, "}\n");
    if (node.kind == kLeaf) return;

    const Attribute& attr = schema.attrs[node.attr];
    if (node.kind == kMultiway) {
        for (int v = 0; v < node.n_values; ++v)
            print_node(t, schema, out, node.child + v, "= " + attr.values[v] + ": ", indent + 1);
    } else if (node.kind == kThreshold) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(node.threshold));
        print_node(t, schema, out, node.child, std::string("<= ") + buf + ": ", indent + 1);
        print_node(t, schema, out, node.child + 1, std::string("> ") + buf + ": ", indent + 1);
    } else {
        std::string set;
        for (int v = 0; v < node.n_values; ++v)
            if (t.in_left(node, v)) set += (set.empty() ? "" : ", ") + attr.values[v];
        print_node(t, schema, out, node.child, "in {" + set + "}: ", indent + 1);
        print_node(t, schema, out, node.child + 1, "not in {" + set + "}: ", indent + 1);
    }
}

}  // namespace

void DecisionTree::print(FILE* out, const Dataset& schema) const { print_node(*this, schema, out, 0, "", 0); }

}  // namespace ml
