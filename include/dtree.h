// Decision tree shared by the CPU and CUDA builders: options, attribute
// binning, the tree itself (prediction, pruning, printing).
//
// Splits
//   information gain / gain ratio (ID3, C4.5)
//       categorical attribute  one branch per value (multiway)
//       numeric attribute      A <= t  /  A > t
//   Gini index (CART)          always binary
//       categorical attribute  A in S  /  A not in S
//       numeric attribute      A <= t  /  A > t
//
// Numeric thresholds are midpoints between adjacent values, as in Han &
// Kamber. Every attribute is first mapped to small integer bins: a
// categorical value is its own bin, and a numeric attribute gets one bin per
// distinct training value, so the search is exact. Only when a numeric
// attribute has more distinct values than TreeOptions::max_bins are adjacent
// values merged into equal-frequency bins, and thresholds are then only
// tried between bins.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "dataset.h"

namespace ml {

enum class Measure { Gain, GainRatio, Gini };

const char* measure_name(Measure m);
bool parse_measure(const std::string& s, Measure& out);

struct TreeOptions {
    Measure measure = Measure::Gain;
    int max_depth = 0;           // 0 = unlimited (the root has depth 0)
    uint32_t min_split = 2;      // nodes with fewer records become leaves
    int max_bins = 256;          // per numeric attribute; 0 = one bin per distinct value
    double prune_cf = 0;         // > 0: pessimistic post-pruning with this confidence (C4.5 uses 0.25)

    std::string describe() const;
};

// Two selection scores are treated as equal when they differ by less than
// this; the earlier candidate then wins (lower attribute index, lower
// threshold, lower subset number). Scores are computed from integer counts,
// so real differences are far larger, while the rounding noise that depends
// on the order of the floating-point operations (which differs between the
// CPU and the GPU) is far smaller.
constexpr double kScoreEps = 1e-9;

// For a Gini split of a categorical attribute, all 2^(v-1) - 1 subsets are
// tried when the node has at most this many distinct values and more than
// two classes. Otherwise the values are ordered by the fraction of the
// node's majority class and only the v - 1 splits of that order are tried,
// which is still optimal for two classes (Breiman et al.).
constexpr int kExhaustiveSubsetValues = 10;

// Pruning replaces a subtree by a leaf unless the subtree's estimated error
// count is lower by more than this (C4.5 uses the same margin).
constexpr double kPruneSlack = 0.1;

// ----------------------------------------------------------------- binning

struct Binning {
    std::vector<int> n_bins;               // per attribute
    std::vector<std::vector<float>> lo;    // numeric: smallest value of every bin
    std::vector<std::vector<float>> hi;    // numeric: largest value of every bin
    int max_n_bins = 0;

    void fit(const Dataset& train, int max_bins);
    // Bin of every cell of `data`, column-major like Dataset::x.
    std::vector<uint16_t> transform(const Dataset& data) const;
    // Threshold that separates bin `left` from the later bin `right`.
    float threshold(int attr, int left, int right) const;
};

// -------------------------------------------------------------------- tree

enum NodeKind : uint8_t { kLeaf = 0, kMultiway = 1, kThreshold = 2, kSubset = 3 };

struct TreeNode {
    uint8_t kind = kLeaf;
    int32_t attr = -1;
    float threshold = 0;     // kThreshold: left child if value <= threshold
    int32_t child = -1;      // first child; children are consecutive
                             //   kMultiway: one per attribute value, indexed by value code
                             //   kThreshold / kSubset: left, right
    int32_t n_values = 0;    // kMultiway / kSubset: number of values of the attribute
    int32_t mask = -1;       // kSubset: offset of the value bit mask (bit set = left child)
    int32_t label = 0;       // majority class of the node's training records
    int32_t depth = 0;
    uint32_t n = 0;          // training records that reached the node

    int num_children() const { return kind == kLeaf ? 0 : kind == kMultiway ? n_values : 2; }
};

struct TreeStats {
    size_t nodes = 0, leaves = 0;
    int depth = 0;
};

struct DecisionTree {
    int n_classes = 0;
    std::vector<TreeNode> nodes;       // nodes[0] is the root; children follow their parent
    std::vector<uint32_t> counts;      // counts[node * n_classes + c]: training records per class
    std::vector<uint64_t> masks;

    static int mask_words(int n_values) { return (n_values + 63) / 64; }
    bool in_left(const TreeNode& node, int code) const {
        return (masks[node.mask + (code >> 6)] >> (code & 63)) & 1u;
    }
    uint32_t errors(size_t node) const { return nodes[node].n - counts[node * n_classes + nodes[node].label]; }

    int32_t predict_row(const Dataset& data, size_t row) const;
    std::vector<int32_t> predict(const Dataset& data) const;
    TreeStats stats() const;

    // C4.5's pessimistic pruning (subtree replacement): bottom-up, a subtree
    // becomes a leaf when the leaf's estimated error count is at most
    // kPruneSlack above the sum of the estimates of the subtree's leaves.
    void prune(double confidence);

    // Indented text, one node per line, with the class counts of every node.
    void print(FILE* out, const Dataset& schema) const;
};

// Score of the best split of one attribute at the root (for validation
// against hand-computed examples).
struct AttrScore {
    int attr = -1;
    bool valid = false;          // false: the attribute cannot split the node
    double gain = 0, split_info = 0, gain_ratio = 0;   // gain / gain ratio
    double gini_split = 0, gini_reduction = 0;         // Gini index of the split, and its reduction
    std::string split;           // description of the split
};

}  // namespace ml
