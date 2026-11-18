// The decision tree program without the tree builder: options, the
// classifier wrapper and the report. dtree_cpu and dtree_gpu differ only in
// the TreeBackend they pass to dtree_main().
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "dtree.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

namespace ml {

struct TreeBackend {
    Device device;
    // Grows the unpruned tree. root_scores (optional) receives the best split
    // of every attribute at the root.
    std::function<DecisionTree(const Dataset& train, const TreeOptions& opt, std::vector<AttrScore>* root_scores)> build;
    std::function<std::vector<int32_t>(const DecisionTree& tree, const Dataset& test)> predict;
    // Backend-specific options (optional).
    std::function<void(Args& args)> parse_options;
    std::string options_help;
};

namespace dtree_app {

inline DecisionTree build_tree(const TreeBackend& backend, const Dataset& train, const TreeOptions& opt,
                               std::vector<AttrScore>* root_scores = nullptr) {
    DecisionTree tree = backend.build(train, opt, root_scores);
    if (opt.prune_cf > 0) tree.prune(opt.prune_cf);
    return tree;
}

class TreeClassifier : public Classifier {
public:
    TreeClassifier(const TreeBackend& backend, const TreeOptions& opt) : backend_(backend), opt_(opt) {}
    std::string name() const override { return std::string("dtree(") + measure_name(opt_.measure) + ")"; }
    void fit(const Dataset& train) override { tree_ = build_tree(backend_, train, opt_); }
    std::vector<int32_t> predict(const Dataset& test) override { return backend_.predict(tree_, test); }
    std::vector<std::pair<std::string, double>> info() const override {
        const TreeStats s = tree_.stats();
        return {{"tree_nodes", static_cast<double>(s.nodes)},
                {"tree_leaves", static_cast<double>(s.leaves)},
                {"tree_depth", static_cast<double>(s.depth)}};
    }

private:
    const TreeBackend& backend_;
    TreeOptions opt_;
    DecisionTree tree_;
};

[[noreturn]] inline void usage(const char* prog, const TreeBackend& backend, const std::string& problem = "") {
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
                 "%s%s",
                 prog, backend.options_help.c_str(), run_options_help());
    std::exit(2);
}

inline void print_scores(const Dataset& d, Measure measure, const std::vector<AttrScore>& scores) {
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

}  // namespace dtree_app

inline int dtree_main(int argc, char** argv, const TreeBackend& backend) {
    using namespace dtree_app;
    TreeOptions tree_opt;
    RunOptions run_opt;
    bool print_tree = false, scores = false;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage(argv[0], backend);
        parse_run_options(args, run_opt);
        std::string measure;
        if (args.value("--measure", measure) && !parse_measure(measure, tree_opt.measure))
            usage(argv[0], backend, "unknown measure '" + measure + "'");
        int min_split = 2;
        args.value("--max-depth", tree_opt.max_depth);
        args.value("--min-split", min_split);
        args.value("--max-bins", tree_opt.max_bins);
        if (args.flag("--prune")) tree_opt.prune_cf = 0.25;
        args.value("--prune-cf", tree_opt.prune_cf);
        print_tree = args.flag("--print-tree");
        scores = args.flag("--scores");
        if (backend.parse_options) backend.parse_options(args);
        if (tree_opt.max_depth < 0 || min_split < 1 || tree_opt.max_bins < 0 || tree_opt.prune_cf < 0 ||
            tree_opt.prune_cf >= 1)
            usage(argv[0], backend, "option value out of range");
        tree_opt.min_split = static_cast<uint32_t>(min_split);
        if (args.rest().size() != 1 || args.rest()[0][0] == '-')
            usage(argv[0], backend,
                  args.rest().empty() ? "no input table" : "unexpected argument '" + args.rest().back() + "'");
        run_opt.input = args.rest()[0];
    } catch (const std::invalid_argument& e) {
        usage(argv[0], backend, e.what());
    }

    try {
        Timer timer;
        Dataset data = load_table(run_opt.input);
        RunInfo info;
        info.algorithm = "dtree";
        info.device = backend.device;
        info.options = tree_opt.describe();
        info.read_ms = timer.ms();

        if (print_tree || scores) {
            std::vector<AttrScore> root;
            DecisionTree tree = build_tree(backend, data, tree_opt, &root);
            if (scores) print_scores(data, tree_opt.measure, root);
            if (print_tree) tree.print(stdout, data);
        }
        run_evaluation(data, run_opt, info,
                       [&] { return std::unique_ptr<Classifier>(new TreeClassifier(backend, tree_opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace ml
