// The k-nearest-neighbor program without the search: options and report.
// knn_cpu and knn_gpu differ only in the classifier they pass to knn_main().
//
// Training only stores the records. A test record gets the most common class
// among the training records closest to it, with the squared distance
//
//   dist(X, Y) = sum over attributes a of
//       numeric      (x_a - y_a)^2, both scaled to [0, 1] by the training minimum and maximum
//       categorical  0 if the values are equal, 1 otherwise
//
// The voters are all training records that are at most as far away as the
// k-th nearest one. That is exactly k records when distances are distinct;
// records tied with the k-th all vote, so the result does not depend on the
// order of the training records. (Categorical attributes make ties the rule
// rather than the exception: the distance is then a count of mismatches.)
// Of two classes with the same number of votes, the lower class id wins.
//
// The search is exhaustive (every test record against every training
// record), in single precision, and the distance of a pair is accumulated in
// attribute order, so the CPU and the GPU compute bit-identical distances.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "dataset.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

namespace ml {

struct KnnBackend {
    Device device;
    std::function<std::unique_ptr<Classifier>(int k)> make;
};

inline int knn_main(int argc, char** argv, const KnnBackend& backend) {
    auto usage = [&](const std::string& problem) {
        if (!problem.empty()) std::fprintf(stderr, "error: %s\n", problem.c_str());
        std::fprintf(stderr,
                     "usage: %s <table.csv> [options]\n"
                     "  --k K                number of neighbors (default 5)\n"
                     "%s",
                     argv[0], run_options_help());
        std::exit(2);
    };
    int k = 5;
    RunOptions run_opt;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage("");
        parse_run_options(args, run_opt);
        args.value("--k", k);
        if (k < 1) usage("--k must be at least 1");
        if (args.rest().size() != 1 || args.rest()[0][0] == '-')
            usage(args.rest().empty() ? "no input table" : "unexpected argument '" + args.rest().back() + "'");
        run_opt.input = args.rest()[0];
    } catch (const std::invalid_argument& e) {
        usage(e.what());
    }

    try {
        Timer timer;
        Dataset data = load_table(run_opt.input);
        RunInfo info;
        info.algorithm = "knn";
        info.device = backend.device;
        info.options = "k=" + std::to_string(k);
        info.read_ms = timer.ms();
        run_evaluation(data, run_opt, info, [&] { return backend.make(k); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace ml
