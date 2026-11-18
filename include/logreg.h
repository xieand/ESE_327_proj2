// Multinomial logistic regression (softmax regression): the model and the
// program shared by logreg_cpu and logreg_gpu, which differ only in where
// the objective, its gradient and the predictions are computed.
//
// Features: one indicator per categorical value and one standardized feature
// per numeric attribute (include/encode.h), plus a bias per class. With
// weights W (features x classes) and bias b, the model is
//
//   P(class c | x) = exp(W_c . x + b_c) / sum_k exp(W_k . x + b_k)
//
// and training minimizes the L2-regularized mean negative log-likelihood
//
//   f(W, b) = (1/n) sum_i -log P(y_i | x_i)  +  1/(2 C n) * ||W||^2
//
// (the objective of scikit-learn's LogisticRegression; the bias is not
// penalized) with L-BFGS on the full training set (include/lbfgs.h). The
// optimizer runs on the host; every evaluation of f and its gradient is one
// pass over the records, and that pass is what the backends implement.
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
#include "encode.h"
#include "eval.h"
#include "lbfgs.h"
#include "runner.h"
#include "util.h"

namespace ml {

struct LogRegOptions {
    double C = 1.0;          // inverse regularization strength
    int max_iter = 100;
    double tol = 1e-4;

    std::string describe() const {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "C=%g max_iter=%d tol=%g", C, max_iter, tol);
        return buf;
    }
};

struct LogRegModel {
    Encoder encoder;
    int n_classes = 0;
    std::vector<double> weights;     // weights[f * n_classes + c]; the bias is feature encoder.n_features
    LbfgsResult result;

    // One line per feature (then the bias): name, then one weight per class.
    void dump(const std::string& path, const Dataset& schema) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output file: " + path);
        std::vector<std::string> names = encoder.feature_names(schema);
        names.push_back("(bias)");
        std::fprintf(f, "feature");
        for (const std::string& c : schema.classes) std::fprintf(f, ",%s", c.c_str());
        std::fprintf(f, "\n");
        for (size_t k = 0; k < names.size(); ++k) {
            std::fprintf(f, "%s", names[k].c_str());
            for (int c = 0; c < n_classes; ++c) std::fprintf(f, ",%.17g", weights[k * n_classes + c]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    }
};

// Last step of every objective evaluation. On entry `g` holds the sums over
// the records of d(loss_i)/dW and `loss_sum` the sum of the losses; on exit
// `g` is the gradient of f and the return value is f.
inline double logreg_finish_objective(const std::vector<double>& w, std::vector<double>& g, double loss_sum, size_t n,
                                      double C, int n_features, int n_classes) {
    const double inv_n = 1.0 / static_cast<double>(n);
    const double lambda = 1.0 / (C * static_cast<double>(n));
    for (double& v : g) v *= inv_n;
    double sq = 0;
    for (size_t k = 0; k < static_cast<size_t>(n_features) * n_classes; ++k) {
        sq += w[k] * w[k];
        g[k] += lambda * w[k];
    }
    return loss_sum * inv_n + 0.5 * lambda * sq;
}

struct LogRegBackend {
    Device device;
    std::function<LogRegModel(const Dataset& train, const LogRegOptions& opt)> fit;
    std::function<std::vector<int32_t>(const LogRegModel& model, const Dataset& test)> predict;
};

namespace logreg_app {

class LogRegClassifier : public Classifier {
public:
    LogRegClassifier(const LogRegBackend& backend, const LogRegOptions& opt) : backend_(backend), opt_(opt) {}
    std::string name() const override { return "logreg"; }
    void fit(const Dataset& train) override { model_ = backend_.fit(train, opt_); }
    std::vector<int32_t> predict(const Dataset& test) override { return backend_.predict(model_, test); }
    std::vector<std::pair<std::string, double>> info() const override {
        return {{"features", static_cast<double>(model_.encoder.n_features)},
                {"iterations", static_cast<double>(model_.result.iterations)},
                {"evaluations", static_cast<double>(model_.result.evaluations)},
                {"objective", model_.result.value},
                {"converged", model_.result.converged ? 1.0 : 0.0}};
    }

private:
    const LogRegBackend& backend_;
    LogRegOptions opt_;
    LogRegModel model_;
};

[[noreturn]] inline void usage(const char* prog, const std::string& problem = "") {
    if (!problem.empty()) std::fprintf(stderr, "error: %s\n", problem.c_str());
    std::fprintf(stderr,
                 "usage: %s <table.csv> [options]\n"
                 "  --C X                inverse of the L2 regularization strength (default 1)\n"
                 "  --max-iter N         L-BFGS iterations (default 100)\n"
                 "  --tol T              stop when every gradient component is below T (default 1e-4)\n"
                 "  --dump-model file    train on the whole table and write the weights\n"
                 "%s",
                 prog, run_options_help());
    std::exit(2);
}

}  // namespace logreg_app

inline int logreg_main(int argc, char** argv, const LogRegBackend& backend) {
    using namespace logreg_app;
    LogRegOptions opt;
    RunOptions run_opt;
    std::string dump;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage(argv[0]);
        parse_run_options(args, run_opt);
        args.value("--C", opt.C);
        args.value("--max-iter", opt.max_iter);
        args.value("--tol", opt.tol);
        args.value("--dump-model", dump);
        if (!(opt.C > 0) || opt.max_iter < 0 || !(opt.tol >= 0)) usage(argv[0], "option value out of range");
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
        info.algorithm = "logreg";
        info.device = backend.device;
        info.options = opt.describe();
        info.read_ms = timer.ms();
        if (!dump.empty()) backend.fit(data, opt).dump(dump, data);
        run_evaluation(data, run_opt, info,
                       [&] { return std::unique_ptr<Classifier>(new LogRegClassifier(backend, opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace ml
