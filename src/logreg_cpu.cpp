// Multinomial logistic regression (softmax regression) on the CPU.
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
// penalized) with L-BFGS on the full training set. Every evaluation of f
// and its gradient is one pass over the records.
//
// usage: logreg_cpu <table.csv> [--C 1.0] [--max-iter 100] [--tol 1e-4] [options]
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
#include "encode.h"
#include "eval.h"
#include "lbfgs.h"
#include "runner.h"
#include "util.h"

namespace {

using namespace ml;

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

class LogisticRegressionCpu : public Classifier {
public:
    explicit LogisticRegressionCpu(const LogRegOptions& opt) : opt_(opt) {}
    std::string name() const override { return "logreg"; }

    void fit(const Dataset& train) override {
        n_classes_ = train.num_classes();
        encoder_.fit(train, Scaling::Standardize);
        encoder_.transform(train, index_, value_);
        const size_t n = train.num_rows();
        const int d = encoder_.n_attrs, C = n_classes_, D = encoder_.n_features;
        const double lambda = 1.0 / (opt_.C * static_cast<double>(n));
        const std::vector<int32_t>& y = train.y;
        std::vector<double> z(C);

        // weights_[f * C + c] for feature f; the bias is feature D.
        weights_.assign(static_cast<size_t>(D + 1) * C, 0.0);
        auto objective = [&](const std::vector<double>& w, std::vector<double>& g) {
            std::fill(g.begin(), g.end(), 0.0);
            const double* bias = &w[static_cast<size_t>(D) * C];
            double loss = 0;
            for (size_t i = 0; i < n; ++i) {
                const int32_t* idx = &index_[i * d];
                const float* val = &value_[i * d];
                for (int c = 0; c < C; ++c) z[c] = bias[c];
                for (int j = 0; j < d; ++j) {
                    const double* wf = &w[static_cast<size_t>(idx[j]) * C];
                    const double v = val[j];
                    for (int c = 0; c < C; ++c) z[c] += wf[c] * v;
                }
                // log-sum-exp, then z becomes dLoss/dz = P(c | x) - [c == y].
                double top = z[0];
                for (int c = 1; c < C; ++c) top = std::max(top, z[c]);
                double sum = 0;
                for (int c = 0; c < C; ++c) sum += std::exp(z[c] - top);
                const double lse = top + std::log(sum);
                loss += lse - z[y[i]];
                for (int c = 0; c < C; ++c) z[c] = std::exp(z[c] - lse);
                z[y[i]] -= 1.0;
                for (int j = 0; j < d; ++j) {
                    double* gf = &g[static_cast<size_t>(idx[j]) * C];
                    const double v = val[j];
                    for (int c = 0; c < C; ++c) gf[c] += z[c] * v;
                }
                double* gb = &g[static_cast<size_t>(D) * C];
                for (int c = 0; c < C; ++c) gb[c] += z[c];
            }
            const double inv_n = 1.0 / static_cast<double>(n);
            for (double& v : g) v *= inv_n;
            double sq = 0;
            for (size_t k = 0; k < static_cast<size_t>(D) * C; ++k) {
                sq += w[k] * w[k];
                g[k] += lambda * w[k];
            }
            return loss * inv_n + 0.5 * lambda * sq;
        };

        LbfgsOptions lo;
        lo.max_iter = opt_.max_iter;
        lo.tol = opt_.tol;
        result_ = lbfgs_minimize(weights_, objective, lo);
        std::vector<int32_t>().swap(index_);
        std::vector<float>().swap(value_);
    }

    std::vector<int32_t> predict(const Dataset& test) override {
        std::vector<int32_t> index;
        std::vector<float> value;
        encoder_.transform(test, index, value);
        const size_t n = test.num_rows();
        const int d = encoder_.n_attrs, C = n_classes_, D = encoder_.n_features;
        const double* bias = &weights_[static_cast<size_t>(D) * C];
        std::vector<double> z(C);
        std::vector<int32_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            for (int c = 0; c < C; ++c) z[c] = bias[c];
            for (int j = 0; j < d; ++j) {
                const double* wf = &weights_[static_cast<size_t>(index[i * d + j]) * C];
                const double v = value[i * d + j];
                for (int c = 0; c < C; ++c) z[c] += wf[c] * v;
            }
            int32_t best = 0;
            for (int c = 1; c < C; ++c)
                if (z[c] > z[best]) best = c;
            out[i] = best;
        }
        return out;
    }

    std::vector<std::pair<std::string, double>> info() const override {
        return {{"features", static_cast<double>(encoder_.n_features)},
                {"iterations", static_cast<double>(result_.iterations)},
                {"evaluations", static_cast<double>(result_.evaluations)},
                {"objective", result_.value},
                {"converged", result_.converged ? 1.0 : 0.0}};
    }

    // One line per feature (then the bias): name, then one weight per class.
    void dump(const std::string& path, const Dataset& schema) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output file: " + path);
        std::vector<std::string> names = encoder_.feature_names(schema);
        names.push_back("(bias)");
        std::fprintf(f, "feature");
        for (const std::string& c : schema.classes) std::fprintf(f, ",%s", c.c_str());
        std::fprintf(f, "\n");
        for (size_t k = 0; k < names.size(); ++k) {
            std::fprintf(f, "%s", names[k].c_str());
            for (int c = 0; c < n_classes_; ++c) std::fprintf(f, ",%.17g", weights_[k * n_classes_ + c]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    }

private:
    LogRegOptions opt_;
    Encoder encoder_;
    int n_classes_ = 0;
    std::vector<int32_t> index_;
    std::vector<float> value_;
    std::vector<double> weights_;
    LbfgsResult result_;
};

[[noreturn]] void usage(const char* prog, const std::string& problem = "") {
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

}  // namespace

int main(int argc, char** argv) {
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
        info.device = "cpu";
        info.options = opt.describe();
        info.read_ms = timer.ms();
        if (!dump.empty()) {
            LogisticRegressionCpu model(opt);
            model.fit(data);
            model.dump(dump, data);
        }
        run_evaluation(data, run_opt, info, [&] { return std::unique_ptr<Classifier>(new LogisticRegressionCpu(opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
