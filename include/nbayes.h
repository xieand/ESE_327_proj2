// Naive Bayesian classification (Han & Kamber, Section 8.3): the model and
// the program shared by nbayes_cpu and nbayes_gpu, which differ only in how
// the training statistics are gathered and where the records are scored.
//
// A record X = (x_1 .. x_d) is assigned to the class C that maximizes
//
//   P(C) * prod_a P(x_a | C)
//
// (evaluated as a sum of logarithms), where
//
//   P(C)              = n_C / n
//   categorical x_a   P(x_a | C) = (count(x_a, C) + alpha) / (n_C + alpha * V_a)
//                     with V_a values of the attribute; alpha = 1 is the
//                     Laplacian correction, alpha = 0 the plain frequencies
//   numeric x_a       P(x_a | C) = Gaussian density with the mean and the
//                     variance of the attribute among the records of C
//
// A small share (var_smoothing) of the largest attribute variance is added to
// every variance, as in scikit-learn's GaussianNB, so that an attribute that
// is constant within a class does not produce a zero variance.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

namespace ml {

struct BayesOptions {
    double alpha = 1.0;
    double var_smoothing = 1e-9;

    std::string describe() const {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "alpha=%g var_smoothing=%g", alpha, var_smoothing);
        return buf;
    }
};

// Training fills the model from sufficient statistics, in this order:
// init(), then set_categorical() / set_numeric() for every attribute, then
// finish().
struct NaiveBayesModel {
    int n_classes = 0, n_attrs = 0, n_values = 0;   // n_values: categorical values of all attributes together
    std::vector<int32_t> numeric;       // per attribute: 1 = numeric
    std::vector<int32_t> offset;        // categorical: first row of the attribute in log_prob
    std::vector<double> class_n;        // training records per class
    std::vector<double> log_prior;      // [class]
    std::vector<double> log_prob;       // [(offset + value) * C + class]
    std::vector<double> mean, var;      // [attribute * C + class]
    std::vector<double> log_norm;       // [attribute * C + class]: -1/2 log(2 pi var)
    double max_var = 0;                 // largest variance of a numeric attribute over all records

    void init(const Dataset& train) {
        const int d = train.num_attrs(), C = train.num_classes();
        const double n = static_cast<double>(train.num_rows());
        n_classes = C;
        n_attrs = d;
        class_n.assign(C, 0.0);
        for (int32_t c : train.y) class_n[c] += 1.0;
        log_prior.assign(C, -std::numeric_limits<double>::infinity());
        for (int c = 0; c < C; ++c)
            if (class_n[c] > 0) log_prior[c] = std::log(class_n[c] / n);
        numeric.assign(d, 0);
        offset.assign(d, 0);
        n_values = 0;
        for (int a = 0; a < d; ++a) {
            numeric[a] = train.attrs[a].numeric;
            offset[a] = n_values;
            if (!numeric[a]) n_values += train.attrs[a].num_values();
        }
        log_prob.assign(static_cast<size_t>(n_values) * C, 0.0);
        mean.assign(static_cast<size_t>(d) * C, 0.0);
        var.assign(static_cast<size_t>(d) * C, 1.0);
        log_norm.assign(static_cast<size_t>(d) * C, 0.0);
        max_var = 0;
    }

    // count[v * C + c]: training records of class c with value v of attribute a.
    template <typename Count>
    void set_categorical(int a, int n_attr_values, const Count* count, double alpha) {
        const int C = n_classes, V = n_attr_values;
        for (int v = 0; v < V; ++v) {
            for (int c = 0; c < C; ++c) {
                if (!(class_n[c] > 0)) continue;   // class absent from the training set: never predicted
                const double p = (static_cast<double>(count[static_cast<size_t>(v) * C + c]) + alpha) /
                                 (class_n[c] + alpha * V);
                log_prob[static_cast<size_t>(offset[a] + v) * C + c] =
                    p > 0 ? std::log(p) : -std::numeric_limits<double>::infinity();
            }
        }
    }

    // class_mean[c], class_var[c]: mean and (population) variance of attribute
    // a among the records of class c; all_var: its variance over all records.
    void set_numeric(int a, const double* class_mean, const double* class_var, double all_var) {
        for (int c = 0; c < n_classes; ++c) {
            mean[static_cast<size_t>(a) * n_classes + c] = class_mean[c];
            var[static_cast<size_t>(a) * n_classes + c] = class_var[c];
        }
        max_var = std::max(max_var, all_var);
    }

    void finish(double var_smoothing) {
        const double epsilon = var_smoothing * max_var;
        for (int a = 0; a < n_attrs; ++a) {
            if (!numeric[a]) continue;
            for (int c = 0; c < n_classes; ++c) {
                const size_t k = static_cast<size_t>(a) * n_classes + c;
                var[k] = std::max(var[k] + epsilon, 1e-300);
                log_norm[k] = -0.5 * std::log(2.0 * kPi * var[k]);
            }
        }
    }

    // score[c] = log P(C) + sum_a log P(x_a | C) for one record given as
    // attribute values (codes for categorical attributes), x[a * stride].
    void scores(const float* x, size_t stride, double* score) const {
        const int C = n_classes;
        for (int c = 0; c < C; ++c) score[c] = log_prior[c];
        for (int a = 0; a < n_attrs; ++a) {
            const float v = x[a * stride];
            if (!numeric[a]) {
                const double* lp = &log_prob[static_cast<size_t>(offset[a] + static_cast<int>(v)) * C];
                for (int c = 0; c < C; ++c) score[c] += lp[c];
            } else if (!std::isnan(v)) {
                const double* m = &mean[static_cast<size_t>(a) * C];
                const double* s2 = &var[static_cast<size_t>(a) * C];
                const double* ln = &log_norm[static_cast<size_t>(a) * C];
                for (int c = 0; c < C; ++c) {
                    const double diff = v - m[c];
                    score[c] += ln[c] - 0.5 * diff * diff / s2[c];
                }
            }
        }
    }

    int32_t argmax(const double* score) const {
        int32_t best = 0;
        for (int c = 1; c < n_classes; ++c)
            if (score[c] > score[best]) best = c;
        return best;
    }

    static constexpr double kPi = 3.14159265358979323846;
};

struct BayesBackend {
    Device device;
    std::function<NaiveBayesModel(const Dataset& train, const BayesOptions& opt)> fit;
    std::function<std::vector<int32_t>(const NaiveBayesModel& model, const Dataset& test)> predict;
};

namespace nbayes_app {

class BayesClassifier : public Classifier {
public:
    BayesClassifier(const BayesBackend& backend, const BayesOptions& opt) : backend_(backend), opt_(opt) {}
    std::string name() const override { return "nbayes"; }
    void fit(const Dataset& train) override { model_ = backend_.fit(train, opt_); }
    std::vector<int32_t> predict(const Dataset& test) override { return backend_.predict(model_, test); }

private:
    const BayesBackend& backend_;
    BayesOptions opt_;
    NaiveBayesModel model_;
};

// Prints P(X | C) P(C) of every class for one record given as
// "value,value,..." in attribute order.
inline void query(const BayesBackend& backend, const Dataset& data, const BayesOptions& opt, const std::string& text) {
    std::vector<float> x;
    size_t pos = 0;
    for (int a = 0; a < data.num_attrs(); ++a) {
        if (pos > text.size()) throw std::runtime_error("--query needs one value per attribute");
        size_t comma = text.find(',', pos);
        if (comma == std::string::npos) comma = text.size();
        const std::string v = text.substr(pos, comma - pos);
        pos = comma + 1;
        const Attribute& attr = data.attrs[a];
        if (attr.numeric) {
            char* end = nullptr;
            x.push_back(std::strtof(v.c_str(), &end));
            if (v.empty() || *end) throw std::runtime_error("--query: '" + v + "' is not a number (" + attr.name + ")");
        } else {
            auto it = std::find(attr.values.begin(), attr.values.end(), v);
            if (it == attr.values.end())
                throw std::runtime_error("--query: '" + v + "' is not a value of " + attr.name);
            x.push_back(static_cast<float>(it - attr.values.begin()));
        }
    }
    if (pos <= text.size()) throw std::runtime_error("--query has more values than the table has attributes");

    const NaiveBayesModel model = backend.fit(data, opt);
    std::vector<double> score(data.num_classes());
    model.scores(x.data(), 1, score.data());
    std::printf("query (%s)\n", text.c_str());
    for (int c = 0; c < data.num_classes(); ++c)
        std::printf("  %-22s P(X|C) %.6g  P(C) %.6g  P(X|C)P(C) %.6g\n", data.classes[c].c_str(),
                    std::exp(score[c] - model.log_prior[c]), std::exp(model.log_prior[c]), std::exp(score[c]));
    std::printf("  predicted class: %s\n", data.classes[model.argmax(score.data())].c_str());
}

[[noreturn]] inline void usage(const char* prog, const std::string& problem = "") {
    if (!problem.empty()) std::fprintf(stderr, "error: %s\n", problem.c_str());
    std::fprintf(stderr,
                 "usage: %s <table.csv> [options]\n"
                 "  --alpha A            additive smoothing of the categorical frequencies\n"
                 "                       (default 1 = Laplacian correction; 0 = none)\n"
                 "  --var-smoothing S    share of the largest variance added to every variance (default 1e-9)\n"
                 "  --query v1,v2,...    train on the whole table and print P(X|C)P(C) of every class\n"
                 "                       for the record with these attribute values\n"
                 "%s",
                 prog, run_options_help());
    std::exit(2);
}

}  // namespace nbayes_app

inline int nbayes_main(int argc, char** argv, const BayesBackend& backend) {
    using namespace nbayes_app;
    BayesOptions opt;
    RunOptions run_opt;
    std::string query_text;
    bool has_query = false;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage(argv[0]);
        parse_run_options(args, run_opt);
        args.value("--alpha", opt.alpha);
        args.value("--var-smoothing", opt.var_smoothing);
        has_query = args.value("--query", query_text);
        if (opt.alpha < 0 || opt.var_smoothing < 0) usage(argv[0], "option value out of range");
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
        info.algorithm = "nbayes";
        info.device = backend.device;
        info.options = opt.describe();
        info.read_ms = timer.ms();
        if (has_query) query(backend, data, opt, query_text);
        run_evaluation(data, run_opt, info,
                       [&] { return std::unique_ptr<Classifier>(new BayesClassifier(backend, opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace ml
