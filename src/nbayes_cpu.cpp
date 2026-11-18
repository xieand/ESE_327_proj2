// Naive Bayesian classification on the CPU (Han & Kamber, Section 8.3).
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
//
// usage: nbayes_cpu <table.csv> [--alpha 1.0] [--var-smoothing 1e-9] [options]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

namespace {

using namespace ml;

struct BayesOptions {
    double alpha = 1.0;
    double var_smoothing = 1e-9;

    std::string describe() const {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "alpha=%g var_smoothing=%g", alpha, var_smoothing);
        return buf;
    }
};

class NaiveBayesCpu : public Classifier {
public:
    explicit NaiveBayesCpu(const BayesOptions& opt) : opt_(opt) {}
    std::string name() const override { return "nbayes"; }

    void fit(const Dataset& train) override {
        const size_t n = train.num_rows();
        const int d = train.num_attrs(), C = train.num_classes();
        n_classes_ = C;
        n_attrs_ = d;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        std::vector<size_t> class_n = train.class_counts();
        log_prior_.assign(C, neg_inf);
        for (int c = 0; c < C; ++c)
            if (class_n[c]) log_prior_[c] = std::log(static_cast<double>(class_n[c]) / n);

        numeric_.assign(d, 0);
        offset_.assign(d, 0);
        int n_values = 0;
        for (int a = 0; a < d; ++a) {
            numeric_[a] = train.attrs[a].numeric;
            offset_[a] = n_values;
            if (!numeric_[a]) n_values += train.attrs[a].num_values();
        }
        log_prob_.assign(static_cast<size_t>(n_values) * C, 0.0);
        mean_.assign(static_cast<size_t>(d) * C, 0.0);
        var_.assign(static_cast<size_t>(d) * C, 1.0);

        // Largest variance of a numeric attribute over all records.
        double max_var = 0;
        std::vector<double> count(C), sum(C), table;
        for (int a = 0; a < d; ++a) {
            const float* col = train.column(a);
            if (!numeric_[a]) {
                // Frequency table count(value, class), then smoothed log-probabilities.
                const int V = train.attrs[a].num_values();
                table.assign(static_cast<size_t>(V) * C, 0.0);
                for (size_t i = 0; i < n; ++i) table[static_cast<size_t>(col[i]) * C + train.y[i]] += 1.0;
                for (int v = 0; v < V; ++v) {
                    for (int c = 0; c < C; ++c) {
                        if (!class_n[c]) continue;   // class absent from the training set: never predicted
                        const double p = (table[static_cast<size_t>(v) * C + c] + opt_.alpha) /
                                         (static_cast<double>(class_n[c]) + opt_.alpha * V);
                        log_prob_[static_cast<size_t>(offset_[a] + v) * C + c] = p > 0 ? std::log(p) : neg_inf;
                    }
                }
                continue;
            }
            // Mean and variance per class, and over all records (missing values are skipped).
            std::fill(count.begin(), count.end(), 0.0);
            std::fill(sum.begin(), sum.end(), 0.0);
            double all_sum = 0, all_count = 0;
            for (size_t i = 0; i < n; ++i) {
                if (std::isnan(col[i])) continue;
                count[train.y[i]] += 1.0;
                sum[train.y[i]] += col[i];
                all_sum += col[i];
                all_count += 1.0;
            }
            const double all_mean = all_count ? all_sum / all_count : 0.0;
            double* mean = &mean_[static_cast<size_t>(a) * C];
            double* var = &var_[static_cast<size_t>(a) * C];
            for (int c = 0; c < C; ++c) {
                mean[c] = count[c] ? sum[c] / count[c] : 0.0;
                var[c] = 0.0;
            }
            double all_ss = 0;
            for (size_t i = 0; i < n; ++i) {
                if (std::isnan(col[i])) continue;
                const int c = train.y[i];
                var[c] += (col[i] - mean[c]) * (col[i] - mean[c]);
                all_ss += (col[i] - all_mean) * (col[i] - all_mean);
            }
            for (int c = 0; c < C; ++c) var[c] = count[c] ? var[c] / count[c] : 0.0;
            if (all_count) max_var = std::max(max_var, all_ss / all_count);
        }
        const double epsilon = opt_.var_smoothing * max_var;
        for (int a = 0; a < d; ++a) {
            if (!numeric_[a]) continue;
            for (int c = 0; c < C; ++c) {
                double& v = var_[static_cast<size_t>(a) * C + c];
                v = std::max(v + epsilon, 1e-300);
            }
        }
    }

    // score[c] = log P(C) + sum_a log P(x_a | C) for one record given as
    // attribute values (codes for categorical attributes).
    void scores(const float* x, size_t stride, double* score) const {
        const int C = n_classes_;
        for (int c = 0; c < C; ++c) score[c] = log_prior_[c];
        for (int a = 0; a < n_attrs_; ++a) {
            const float v = x[a * stride];
            if (!numeric_[a]) {
                const double* lp = &log_prob_[static_cast<size_t>(offset_[a] + static_cast<int>(v)) * C];
                for (int c = 0; c < C; ++c) score[c] += lp[c];
            } else if (!std::isnan(v)) {
                const double* mean = &mean_[static_cast<size_t>(a) * C];
                const double* var = &var_[static_cast<size_t>(a) * C];
                for (int c = 0; c < C; ++c) {
                    const double diff = v - mean[c];
                    score[c] += -0.5 * std::log(2.0 * kPi * var[c]) - 0.5 * diff * diff / var[c];
                }
            }
        }
    }

    std::vector<int32_t> predict(const Dataset& test) override {
        const size_t n = test.num_rows();
        std::vector<double> score(n_classes_);
        std::vector<int32_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            // Record i is every n-th cell of the column-major table.
            scores(test.x.data() + i, n, score.data());
            out[i] = argmax(score.data());
        }
        return out;
    }

    int32_t argmax(const double* score) const {
        int32_t best = 0;
        for (int c = 1; c < n_classes_; ++c)
            if (score[c] > score[best]) best = c;
        return best;
    }

    double log_prior(int c) const { return log_prior_[c]; }

private:
    static constexpr double kPi = 3.14159265358979323846;
    BayesOptions opt_;
    int n_classes_ = 0, n_attrs_ = 0;
    std::vector<uint8_t> numeric_;
    std::vector<int32_t> offset_;       // categorical: first row of the attribute in log_prob_
    std::vector<double> log_prior_;     // [class]
    std::vector<double> log_prob_;      // [(offset + value) * C + class]
    std::vector<double> mean_, var_;    // [attribute * C + class]
};

// Prints P(X | C) P(C) of every class for one record given as
// "value,value,..." in attribute order.
void query(const Dataset& data, const BayesOptions& opt, const std::string& text) {
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

    NaiveBayesCpu model(opt);
    model.fit(data);
    std::vector<double> score(data.num_classes());
    model.scores(x.data(), 1, score.data());
    std::printf("query (%s)\n", text.c_str());
    for (int c = 0; c < data.num_classes(); ++c)
        std::printf("  %-22s P(X|C) %.6g  P(C) %.6g  P(X|C)P(C) %.6g\n", data.classes[c].c_str(),
                    std::exp(score[c] - model.log_prior(c)), std::exp(model.log_prior(c)), std::exp(score[c]));
    std::printf("  predicted class: %s\n", data.classes[model.argmax(score.data())].c_str());
}

[[noreturn]] void usage(const char* prog, const std::string& problem = "") {
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

}  // namespace

int main(int argc, char** argv) {
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
        info.device = "cpu";
        info.options = opt.describe();
        info.read_ms = timer.ms();
        if (has_query) query(data, opt, query_text);
        run_evaluation(data, run_opt, info, [&] { return std::unique_ptr<Classifier>(new NaiveBayesCpu(opt)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
