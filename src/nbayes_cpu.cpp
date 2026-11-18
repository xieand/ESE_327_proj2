// Naive Bayesian classification on the CPU. The model and its formulas are
// in include/nbayes.h; this file gathers the training statistics with one
// pass (categorical) or two passes (numeric) over every attribute column,
// and scores the test records one by one.
//
// usage: nbayes_cpu <table.csv> [--alpha 1.0] [--var-smoothing 1e-9] [options]
#include <cmath>
#include <vector>

#include "dataset.h"
#include "nbayes.h"

namespace {

using namespace ml;

NaiveBayesModel fit(const Dataset& train, const BayesOptions& opt) {
    const size_t n = train.num_rows();
    const int d = train.num_attrs(), C = train.num_classes();
    NaiveBayesModel model;
    model.init(train);

    std::vector<double> count(C), sum(C), mean(C), var(C), table;
    for (int a = 0; a < d; ++a) {
        const float* col = train.column(a);
        if (!model.numeric[a]) {
            // Frequency table count(value, class).
            const int V = train.attrs[a].num_values();
            table.assign(static_cast<size_t>(V) * C, 0.0);
            for (size_t i = 0; i < n; ++i) table[static_cast<size_t>(col[i]) * C + train.y[i]] += 1.0;
            model.set_categorical(a, V, table.data(), opt.alpha);
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
        model.set_numeric(a, mean.data(), var.data(), all_count ? all_ss / all_count : 0.0);
    }
    model.finish(opt.var_smoothing);
    return model;
}

std::vector<int32_t> predict(const NaiveBayesModel& model, const Dataset& test) {
    const size_t n = test.num_rows();
    std::vector<double> score(model.n_classes);
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        // Record i is every n-th cell of the column-major table.
        model.scores(test.x.data() + i, n, score.data());
        out[i] = model.argmax(score.data());
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    BayesBackend backend;
    backend.fit = fit;
    backend.predict = predict;
    return nbayes_main(argc, argv, backend);
}
