// Multinomial logistic regression on the CPU. The model, the objective and
// the program are described in include/logreg.h; this file evaluates the
// objective and its gradient with one sequential pass over the records.
//
// usage: logreg_cpu <table.csv> [--C 1.0] [--max-iter 100] [--tol 1e-4] [options]
#include <algorithm>
#include <cmath>
#include <vector>

#include "dataset.h"
#include "encode.h"
#include "lbfgs.h"
#include "logreg.h"

namespace {

using namespace ml;

LogRegModel fit(const Dataset& train, const LogRegOptions& opt) {
    LogRegModel model;
    model.n_classes = train.num_classes();
    model.encoder.fit(train, Scaling::Standardize);
    std::vector<int32_t> index;
    std::vector<float> value;
    model.encoder.transform(train, index, value);
    const size_t n = train.num_rows();
    const int d = model.encoder.n_attrs, C = model.n_classes, D = model.encoder.n_features;
    const std::vector<int32_t>& y = train.y;
    std::vector<double> z(C);

    model.weights.assign(static_cast<size_t>(D + 1) * C, 0.0);
    auto objective = [&](const std::vector<double>& w, std::vector<double>& g) {
        std::fill(g.begin(), g.end(), 0.0);
        const double* bias = &w[static_cast<size_t>(D) * C];
        double loss = 0;
        for (size_t i = 0; i < n; ++i) {
            const int32_t* idx = &index[i * d];
            const float* val = &value[i * d];
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
        return logreg_finish_objective(w, g, loss, n, opt.C, D, C);
    };

    LbfgsOptions lo;
    lo.max_iter = opt.max_iter;
    lo.tol = opt.tol;
    model.result = lbfgs_minimize(model.weights, objective, lo);
    return model;
}

std::vector<int32_t> predict(const LogRegModel& model, const Dataset& test) {
    std::vector<int32_t> index;
    std::vector<float> value;
    model.encoder.transform(test, index, value);
    const size_t n = test.num_rows();
    const int d = model.encoder.n_attrs, C = model.n_classes, D = model.encoder.n_features;
    const double* bias = &model.weights[static_cast<size_t>(D) * C];
    std::vector<double> z(C);
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < C; ++c) z[c] = bias[c];
        for (int j = 0; j < d; ++j) {
            const double* wf = &model.weights[static_cast<size_t>(index[i * d + j]) * C];
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

}  // namespace

int main(int argc, char** argv) {
    LogRegBackend backend;
    backend.fit = fit;
    backend.predict = predict;
    return logreg_main(argc, argv, backend);
}
