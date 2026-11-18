// Naive Bayesian classification on the GPU. The model and its formulas are
// in include/nbayes.h; the GPU gathers the training statistics and scores
// the test records.
//
// Training
//   categorical  One thread per (attribute, record) adds 1 to the cell
//                (value, class) of the attribute's frequency table with an
//                atomic increment. The counts are integers, so the table is
//                exactly the one the CPU builds.
//   numeric      The records are cut into blocks of kBlockRecords. One
//                thread per (attribute, block) sums its block, the host adds
//                the block sums in order. Two passes as on the CPU: counts
//                and sums (for the means), then squared deviations (for the
//                variances). No atomics are involved, so the result does not
//                depend on the thread schedule; it differs from the CPU's
//                only by the rounding of a different summation order.
//   The host turns the statistics into log-probabilities (a few thousand
//   numbers).
//
// Prediction
//   One thread per test record computes the score of every class in turn,
//   adding the attributes in the same order as the CPU, and keeps the best.
//
// usage: nbayes_gpu <table.csv> [--alpha 1.0] [--var-smoothing 1e-9] [options]
#include <vector>

#include "cuda_compat.h"
#include "dataset.h"
#include "nbayes.h"

namespace {

using namespace ml;

constexpr size_t kBlockRecords = 256;

// counts[(offset[a] + value) * C + class] over all categorical attributes.
__global__ void k_count(const float* x, const int32_t* y, const int32_t* numeric, const int32_t* offset,
                        uint32_t* counts, size_t n, int d, int C) {
    GRID_STRIDE(i, n * d) {
        const size_t a = i / n, r = i - a * n;
        if (numeric[a]) continue;
        atomicAdd(&counts[static_cast<size_t>(offset[a] + static_cast<int>(x[i])) * C + y[r]], 1u);
    }
}

// Pass 1 over the numeric attributes. For block j of numeric attribute q,
// out[(q * n_blocks + j) * (2C + 1) + ...] = count per class (C), sum per
// class (C), sum over all records (1). Missing values (NaN) are skipped.
__global__ void k_sums(const float* x, const int32_t* y, const int32_t* num_attr, double* out, size_t n,
                       size_t n_blocks, int n_numeric, int C) {
    GRID_STRIDE(i, n_blocks * n_numeric) {
        const size_t q = i / n_blocks, j = i - q * n_blocks;
        const float* col = x + static_cast<size_t>(num_attr[q]) * n;
        double* o = out + i * (2 * C + 1);
        for (int c = 0; c < 2 * C + 1; ++c) o[c] = 0;
        const size_t end = j * kBlockRecords + kBlockRecords < n ? j * kBlockRecords + kBlockRecords : n;
        for (size_t r = j * kBlockRecords; r < end; ++r) {
            const float v = col[r];
            if (v != v) continue;
            o[y[r]] += 1.0;
            o[C + y[r]] += v;
            o[2 * C] += v;
        }
    }
}

// Pass 2: out[(q * n_blocks + j) * (C + 1) + ...] = sum of squared
// deviations from the class mean per class (C), from the overall mean (1).
__global__ void k_squares(const float* x, const int32_t* y, const int32_t* num_attr, const double* mean,
                          const double* all_mean, double* out, size_t n, size_t n_blocks, int n_numeric, int C) {
    GRID_STRIDE(i, n_blocks * n_numeric) {
        const size_t q = i / n_blocks, j = i - q * n_blocks;
        const float* col = x + static_cast<size_t>(num_attr[q]) * n;
        const double* m = mean + q * C;
        double* o = out + i * (C + 1);
        for (int c = 0; c < C + 1; ++c) o[c] = 0;
        const size_t end = j * kBlockRecords + kBlockRecords < n ? j * kBlockRecords + kBlockRecords : n;
        for (size_t r = j * kBlockRecords; r < end; ++r) {
            const float v = col[r];
            if (v != v) continue;
            const int c = y[r];
            o[c] += (v - m[c]) * (v - m[c]);
            o[C] += (v - all_mean[q]) * (v - all_mean[q]);
        }
    }
}

// Class with the highest log P(C) + sum_a log P(x_a | C) (ties: lowest id).
__global__ void k_predict(const float* x, const int32_t* numeric, const int32_t* offset, const double* log_prior,
                          const double* log_prob, const double* mean, const double* var, const double* log_norm,
                          int32_t* out, size_t n, int d, int C) {
    GRID_STRIDE(i, n) {
        int32_t best = 0;
        double best_score = 0;
        for (int c = 0; c < C; ++c) {
            double score = log_prior[c];
            for (int a = 0; a < d; ++a) {
                const float v = x[static_cast<size_t>(a) * n + i];
                if (!numeric[a]) {
                    score += log_prob[static_cast<size_t>(offset[a] + static_cast<int>(v)) * C + c];
                } else if (v == v) {
                    const double diff = v - mean[a * C + c];
                    score += log_norm[a * C + c] - 0.5 * diff * diff / var[a * C + c];
                }
            }
            if (c == 0 || score > best_score) {
                best = c;
                best_score = score;
            }
        }
        out[i] = best;
    }
}

NaiveBayesModel fit(const Dataset& train, const BayesOptions& opt) {
    const size_t n = train.num_rows();
    const int d = train.num_attrs(), C = train.num_classes();
    NaiveBayesModel model;
    model.init(train);

    gpu::DevBuf<float> d_x;
    gpu::DevBuf<int32_t> d_y, d_numeric, d_offset;
    d_x.upload(train.x);
    d_y.upload(train.y);
    d_numeric.upload(model.numeric);
    d_offset.upload(model.offset);

    // ---- categorical attributes: frequency tables
    if (model.n_values > 0) {
        const size_t cells = static_cast<size_t>(model.n_values) * C;
        gpu::DevBuf<uint32_t> d_counts;
        d_counts.zero(cells);
        gpu::launch(k_count, n * d, d_x.get(), d_y.get(), d_numeric.get(), d_offset.get(), d_counts.get(), n, d, C);
        std::vector<uint32_t> counts;
        d_counts.download(counts, cells);
        for (int a = 0; a < d; ++a)
            if (!model.numeric[a])
                model.set_categorical(a, train.attrs[a].num_values(),
                                      counts.data() + static_cast<size_t>(model.offset[a]) * C, opt.alpha);
    }

    // ---- numeric attributes: mean and variance per class
    std::vector<int32_t> num_attr;
    for (int a = 0; a < d; ++a)
        if (model.numeric[a]) num_attr.push_back(a);
    const int Q = static_cast<int>(num_attr.size());
    if (Q > 0) {
        const size_t n_blocks = (n + kBlockRecords - 1) / kBlockRecords;
        gpu::DevBuf<int32_t> d_num_attr;
        d_num_attr.upload(num_attr);

        const int w1 = 2 * C + 1, w2 = C + 1;
        gpu::DevBuf<double> d_part(n_blocks * Q * w1);
        std::vector<double> part;
        gpu::launch(k_sums, n_blocks * Q, d_x.get(), d_y.get(), d_num_attr.get(), d_part.get(), n, n_blocks, Q, C);
        d_part.download(part, n_blocks * Q * w1);
        std::vector<double> count(static_cast<size_t>(Q) * C, 0.0), mean(static_cast<size_t>(Q) * C, 0.0);
        std::vector<double> all_count(Q, 0.0), all_mean(Q, 0.0);
        for (int q = 0; q < Q; ++q) {
            double all_sum = 0;
            for (size_t j = 0; j < n_blocks; ++j) {
                const double* p = &part[(q * n_blocks + j) * w1];
                for (int c = 0; c < C; ++c) {
                    count[q * C + c] += p[c];
                    mean[q * C + c] += p[C + c];
                }
                all_sum += p[2 * C];
            }
            for (int c = 0; c < C; ++c) {
                all_count[q] += count[q * C + c];
                mean[q * C + c] = count[q * C + c] ? mean[q * C + c] / count[q * C + c] : 0.0;
            }
            all_mean[q] = all_count[q] ? all_sum / all_count[q] : 0.0;
        }

        gpu::DevBuf<double> d_mean, d_all_mean;
        d_mean.upload(mean);
        d_all_mean.upload(all_mean);
        gpu::launch(k_squares, n_blocks * Q, d_x.get(), d_y.get(), d_num_attr.get(), d_mean.get(), d_all_mean.get(),
                    d_part.get(), n, n_blocks, Q, C);
        d_part.download(part, n_blocks * Q * w2);
        std::vector<double> var(C);
        for (int q = 0; q < Q; ++q) {
            double all_ss = 0;
            std::fill(var.begin(), var.end(), 0.0);
            for (size_t j = 0; j < n_blocks; ++j) {
                const double* p = &part[(q * n_blocks + j) * w2];
                for (int c = 0; c < C; ++c) var[c] += p[c];
                all_ss += p[C];
            }
            for (int c = 0; c < C; ++c) var[c] = count[q * C + c] ? var[c] / count[q * C + c] : 0.0;
            model.set_numeric(num_attr[q], &mean[static_cast<size_t>(q) * C], var.data(),
                              all_count[q] ? all_ss / all_count[q] : 0.0);
        }
    }
    model.finish(opt.var_smoothing);
    gpu::synchronize();
    return model;
}

std::vector<int32_t> predict(const NaiveBayesModel& model, const Dataset& test) {
    const size_t n = test.num_rows();
    std::vector<int32_t> out(n);
    if (n == 0) return out;
    gpu::DevBuf<float> d_x;
    gpu::DevBuf<int32_t> d_numeric, d_offset, d_out(n);
    gpu::DevBuf<double> d_log_prior, d_log_prob, d_mean, d_var, d_log_norm;
    d_x.upload(test.x);
    d_numeric.upload(model.numeric);
    d_offset.upload(model.offset);
    d_log_prior.upload(model.log_prior);
    d_log_prob.upload(model.log_prob);
    d_mean.upload(model.mean);
    d_var.upload(model.var);
    d_log_norm.upload(model.log_norm);
    gpu::launch(k_predict, n, d_x.get(), d_numeric.get(), d_offset.get(), d_log_prior.get(), d_log_prob.get(),
                d_mean.get(), d_var.get(), d_log_norm.get(), d_out.get(), n, model.n_attrs, model.n_classes);
    d_out.download(out.data(), n);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    BayesBackend backend;
    backend.device.id = gpu::device_id();
    backend.device.name = gpu::device_name();
    backend.device.peak_bytes = &gpu::g_peak_bytes;
    backend.fit = fit;
    backend.predict = predict;
    gpu::warm_up();
    return nbayes_main(argc, argv, backend);
}
