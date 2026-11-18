// Multinomial logistic regression on the GPU. The model, the objective and
// the program are described in include/logreg.h. L-BFGS runs on the host
// over the weights (a few thousand numbers); the GPU evaluates the objective
// and its gradient, which is one pass over the training records, and
// classifies the test records.
//
// With X the (records x features) matrix, W the weights and Y the one-hot
// labels, an evaluation is
//
//   Z = X W            forward:  one thread per record
//   P = softmax(Z)               (same thread; also the record's loss)
//   G = X^T (P - Y)    backward: one thread per (feature, class), in two steps
//
// X is never stored as a matrix. Every record has exactly one active feature
// per attribute, so the forward pass reads X by rows as n_attrs (feature,
// value) pairs per record. The backward pass needs X by columns: for every
// feature, the records that have it. That is one list of record numbers per
// categorical attribute, sorted by value (built once, on the host); a
// numeric attribute and the bias are features that every record has.
//
// A gradient entry is a sum over up to n records, so it is computed in two
// steps: the records of a feature are cut into blocks of kBlockRecords, one
// thread per (block, class) sums its block, then one thread per (feature,
// class) adds the block sums. There are no atomic updates, so an evaluation
// gives the same bits on every run; it differs from the CPU's only by the
// rounding of a different summation order (and of the device's exp and log).
//
// usage: logreg_gpu <table.csv> [--C 1.0] [--max-iter 100] [--tol 1e-4] [options]
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cuda_compat.h"
#include "dataset.h"
#include "encode.h"
#include "lbfgs.h"
#include "logreg.h"

namespace {

using namespace ml;

constexpr uint32_t kBlockRecords = 256;

// Forward pass. index / value are column-major: [a * n + record]. W is
// [feature * C + class] with the bias as feature D. Writes
// dz[class * n + record] = P(class | record) - [class == y] and the loss
// -log P(y | record).
__global__ void k_forward(const int32_t* index, const float* value, const int32_t* numeric, const int32_t* y,
                          const double* W, double* dz, double* loss, size_t n, int d, int C, int D) {
    GRID_STRIDE(i, n) {
        double top = 0;
        for (int c = 0; c < C; ++c) {
            double z = W[static_cast<size_t>(D) * C + c];
            for (int a = 0; a < d; ++a) {
                const double w = W[static_cast<size_t>(index[a * n + i]) * C + c];
                z += numeric[a] ? w * static_cast<double>(value[a * n + i]) : w;
            }
            dz[c * n + i] = z;
            if (c == 0 || z > top) top = z;
        }
        // log-sum-exp, then dz = P(c | x) - [c == y].
        double sum = 0;
        for (int c = 0; c < C; ++c) sum += exp(dz[c * n + i] - top);
        const double lse = top + log(sum);
        loss[i] = lse - dz[static_cast<size_t>(y[i]) * n + i];
        for (int c = 0; c < C; ++c) dz[c * n + i] = exp(dz[c * n + i] - lse);
        dz[static_cast<size_t>(y[i]) * n + i] -= 1.0;
    }
}

// Backward pass, step 1. The positions [0, n * (d + 1)) list, attribute by
// attribute (the bias is attribute d), the records in feature order: record
// perm[p] for a categorical attribute, record p - a * n otherwise. Block k
// is the positions [block_pos[k], block_pos[k + 1]), all of one feature.
// partial[k * C + c] = sum over the block of dz * (value of the feature).
__global__ void k_grad_blocks(const double* dz, const float* value, const uint32_t* perm, const int32_t* numeric,
                              const uint32_t* block_pos, double* partial, size_t n, int d, int C, size_t n_blocks) {
    GRID_STRIDE(t, n_blocks * C) {
        const size_t k = t / C, c = t - k * C;
        const size_t begin = block_pos[k], end = block_pos[k + 1];
        const size_t a = begin / n, base = a * n;
        const double* z = dz + c * n;
        double s = 0;
        if (a == static_cast<size_t>(d)) {
            for (size_t p = begin; p < end; ++p) s += z[p - base];
        } else if (numeric[a]) {
            for (size_t p = begin; p < end; ++p) s += z[p - base] * static_cast<double>(value[p]);
        } else {
            for (size_t p = begin; p < end; ++p) s += z[perm[p]];
        }
        partial[t] = s;
    }
}

// Backward pass, step 2: g[f * C + c] = sum of the blocks of feature f.
__global__ void k_grad_features(const double* partial, const int32_t* feature_block, double* g, int C,
                                size_t n_features) {
    GRID_STRIDE(t, n_features * C) {
        const size_t f = t / C, c = t - f * C;
        double s = 0;
        for (int32_t k = feature_block[f]; k < feature_block[f + 1]; ++k) s += partial[static_cast<size_t>(k) * C + c];
        g[t] = s;
    }
}

// Sums of blocks of kBlockRecords consecutive losses.
__global__ void k_loss_blocks(const double* loss, double* out, size_t n, size_t n_blocks) {
    GRID_STRIDE(j, n_blocks) {
        const size_t begin = j * kBlockRecords, end = begin + kBlockRecords < n ? begin + kBlockRecords : n;
        double s = 0;
        for (size_t i = begin; i < end; ++i) s += loss[i];
        out[j] = s;
    }
}

// Class with the largest W_c . x + b_c (ties: lowest id).
__global__ void k_predict(const int32_t* index, const float* value, const int32_t* numeric, const double* W,
                          int32_t* out, size_t n, int d, int C, int D) {
    GRID_STRIDE(i, n) {
        int32_t best = 0;
        double best_z = 0;
        for (int c = 0; c < C; ++c) {
            double z = W[static_cast<size_t>(D) * C + c];
            for (int a = 0; a < d; ++a) {
                const double w = W[static_cast<size_t>(index[a * n + i]) * C + c];
                z += numeric[a] ? w * static_cast<double>(value[a * n + i]) : w;
            }
            if (c == 0 || z > best_z) {
                best = c;
                best_z = z;
            }
        }
        out[i] = best;
    }
}

LogRegModel fit(const Dataset& train, const LogRegOptions& opt) {
    LogRegModel model;
    model.n_classes = train.num_classes();
    model.encoder.fit(train, Scaling::Standardize);
    const Encoder& enc = model.encoder;
    const size_t n = train.num_rows();
    const int d = enc.n_attrs, C = model.n_classes, D = enc.n_features;
    if (static_cast<uint64_t>(n) * (d + 1) >= 0xFFFFFFFFull) throw std::runtime_error("training table too large");

    std::vector<int32_t> index;
    std::vector<float> value;
    enc.transform(train, index, value, /*column_major=*/true);
    std::vector<int32_t> numeric(enc.numeric.begin(), enc.numeric.end());

    // X by columns: per categorical attribute the records sorted by value
    // (counting sort), and the blocks of every feature.
    std::vector<uint32_t> perm(n * d, 0), block_pos, start;
    std::vector<int32_t> feature_block(D + 2, 0);
    auto add_blocks = [&](int feature, size_t begin, size_t end) {
        feature_block[feature] = static_cast<int32_t>(block_pos.size());
        for (size_t p = begin; p < end; p += kBlockRecords) block_pos.push_back(static_cast<uint32_t>(p));
    };
    for (int a = 0; a <= d; ++a) {
        const size_t base = static_cast<size_t>(a) * n;
        if (a == d || numeric[a]) {
            add_blocks(a == d ? D : enc.offset[a], base, base + n);
            continue;
        }
        const int first = enc.offset[a], V = (a + 1 < d ? enc.offset[a + 1] : D) - first;
        const int32_t* col = &index[base];
        start.assign(V + 1, 0);
        for (size_t i = 0; i < n; ++i) ++start[col[i] - first + 1];
        for (int v = 0; v < V; ++v) start[v + 1] += start[v];
        for (int v = 0; v < V; ++v) add_blocks(first + v, base + start[v], base + start[v + 1]);
        for (size_t i = 0; i < n; ++i) perm[base + start[col[i] - first]++] = static_cast<uint32_t>(i);
    }
    feature_block[D + 1] = static_cast<int32_t>(block_pos.size());
    const size_t n_blocks = block_pos.size();
    block_pos.push_back(static_cast<uint32_t>(n * (d + 1)));
    const size_t n_loss_blocks = (n + kBlockRecords - 1) / kBlockRecords;

    gpu::DevBuf<int32_t> d_index, d_numeric, d_y, d_feature_block;
    gpu::DevBuf<float> d_value;
    gpu::DevBuf<uint32_t> d_perm, d_block_pos;
    d_index.upload(index);
    d_value.upload(value);
    d_numeric.upload(numeric);
    d_y.upload(train.y);
    d_perm.upload(perm);
    d_block_pos.upload(block_pos);
    d_feature_block.upload(feature_block);
    const size_t n_weights = static_cast<size_t>(D + 1) * C;
    gpu::DevBuf<double> d_w(n_weights), d_g(n_weights), d_dz(n * C), d_loss(n), d_partial(n_blocks * C),
        d_loss_part(n_loss_blocks);
    std::vector<double> loss_part;

    model.weights.assign(n_weights, 0.0);
    auto objective = [&](const std::vector<double>& w, std::vector<double>& g) {
        d_w.upload(w);
        gpu::launch(k_forward, n, d_index.get(), d_value.get(), d_numeric.get(), d_y.get(), d_w.get(), d_dz.get(),
                    d_loss.get(), n, d, C, D);
        gpu::launch(k_grad_blocks, n_blocks * C, d_dz.get(), d_value.get(), d_perm.get(), d_numeric.get(),
                    d_block_pos.get(), d_partial.get(), n, d, C, n_blocks);
        gpu::launch(k_grad_features, n_weights, d_partial.get(), d_feature_block.get(), d_g.get(), C,
                    static_cast<size_t>(D + 1));
        gpu::launch(k_loss_blocks, n_loss_blocks, d_loss.get(), d_loss_part.get(), n, n_loss_blocks);
        d_g.download(g.data(), n_weights);
        d_loss_part.download(loss_part, n_loss_blocks);
        double loss = 0;
        for (double v : loss_part) loss += v;
        return logreg_finish_objective(w, g, loss, n, opt.C, D, C);
    };

    LbfgsOptions lo;
    lo.max_iter = opt.max_iter;
    lo.tol = opt.tol;
    model.result = lbfgs_minimize(model.weights, objective, lo);
    gpu::synchronize();
    return model;
}

std::vector<int32_t> predict(const LogRegModel& model, const Dataset& test) {
    const size_t n = test.num_rows();
    std::vector<int32_t> out(n);
    if (n == 0) return out;
    const Encoder& enc = model.encoder;
    std::vector<int32_t> index;
    std::vector<float> value;
    enc.transform(test, index, value, /*column_major=*/true);
    std::vector<int32_t> numeric(enc.numeric.begin(), enc.numeric.end());
    gpu::DevBuf<int32_t> d_index, d_numeric, d_out(n);
    gpu::DevBuf<float> d_value;
    gpu::DevBuf<double> d_w;
    d_index.upload(index);
    d_value.upload(value);
    d_numeric.upload(numeric);
    d_w.upload(model.weights);
    gpu::launch(k_predict, n, d_index.get(), d_value.get(), d_numeric.get(), d_w.get(), d_out.get(), n,
                enc.n_attrs, model.n_classes, enc.n_features);
    d_out.download(out.data(), n);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    LogRegBackend backend;
    backend.device.id = gpu::device_id();
    backend.device.name = gpu::device_name();
    backend.device.peak_bytes = &gpu::g_peak_bytes;
    backend.fit = fit;
    backend.predict = predict;
    gpu::warm_up();
    return logreg_main(argc, argv, backend);
}
