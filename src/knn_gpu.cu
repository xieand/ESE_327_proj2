// k-nearest-neighbor classification on the GPU. The distance and the voting
// rule are described in include/knn_app.h; the result is identical to
// knn_cpu, record for record.
//
// The search is exhaustive. The training records are cut into blocks of
// kBlockRecords, and the unit of parallel work is the pair (test record,
// training block):
//
//   1. nearest  One thread per pair computes the distances of its block and
//               keeps the k smallest in a short sorted list.
//   2. radius   One thread per test record merges the lists of its blocks;
//               the k-th smallest distance overall is the distance of the
//               k-th nearest record.
//   3. vote     One thread per pair recomputes the distances of its block;
//               every training record within the radius votes (all records
//               tied with the k-th vote), with an atomic increment.
//   4. label    One thread per test record picks the class with most votes.
//
// Recomputing the distances costs a second pass but needs no memory per
// (test, training) pair, so the training set can be of any size. Threads
// that run side by side work on the same training block for different test
// records, so the block is read from the cache. The distance of a pair is
// accumulated in attribute order in single precision without fused
// multiply-add, exactly as on the CPU, so both find the same neighbors,
// ties included.
//
// A launch handles as many test records as fit in a fixed amount of work,
// which keeps every kernel short (a long-running kernel would stall a GPU
// that also drives the display).
//
// usage: knn_gpu <table.csv> [--k 5] [options]
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "cuda_compat.h"
#include "dataset.h"
#include "encode.h"
#include "eval.h"
#include "knn_app.h"

namespace {

using namespace ml;

constexpr size_t kBlockRecords = 512;
// (test record, training record, attribute) triples per kernel launch.
constexpr double kWorkPerLaunch = 2e10;
// Device memory for the lists of nearest distances of one launch.
constexpr size_t kListBytes = size_t(128) << 20;

// Squared distance between a training record and a test record, both rows
// of d cells: the scaled value of a numeric attribute, the value code of a
// categorical one.
__device__ inline float distance(const float* t, const float* q, const int32_t* numeric, int d) {
    float dist = 0.0f;
    for (int a = 0; a < d; ++a) {
        if (numeric[a]) {
            const float diff = t[a] - q[a];
            dist += diff * diff;
        } else {
            dist += t[a] != q[a] ? 1.0f : 0.0f;
        }
    }
    return dist;
}

// train[j * d + a], query[i * d + a]. Thread idx handles training block
// idx / n_query and test record idx % n_query.

// 1. best[(i * n_blocks + t) * k ...]: the k smallest distances between test
//    record i and the records of training block t, ascending.
__global__ void k_nearest(const float* train, const float* query, const int32_t* numeric, float* best,
                          size_t n_train, size_t n_query, size_t n_blocks, int d, int k, float infinity) {
    GRID_STRIDE(idx, n_blocks * n_query) {
        const size_t t = idx / n_query, i = idx - t * n_query;
        const float* q = query + i * d;
        float* b = best + (i * n_blocks + t) * k;
        for (int p = 0; p < k; ++p) b[p] = infinity;
        float radius = infinity;   // b[k - 1]
        const size_t begin = t * kBlockRecords, end = begin + kBlockRecords < n_train ? begin + kBlockRecords : n_train;
        for (size_t j = begin; j < end; ++j) {
            const float dist = distance(train + j * d, q, numeric, d);
            if (!(dist < radius)) continue;
            int p = k - 1;
            while (p > 0 && b[p - 1] > dist) {
                b[p] = b[p - 1];
                --p;
            }
            b[p] = dist;
            radius = b[k - 1];
        }
    }
}

// 2. radius[i]: the k-th smallest distance of test record i. The lists of
//    the other blocks are merged into the list of block 0.
__global__ void k_radius(float* best, float* radius, size_t n_query, size_t n_blocks, int k) {
    GRID_STRIDE(i, n_query) {
        float* b = best + i * n_blocks * k;
        for (size_t t = 1; t < n_blocks; ++t) {
            const float* other = b + t * k;
            for (int o = 0; o < k; ++o) {
                const float dist = other[o];
                if (!(dist < b[k - 1])) break;   // the rest of this list is larger still
                int p = k - 1;
                while (p > 0 && b[p - 1] > dist) {
                    b[p] = b[p - 1];
                    --p;
                }
                b[p] = dist;
            }
        }
        radius[i] = b[k - 1];
    }
}

// 3. votes[i * C + c]: training records of class c within radius[i] of test record i.
__global__ void k_vote(const float* train, const int32_t* y, const float* query, const int32_t* numeric,
                       const float* radius, uint32_t* votes, size_t n_train, size_t n_query, size_t n_blocks, int d,
                       int C) {
    GRID_STRIDE(idx, n_blocks * n_query) {
        const size_t t = idx / n_query, i = idx - t * n_query;
        const float* q = query + i * d;
        const float r = radius[i];
        const size_t begin = t * kBlockRecords, end = begin + kBlockRecords < n_train ? begin + kBlockRecords : n_train;
        for (size_t j = begin; j < end; ++j)
            if (distance(train + j * d, q, numeric, d) <= r) atomicAdd(&votes[i * C + y[j]], 1u);
    }
}

// 4. Class with the most votes (ties: lowest id).
__global__ void k_label(const uint32_t* votes, int32_t* out, size_t n_query, int C) {
    GRID_STRIDE(i, n_query) {
        const uint32_t* v = votes + i * C;
        int32_t top = 0;
        for (int c = 1; c < C; ++c)
            if (v[c] > v[top]) top = c;
        out[i] = top;
    }
}

class KnnGpu : public Classifier {
public:
    explicit KnnGpu(int k) : k_(k) {}
    std::string name() const override { return "knn"; }

    void fit(const Dataset& train) override {
        n_train_ = train.num_rows();
        n_classes_ = train.num_classes();
        encoder_.fit(train, Scaling::MinMax);
        std::vector<int32_t> numeric(encoder_.numeric.begin(), encoder_.numeric.end());
        d_train_.upload(cells(train));
        d_y_.upload(train.y);
        d_numeric_.upload(numeric);
        gpu::synchronize();
    }

    std::vector<int32_t> predict(const Dataset& test) override {
        const size_t n_test = test.num_rows(), n = n_train_;
        const int d = encoder_.n_attrs, C = n_classes_;
        const int k = static_cast<int>(std::min<size_t>(k_, n));
        std::vector<int32_t> out(n_test);
        if (n_test == 0) return out;

        const size_t n_blocks = (n + kBlockRecords - 1) / kBlockRecords;
        const double per_query = static_cast<double>(n) * std::max(d, 1);
        const size_t list_bytes = n_blocks * k * sizeof(float);
        const size_t batch = std::max<size_t>(
            1, std::min<size_t>({n_test, static_cast<size_t>(kWorkPerLaunch / per_query), kListBytes / list_bytes}));
        const float infinity = std::numeric_limits<float>::infinity();
        gpu::DevBuf<float> d_query, d_best(batch * n_blocks * k), d_radius(batch);
        gpu::DevBuf<uint32_t> d_votes(batch * C);
        gpu::DevBuf<int32_t> d_out(n_test);
        d_query.upload(cells(test));
        for (size_t begin = 0; begin < n_test; begin += batch) {
            const size_t count = std::min(batch, n_test - begin);
            const float* q = d_query.get() + begin * d;
            gpu::launch(k_nearest, count * n_blocks, d_train_.get(), q, d_numeric_.get(), d_best.get(), n, count,
                        n_blocks, d, k, infinity);
            gpu::launch(k_radius, count, d_best.get(), d_radius.get(), count, n_blocks, k);
            gpu::synchronize();
            d_votes.zero(count * C);
            gpu::launch(k_vote, count * n_blocks, d_train_.get(), d_y_.get(), q, d_numeric_.get(), d_radius.get(),
                        d_votes.get(), n, count, n_blocks, d, C);
            gpu::launch(k_label, count, d_votes.get(), d_out.get() + begin, count, C);
            gpu::synchronize();
        }
        d_out.download(out.data(), n_test);
        return out;
    }

    std::vector<std::pair<std::string, double>> info() const override {
        return {{"k", static_cast<double>(k_)}, {"train_records", static_cast<double>(n_train_)}};
    }

private:
    // One row of n_attrs cells per record: the scaled value of a numeric
    // attribute, the feature index of a categorical value (only compared
    // for equality).
    std::vector<float> cells(const Dataset& data) const {
        std::vector<int32_t> index;
        std::vector<float> value;
        encoder_.transform(data, index, value);
        const int d = encoder_.n_attrs;
        for (size_t i = 0; i < value.size(); ++i)
            if (!encoder_.numeric[i % d]) value[i] = static_cast<float>(index[i]);
        return value;
    }

    int k_;
    size_t n_train_ = 0;
    int n_classes_ = 0;
    Encoder encoder_;
    gpu::DevBuf<float> d_train_;
    gpu::DevBuf<int32_t> d_y_, d_numeric_;
};

}  // namespace

int main(int argc, char** argv) {
    KnnBackend backend;
    backend.device.id = gpu::device_id();
    backend.device.name = gpu::device_name();
    backend.device.peak_bytes = &gpu::g_peak_bytes;
    backend.make = [](int k) { return std::unique_ptr<Classifier>(new KnnGpu(k)); };
    gpu::warm_up();
    return knn_main(argc, argv, backend);
}
