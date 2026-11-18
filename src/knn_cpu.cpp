// k-nearest-neighbor classification on the CPU (Han & Kamber, Section 9.5.1).
// The distance and the voting rule are described in include/knn_app.h.
//
// For every test record the distances to all training records are
// accumulated one attribute at a time (the training table is stored column
// by column), which keeps the inner loop a plain array operation. One scan
// then finds the distance of the k-th nearest record and a second scan
// counts the votes.
//
// usage: knn_cpu <table.csv> [--k 5] [options]
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "dataset.h"
#include "encode.h"
#include "eval.h"
#include "knn_app.h"

namespace {

using namespace ml;

class KnnCpu : public Classifier {
public:
    explicit KnnCpu(int k) : k_(k) {}
    std::string name() const override { return "knn"; }

    void fit(const Dataset& train) override {
        n_train_ = train.num_rows();
        n_classes_ = train.num_classes();
        encoder_.fit(train, Scaling::MinMax);
        encoder_.transform(train, index_, value_, /*column_major=*/true);
        y_ = train.y;
    }

    std::vector<int32_t> predict(const Dataset& test) override {
        std::vector<int32_t> q_index;
        std::vector<float> q_value;
        encoder_.transform(test, q_index, q_value);
        const size_t n_test = test.num_rows(), n = n_train_;
        const int d = encoder_.n_attrs;
        const int k = static_cast<int>(std::min<size_t>(k_, n));
        const float inf = std::numeric_limits<float>::infinity();

        std::vector<float> dist(n);
        std::vector<float> best_dist(k);
        std::vector<int32_t> votes(n_classes_);
        std::vector<int32_t> out(n_test);
        for (size_t i = 0; i < n_test; ++i) {
            std::fill(dist.begin(), dist.end(), 0.0f);
            for (int a = 0; a < d; ++a) {
                if (encoder_.numeric[a]) {
                    const float* t = &value_[static_cast<size_t>(a) * n];
                    const float q = q_value[i * d + a];
                    for (size_t j = 0; j < n; ++j) {
                        const float diff = t[j] - q;
                        dist[j] += diff * diff;
                    }
                } else {
                    const int32_t* t = &index_[static_cast<size_t>(a) * n];
                    const int32_t q = q_index[i * d + a];
                    for (size_t j = 0; j < n; ++j) dist[j] += t[j] != q ? 1.0f : 0.0f;
                }
            }
            // Distance of the k-th nearest record: the largest of the k smallest
            // distances, which are kept sorted.
            std::fill(best_dist.begin(), best_dist.end(), inf);
            for (size_t j = 0; j < n; ++j) {
                const float dj = dist[j];
                if (!(dj < best_dist[k - 1])) continue;
                int p = k - 1;
                while (p > 0 && best_dist[p - 1] > dj) {
                    best_dist[p] = best_dist[p - 1];
                    --p;
                }
                best_dist[p] = dj;
            }
            const float radius = best_dist[k - 1];
            std::fill(votes.begin(), votes.end(), 0);
            for (size_t j = 0; j < n; ++j)
                if (dist[j] <= radius) ++votes[y_[j]];
            out[i] = static_cast<int32_t>(std::max_element(votes.begin(), votes.end()) - votes.begin());
        }
        return out;
    }

    std::vector<std::pair<std::string, double>> info() const override {
        return {{"k", static_cast<double>(k_)}, {"train_records", static_cast<double>(n_train_)}};
    }

private:
    int k_;
    size_t n_train_ = 0;
    int n_classes_ = 0;
    Encoder encoder_;
    std::vector<int32_t> index_;    // column-major: index_[a * n + j]
    std::vector<float> value_;
    std::vector<int32_t> y_;
};

}  // namespace

int main(int argc, char** argv) {
    KnnBackend backend;
    backend.make = [](int k) { return std::unique_ptr<Classifier>(new KnnCpu(k)); };
    return knn_main(argc, argv, backend);
}
