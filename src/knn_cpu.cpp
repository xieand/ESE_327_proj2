// k-nearest-neighbor classification on the CPU (Han & Kamber, Section 9.5.1).
//
// Training only stores the records. A test record gets the most common class
// among the training records closest to it, with the squared distance
//
//   dist(X, Y) = sum over attributes a of
//       numeric      (x_a - y_a)^2, both scaled to [0, 1] by the training minimum and maximum
//       categorical  0 if the values are equal, 1 otherwise
//
// The voters are all training records that are at most as far away as the
// k-th nearest one. That is exactly k records when distances are distinct;
// records tied with the k-th all vote, so the result does not depend on the
// order of the training records. (Categorical attributes make ties the rule
// rather than the exception: the distance is then a count of mismatches.)
// Of two classes with the same number of votes, the lower class id wins.
//
// The search is exhaustive (every test record against every training
// record), in single precision. Distances are accumulated one attribute at a
// time over all training records, which keeps the inner loop a plain array
// operation and fixes the order of the floating-point additions.
//
// usage: knn_cpu <table.csv> [--k 5] [options]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "encode.h"
#include "eval.h"
#include "runner.h"
#include "util.h"

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

[[noreturn]] void usage(const char* prog, const std::string& problem = "") {
    if (!problem.empty()) std::fprintf(stderr, "error: %s\n", problem.c_str());
    std::fprintf(stderr,
                 "usage: %s <table.csv> [options]\n"
                 "  --k K                number of neighbors (default 5)\n"
                 "%s",
                 prog, run_options_help());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    int k = 5;
    RunOptions run_opt;
    try {
        Args args(argc, argv);
        if (args.flag("-h") || args.flag("--help")) usage(argv[0]);
        parse_run_options(args, run_opt);
        args.value("--k", k);
        if (k < 1) usage(argv[0], "--k must be at least 1");
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
        info.algorithm = "knn";
        info.device = "cpu";
        info.options = "k=" + std::to_string(k);
        info.read_ms = timer.ms();
        run_evaluation(data, run_opt, info, [&] { return std::unique_ptr<Classifier>(new KnnCpu(k)); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
