#include "eval.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "util.h"

namespace ml {

namespace {

// Record indices of every class, each list shuffled.
std::vector<std::vector<uint32_t>> shuffled_by_class(const std::vector<int32_t>& y, int n_classes, uint64_t seed) {
    std::vector<std::vector<uint32_t>> by_class(n_classes);
    for (size_t i = 0; i < y.size(); ++i) by_class[y[i]].push_back(static_cast<uint32_t>(i));
    Rng rng(seed);
    for (auto& rows : by_class) rng.shuffle(rows);
    return by_class;
}

}  // namespace

std::vector<int32_t> stratified_fold_ids(const std::vector<int32_t>& y, int n_classes, int k, uint64_t seed) {
    if (k < 2) throw std::runtime_error("cross-validation needs at least 2 folds");
    if (y.size() < static_cast<size_t>(k)) throw std::runtime_error("fewer records than folds");
    std::vector<int32_t> fold(y.size());
    int next = 0;
    for (const auto& rows : shuffled_by_class(y, n_classes, seed)) {
        for (uint32_t i : rows) {
            fold[i] = next;
            next = (next + 1) % k;
        }
    }
    return fold;
}

std::vector<Split> stratified_kfold(const std::vector<int32_t>& y, int n_classes, int k, uint64_t seed) {
    std::vector<int32_t> fold = stratified_fold_ids(y, n_classes, k, seed);
    std::vector<Split> splits(k);
    for (size_t i = 0; i < y.size(); ++i) {
        for (int f = 0; f < k; ++f) {
            auto& part = f == fold[i] ? splits[f].test : splits[f].train;
            part.push_back(static_cast<uint32_t>(i));
        }
    }
    return splits;
}

Split stratified_holdout(const std::vector<int32_t>& y, int n_classes, double test_fraction, uint64_t seed) {
    if (!(test_fraction > 0.0 && test_fraction < 1.0)) throw std::runtime_error("holdout fraction must be in (0, 1)");
    std::vector<char> is_test(y.size(), 0);
    for (const auto& rows : shuffled_by_class(y, n_classes, seed)) {
        size_t n_test = static_cast<size_t>(std::floor(test_fraction * rows.size() + 0.5));
        for (size_t j = 0; j < n_test; ++j) is_test[rows[j]] = 1;
    }
    Split s;
    for (size_t i = 0; i < y.size(); ++i) (is_test[i] ? s.test : s.train).push_back(static_cast<uint32_t>(i));
    if (s.train.empty() || s.test.empty()) throw std::runtime_error("holdout split left one side empty");
    return s;
}

double Metrics::macro_f1() const {
    double sum = 0;
    int used = 0;
    for (int c = 0; c < n_classes; ++c) {
        size_t tp = confusion[static_cast<size_t>(c) * n_classes + c];
        size_t actual = 0, predicted = 0;
        for (int j = 0; j < n_classes; ++j) {
            actual += confusion[static_cast<size_t>(c) * n_classes + j];
            predicted += confusion[static_cast<size_t>(j) * n_classes + c];
        }
        if (actual + predicted == 0) continue;
        sum += 2.0 * tp / static_cast<double>(actual + predicted);
        ++used;
    }
    return used ? sum / used : 0.0;
}

void Metrics::add(const Metrics& other) {
    if (confusion.empty()) {
        *this = other;
        return;
    }
    if (other.n_classes != n_classes) throw std::runtime_error("metrics with different class counts");
    n += other.n;
    correct += other.correct;
    for (size_t i = 0; i < confusion.size(); ++i) confusion[i] += other.confusion[i];
}

Metrics compute_metrics(const std::vector<int32_t>& truth, const std::vector<int32_t>& predicted, int n_classes) {
    if (truth.size() != predicted.size()) throw std::runtime_error("prediction count does not match record count");
    Metrics m;
    m.n_classes = n_classes;
    m.n = truth.size();
    m.confusion.assign(static_cast<size_t>(n_classes) * n_classes, 0);
    for (size_t i = 0; i < truth.size(); ++i) {
        if (predicted[i] < 0 || predicted[i] >= n_classes) throw std::runtime_error("predicted class id out of range");
        ++m.confusion[static_cast<size_t>(truth[i]) * n_classes + predicted[i]];
        m.correct += truth[i] == predicted[i];
    }
    return m;
}

void MajorityClassifier::fit(const Dataset& train) {
    std::vector<size_t> counts = train.class_counts();
    label_ = static_cast<int32_t>(std::max_element(counts.begin(), counts.end()) - counts.begin());
}

std::vector<int32_t> MajorityClassifier::predict(const Dataset& test) {
    return std::vector<int32_t>(test.num_rows(), label_);
}

EvalResult evaluate(const Dataset& data, const std::vector<Split>& splits, const ClassifierFactory& make) {
    EvalResult r;
    for (const Split& s : splits) {
        Dataset train = take_rows(data, s.train);
        Dataset test = take_rows(data, s.test);
        std::unique_ptr<Classifier> clf = make();
        if (r.classifier.empty()) r.classifier = clf->name();

        FoldResult f;
        f.n_train = train.num_rows();
        f.n_test = test.num_rows();
        Timer t;
        clf->fit(train);
        f.train_ms = t.ms();
        t.reset();
        std::vector<int32_t> pred = clf->predict(test);
        f.predict_ms = t.ms();
        f.metrics = compute_metrics(test.y, pred, data.num_classes());

        r.train_ms += f.train_ms;
        r.predict_ms += f.predict_ms;
        r.pooled.add(f.metrics);
        r.folds.push_back(std::move(f));
    }
    const size_t k = r.folds.size();
    for (const auto& f : r.folds) r.mean_accuracy += f.metrics.accuracy();
    if (k) r.mean_accuracy /= k;
    if (k > 1) {
        double ss = 0;
        for (const auto& f : r.folds) ss += std::pow(f.metrics.accuracy() - r.mean_accuracy, 2);
        r.std_accuracy = std::sqrt(ss / (k - 1));
    }
    return r;
}

}  // namespace ml
