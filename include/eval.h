// Evaluation shared by every classifier: the classifier interface, stratified
// train/test splits, metrics, and the timed cross-validation loop.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "dataset.h"

namespace ml {

// ------------------------------------------------------------------ splits

struct Split {
    std::vector<uint32_t> train, test;   // record indices, ascending
};

// Fold id (0 .. k-1) of every record. Each class is shuffled and dealt
// round-robin over the folds, and the deal continues from one class to the
// next, so every fold gets floor or ceil of (class size / k) records of each
// class and fold sizes differ by at most one.
std::vector<int32_t> stratified_fold_ids(const std::vector<int32_t>& y, int n_classes, int k, uint64_t seed);

// k splits; split f tests on fold f and trains on the other folds.
std::vector<Split> stratified_kfold(const std::vector<int32_t>& y, int n_classes, int k, uint64_t seed);

// One split with about test_fraction of every class in the test set.
Split stratified_holdout(const std::vector<int32_t>& y, int n_classes, double test_fraction, uint64_t seed);

// ----------------------------------------------------------------- metrics

struct Metrics {
    int n_classes = 0;
    size_t n = 0;
    size_t correct = 0;
    std::vector<size_t> confusion;   // confusion[actual * n_classes + predicted]

    double accuracy() const { return n ? static_cast<double>(correct) / n : 0.0; }
    // Unweighted mean F1 over the classes that occur in the truth or in the
    // predictions (a class with no true positives has F1 = 0).
    double macro_f1() const;
    void add(const Metrics& other);
};

Metrics compute_metrics(const std::vector<int32_t>& truth, const std::vector<int32_t>& predicted, int n_classes);

// -------------------------------------------------------------- classifier

class Classifier {
public:
    virtual ~Classifier() = default;
    virtual std::string name() const = 0;
    virtual void fit(const Dataset& train) = 0;
    // Class id for every record of `test` (same schema as the training table).
    virtual std::vector<int32_t> predict(const Dataset& test) = 0;
};

// Predicts the most frequent training class (ties: lowest class id). The
// floor any real classifier has to beat.
class MajorityClassifier : public Classifier {
public:
    std::string name() const override { return "majority"; }
    void fit(const Dataset& train) override;
    std::vector<int32_t> predict(const Dataset& test) override;

private:
    int32_t label_ = 0;
};

// -------------------------------------------------------------- evaluation

struct FoldResult {
    size_t n_train = 0, n_test = 0;
    double train_ms = 0, predict_ms = 0;
    Metrics metrics;
};

struct EvalResult {
    std::string classifier;
    std::vector<FoldResult> folds;
    Metrics pooled;                  // all test predictions together
    double mean_accuracy = 0;        // mean of the per-fold accuracies
    double std_accuracy = 0;         // sample standard deviation over folds
    double train_ms = 0;             // summed over folds
    double predict_ms = 0;
};

using ClassifierFactory = std::function<std::unique_ptr<Classifier>()>;

// Trains a fresh classifier on every split's training records and tests it on
// the split's test records. Times cover fit() and predict() only, not the
// row gathering.
EvalResult evaluate(const Dataset& data, const std::vector<Split>& splits, const ClassifierFactory& make);

}  // namespace ml
