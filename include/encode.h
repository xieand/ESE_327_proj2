// Feature encoding shared by k-nearest-neighbors and logistic regression
// (CPU and CUDA).
//
// A categorical attribute becomes one indicator feature per value (one-hot)
// and a numeric attribute becomes one scaled feature. A one-hot matrix is
// never stored: every record has exactly one active feature per attribute,
// so a record is n_attrs pairs (feature index, value):
//
//   categorical  (offset[a] + code, 1)
//   numeric      (offset[a], (x - shift[a]) / scale[a])
//
// A missing categorical value is the ordinary category "?". A missing numeric
// value is replaced by the attribute's training mean.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dataset.h"

namespace ml {

enum class Scaling {
    Standardize,   // (x - mean) / standard deviation      (logistic regression)
    MinMax,        // (x - min) / (max - min), in [0, 1]   (k-nearest-neighbors)
};

struct Encoder {
    int n_attrs = 0;
    int n_features = 0;
    std::vector<int32_t> offset;     // first feature of every attribute
    std::vector<uint8_t> numeric;    // per attribute
    std::vector<double> shift, scale, fill;   // per attribute (numeric only)

    // Statistics come from the training table only. A constant attribute
    // gets scale 1 (its feature is then 0 for every training record).
    void fit(const Dataset& train, Scaling scaling);

    // index[r * n_attrs + a], value[r * n_attrs + a] for record r (row-major),
    // or index[a * n + r], value[a * n + r] with column_major.
    void transform(const Dataset& data, std::vector<int32_t>& index, std::vector<float>& value,
                   bool column_major = false) const;

    // "attribute" or "attribute=value" for every feature.
    std::vector<std::string> feature_names(const Dataset& schema) const;
};

}  // namespace ml
