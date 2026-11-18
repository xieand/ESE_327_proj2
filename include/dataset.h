// Classification tables shared by every classifier (CPU and CUDA).
//
// File format (written by scripts/fetch_uci.py): comma-separated, one record
// per line, no quoting. The header gives each column as "name:type" with type
//   num    numeric attribute
//   cat    categorical attribute
//   class  class label (exactly one column, in any position)
// A missing value is written "?".
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml {

struct Attribute {
    std::string name;
    bool numeric = false;
    // Categorical only: value labels, indexed by code, in order of first
    // appearance. A missing value is the ordinary category "?".
    std::vector<std::string> values;
    // Categorical: code of "?" or -1 if the column has no missing values.
    int32_t missing_code = -1;
    // Number of records with a missing value in this column.
    size_t missing = 0;

    int num_values() const { return static_cast<int>(values.size()); }
};

// Column-major table. Every cell is a float: numeric attributes hold their
// value (NaN when missing), categorical attributes hold their code
// (0 .. num_values-1, exact in a float). One flat array keeps the layout
// identical on the host and on the GPU.
struct Dataset {
    std::string name;
    std::vector<Attribute> attrs;
    std::string class_name;
    std::vector<std::string> classes;   // class labels, indexed by class id
    size_t n = 0;                       // number of records
    std::vector<float> x;               // x[a * n + i] = attribute a of record i
    std::vector<int32_t> y;             // class id of every record

    size_t num_rows() const { return n; }
    int num_attrs() const { return static_cast<int>(attrs.size()); }
    int num_classes() const { return static_cast<int>(classes.size()); }
    int num_numeric() const;
    int num_categorical() const;

    const float* column(int a) const { return x.data() + static_cast<size_t>(a) * n; }
    float value(size_t i, int a) const { return x[static_cast<size_t>(a) * n + i]; }
    int32_t code(size_t i, int a) const { return static_cast<int32_t>(value(i, a)); }

    std::vector<size_t> class_counts() const;
};

// Throws std::runtime_error on I/O or format errors.
Dataset load_table(const std::string& path);

// The records rows[0], rows[1], ... as a new table. Attribute dictionaries
// and class ids are kept, so codes mean the same in every subset.
Dataset take_rows(const Dataset& data, const std::vector<uint32_t>& rows);

}  // namespace ml
