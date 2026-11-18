#include "encode.h"

#include <cmath>

namespace ml {

void Encoder::fit(const Dataset& train, Scaling scaling) {
    n_attrs = train.num_attrs();
    const size_t n = train.num_rows();
    offset.assign(n_attrs, 0);
    numeric.assign(n_attrs, 0);
    shift.assign(n_attrs, 0.0);
    scale.assign(n_attrs, 1.0);
    fill.assign(n_attrs, 0.0);
    n_features = 0;
    for (int a = 0; a < n_attrs; ++a) {
        const Attribute& attr = train.attrs[a];
        offset[a] = n_features;
        numeric[a] = attr.numeric;
        if (!attr.numeric) {
            n_features += attr.num_values();
            continue;
        }
        n_features += 1;
        const float* col = train.column(a);
        double sum = 0, lo = 0, hi = 0;
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(col[i])) continue;
            const double v = col[i];
            sum += v;
            if (count == 0 || v < lo) lo = v;
            if (count == 0 || v > hi) hi = v;
            ++count;
        }
        const double mean = count ? sum / count : 0.0;
        fill[a] = mean;
        if (scaling == Scaling::MinMax) {
            shift[a] = lo;
            if (hi > lo) scale[a] = hi - lo;
        } else {
            double ss = 0;
            for (size_t i = 0; i < n; ++i)
                if (!std::isnan(col[i])) ss += (col[i] - mean) * (col[i] - mean);
            const double sd = count ? std::sqrt(ss / count) : 0.0;
            shift[a] = mean;
            if (sd > 0) scale[a] = sd;
        }
    }
}

void Encoder::transform(const Dataset& data, std::vector<int32_t>& index, std::vector<float>& value,
                        bool column_major) const {
    const size_t n = data.num_rows();
    index.resize(n * n_attrs);
    value.resize(n * n_attrs);
    for (int a = 0; a < n_attrs; ++a) {
        const float* col = data.column(a);
        for (size_t i = 0; i < n; ++i) {
            const size_t k = column_major ? static_cast<size_t>(a) * n + i : i * n_attrs + a;
            if (numeric[a]) {
                const double v = std::isnan(col[i]) ? fill[a] : static_cast<double>(col[i]);
                index[k] = offset[a];
                value[k] = static_cast<float>((v - shift[a]) / scale[a]);
            } else {
                index[k] = offset[a] + static_cast<int32_t>(col[i]);
                value[k] = 1.0f;
            }
        }
    }
}

std::vector<std::string> Encoder::feature_names(const Dataset& schema) const {
    std::vector<std::string> names;
    for (int a = 0; a < n_attrs; ++a) {
        const Attribute& attr = schema.attrs[a];
        if (attr.numeric) names.push_back(attr.name);
        else
            for (const std::string& v : attr.values) names.push_back(attr.name + "=" + v);
    }
    return names;
}

}  // namespace ml
