#include "dataset.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace ml {

int Dataset::num_numeric() const {
    int k = 0;
    for (const auto& a : attrs) k += a.numeric;
    return k;
}

int Dataset::num_categorical() const { return num_attrs() - num_numeric(); }

std::vector<size_t> Dataset::class_counts() const {
    std::vector<size_t> c(classes.size(), 0);
    for (int32_t v : y) ++c[v];
    return c;
}

namespace {

std::string_view trim(std::string_view s) {
    auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    while (!s.empty() && blank(s.front())) s.remove_prefix(1);
    while (!s.empty() && blank(s.back())) s.remove_suffix(1);
    return s;
}

// Splits text[begin, end) at commas into trimmed fields.
void split_line(const std::string& text, size_t begin, size_t end, std::vector<std::string_view>& out) {
    out.clear();
    size_t start = begin;
    for (size_t i = begin; i <= end; ++i) {
        if (i == end || text[i] == ',') {
            out.push_back(trim(std::string_view(text.data() + start, i - start)));
            start = i + 1;
        }
    }
}

std::string base_name(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string s = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = s.find_last_of('.');
    return dot == std::string::npos ? s : s.substr(0, dot);
}

[[noreturn]] void fail(const std::string& path, size_t line, const std::string& what) {
    throw std::runtime_error(path + ":" + std::to_string(line) + ": " + what);
}

}  // namespace

Dataset load_table(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t size = text.size();

    Dataset d;
    d.name = base_name(path);

    // ---- header
    size_t pos = 0;
    size_t eol = text.find('\n');
    if (eol == std::string::npos) eol = size;
    std::vector<std::string_view> fields;
    split_line(text, 0, eol, fields);
    pos = eol + 1;

    const int n_cols = static_cast<int>(fields.size());
    int class_col = -1;
    std::vector<int> attr_of(n_cols, -1);   // file column -> attribute index
    for (int c = 0; c < n_cols; ++c) {
        std::string_view f = fields[c];
        size_t colon = f.rfind(':');
        if (colon == std::string_view::npos) fail(path, 1, "header field '" + std::string(f) + "' is not name:type");
        std::string_view name = f.substr(0, colon), type = f.substr(colon + 1);
        if (type == "class") {
            if (class_col >= 0) fail(path, 1, "more than one class column");
            class_col = c;
            d.class_name = std::string(name);
        } else if (type == "num" || type == "cat") {
            attr_of[c] = d.num_attrs();
            Attribute a;
            a.name = std::string(name);
            a.numeric = type == "num";
            d.attrs.push_back(std::move(a));
        } else {
            fail(path, 1, "unknown column type '" + std::string(type) + "' (expected num, cat or class)");
        }
    }
    if (class_col < 0) fail(path, 1, "no class column");

    // ---- records, row-major into per-column vectors
    size_t approx_rows = 0;
    for (size_t i = pos; i < size; ++i) approx_rows += text[i] == '\n';
    ++approx_rows;

    const int n_attrs = d.num_attrs();
    std::vector<std::vector<float>> cols(n_attrs);
    for (auto& c : cols) c.reserve(approx_rows);
    d.y.reserve(approx_rows);

    // Dictionaries key on views into `text`, which outlives them.
    std::vector<std::unordered_map<std::string_view, int32_t>> dict(n_attrs);
    std::unordered_map<std::string_view, int32_t> class_dict;

    size_t line_no = 1;
    std::string num_buf;
    while (pos < size) {
        eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = size;
        ++line_no;
        if (trim(std::string_view(text.data() + pos, eol - pos)).empty()) {
            pos = eol + 1;
            continue;
        }
        split_line(text, pos, eol, fields);
        pos = eol + 1;
        if (static_cast<int>(fields.size()) != n_cols)
            fail(path, line_no, "expected " + std::to_string(n_cols) + " fields, found " + std::to_string(fields.size()));

        for (int c = 0; c < n_cols; ++c) {
            std::string_view f = fields[c];
            if (c == class_col) {
                if (f.empty() || f == "?") fail(path, line_no, "missing class label");
                auto it = class_dict.find(f);
                if (it == class_dict.end()) {
                    it = class_dict.emplace(f, static_cast<int32_t>(d.classes.size())).first;
                    d.classes.emplace_back(f);
                }
                d.y.push_back(it->second);
                continue;
            }
            const int a = attr_of[c];
            Attribute& attr = d.attrs[a];
            const bool missing = f.empty() || f == "?";
            if (missing) ++attr.missing;
            if (attr.numeric) {
                float v = std::numeric_limits<float>::quiet_NaN();
                if (!missing) {
                    num_buf.assign(f.data(), f.size());
                    char* end = nullptr;
                    errno = 0;
                    v = std::strtof(num_buf.c_str(), &end);
                    if (end != num_buf.c_str() + num_buf.size() || !std::isfinite(v))
                        fail(path, line_no, "'" + num_buf + "' is not a number (column " + attr.name + ")");
                }
                cols[a].push_back(v);
            } else {
                if (missing) f = "?";
                auto it = dict[a].find(f);
                if (it == dict[a].end()) {
                    int32_t id = attr.num_values();
                    // "?" may come from an empty field, so key on the stored label.
                    attr.values.emplace_back(f);
                    if (missing) attr.missing_code = id;
                    it = dict[a].emplace(missing ? std::string_view("?") : f, id).first;
                }
                cols[a].push_back(static_cast<float>(it->second));
            }
        }
    }

    d.n = d.y.size();
    if (d.n == 0) throw std::runtime_error(path + ": no records");
    d.x.resize(static_cast<size_t>(n_attrs) * d.n);
    for (int a = 0; a < n_attrs; ++a) {
        std::copy(cols[a].begin(), cols[a].end(), d.x.begin() + static_cast<size_t>(a) * d.n);
        std::vector<float>().swap(cols[a]);
    }
    return d;
}

Dataset take_rows(const Dataset& data, const std::vector<uint32_t>& rows) {
    Dataset s;
    s.name = data.name;
    s.attrs = data.attrs;
    s.class_name = data.class_name;
    s.classes = data.classes;
    s.n = rows.size();
    s.x.resize(static_cast<size_t>(data.num_attrs()) * s.n);
    s.y.resize(s.n);
    for (int a = 0; a < data.num_attrs(); ++a) {
        const float* src = data.column(a);
        float* dst = s.x.data() + static_cast<size_t>(a) * s.n;
        for (size_t i = 0; i < s.n; ++i) dst[i] = src[rows[i]];
    }
    for (size_t i = 0; i < s.n; ++i) s.y[i] = data.y[rows[i]];
    // Missing counts describe the subset, not the parent table.
    for (int a = 0; a < s.num_attrs(); ++a) {
        Attribute& attr = s.attrs[a];
        attr.missing = 0;
        const float* col = s.column(a);
        for (size_t i = 0; i < s.n; ++i) {
            if (attr.numeric ? std::isnan(col[i]) : (attr.missing_code >= 0 && col[i] == attr.missing_code))
                ++attr.missing;
        }
    }
    return s;
}

}  // namespace ml
