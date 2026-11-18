// Loads a classification table and reports what the shared infrastructure
// sees: schema, class distribution, and the stratified split with the
// majority-class baseline run through the common evaluation loop.
//
// usage: dataset_info <table.csv> [--folds K | --holdout F] [--seed S]
//                     [--dump-folds file] [--json]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "eval.h"
#include "util.h"

namespace {

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <table.csv> [options]\n"
                 "  --folds K          stratified K-fold cross-validation (default 10)\n"
                 "  --holdout F        one stratified split with fraction F as test set instead\n"
                 "  --seed S           seed of the split (default 327)\n"
                 "  --dump-folds file  write the fold id of every record, one per line\n"
                 "                     (with --holdout: 1 = test, 0 = train)\n"
                 "  --json             print everything as one JSON line\n",
                 prog);
    std::exit(2);
}

struct ColumnStats {
    double sum = 0, min = 0, max = 0;   // numeric
    std::vector<size_t> counts;         // categorical: records per value
};

ColumnStats column_stats(const ml::Dataset& d, int a) {
    ColumnStats s;
    const float* col = d.column(a);
    if (d.attrs[a].numeric) {
        bool first = true;
        for (size_t i = 0; i < d.n; ++i) {
            if (std::isnan(col[i])) continue;
            double v = col[i];
            s.sum += v;
            if (first || v < s.min) s.min = v;
            if (first || v > s.max) s.max = v;
            first = false;
        }
    } else {
        s.counts.assign(d.attrs[a].values.size(), 0);
        for (size_t i = 0; i < d.n; ++i) ++s.counts[static_cast<size_t>(col[i])];
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, dump;
    int folds = 10;
    double holdout = 0;
    unsigned long long seed = 327;
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        try {
            if (a == "--folds") folds = std::stoi(next());
            else if (a == "--holdout") holdout = std::stod(next());
            else if (a == "--seed") seed = std::stoull(next());
            else if (a == "--dump-folds") dump = next();
            else if (a == "--json") json = true;
            else if (!a.empty() && a[0] == '-') usage(argv[0]);
            else if (input.empty()) input = a;
            else usage(argv[0]);
        } catch (const std::logic_error&) {
            usage(argv[0]);
        }
    }
    if (input.empty()) usage(argv[0]);

    try {
        ml::Timer timer;
        ml::Dataset d = ml::load_table(input);
        const double read_ms = timer.ms();
        const int n_classes = d.num_classes();
        std::vector<size_t> class_counts = d.class_counts();

        // ---- split
        std::vector<ml::Split> splits;
        std::vector<int32_t> fold_of;
        if (holdout > 0) {
            splits.push_back(ml::stratified_holdout(d.y, n_classes, holdout, seed));
            fold_of.assign(d.n, 0);
            for (uint32_t i : splits[0].test) fold_of[i] = 1;
        } else {
            fold_of = ml::stratified_fold_ids(d.y, n_classes, folds, seed);
            splits = ml::stratified_kfold(d.y, n_classes, folds, seed);
        }
        if (!dump.empty()) {
            FILE* f = std::fopen(dump.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot open output file: " + dump);
            for (int32_t v : fold_of) std::fprintf(f, "%d\n", v);
            std::fclose(f);
        }

        ml::EvalResult base =
            ml::evaluate(d, splits, [] { return std::unique_ptr<ml::Classifier>(new ml::MajorityClassifier()); });
        const double peak_mb = ml::peak_host_memory_bytes() / (1024.0 * 1024.0);

        if (json) {
            std::printf("{\"name\":\"%s\",\"records\":%zu,\"class\":\"%s\",\"classes\":{", ml::json_escape(d.name).c_str(),
                        d.n, ml::json_escape(d.class_name).c_str());
            for (int c = 0; c < n_classes; ++c)
                std::printf("%s\"%s\":%zu", c ? "," : "", ml::json_escape(d.classes[c]).c_str(), class_counts[c]);
            std::printf("},\"attributes\":[");
            for (int a = 0; a < d.num_attrs(); ++a) {
                const ml::Attribute& at = d.attrs[a];
                ColumnStats s = column_stats(d, a);
                std::printf("%s{\"name\":\"%s\",\"type\":\"%s\",\"missing\":%zu,", a ? "," : "",
                            ml::json_escape(at.name).c_str(), at.numeric ? "num" : "cat", at.missing);
                if (at.numeric) {
                    std::printf("\"sum\":%.17g,\"min\":%.17g,\"max\":%.17g}", s.sum, s.min, s.max);
                } else {
                    std::printf("\"values\":{");
                    for (size_t v = 0; v < s.counts.size(); ++v)
                        std::printf("%s\"%s\":%zu", v ? "," : "", ml::json_escape(at.values[v]).c_str(), s.counts[v]);
                    std::printf("}}");
                }
            }
            std::printf("],\"split\":\"%s\",\"seed\":%llu,\"folds\":[", holdout > 0 ? "holdout" : "kfold", seed);
            for (size_t f = 0; f < base.folds.size(); ++f)
                std::printf("%s{\"train\":%zu,\"test\":%zu,\"majority_accuracy\":%.6f}", f ? "," : "",
                            base.folds[f].n_train, base.folds[f].n_test, base.folds[f].metrics.accuracy());
            std::printf("],\"majority_accuracy\":%.6f,\"majority_macro_f1\":%.6f,\"read_ms\":%.3f,\"peak_host_mb\":%.2f}\n",
                        base.pooled.accuracy(), base.pooled.macro_f1(), read_ms, peak_mb);
            return 0;
        }

        size_t missing = 0;
        for (const auto& a : d.attrs) missing += a.missing;
        std::printf("== %s\n", input.c_str());
        std::printf("records          : %zu\n", d.n);
        std::printf("attributes       : %d  (%d numeric, %d categorical)\n", d.num_attrs(), d.num_numeric(),
                    d.num_categorical());
        std::printf("missing values   : %zu\n", missing);
        std::printf("class            : %s  (%d classes)\n", d.class_name.c_str(), n_classes);
        for (int c = 0; c < n_classes; ++c)
            std::printf("    %-22s %9zu  (%5.1f%%)\n", d.classes[c].c_str(), class_counts[c],
                        100.0 * class_counts[c] / d.n);
        std::printf("attribute list\n");
        for (int a = 0; a < d.num_attrs(); ++a) {
            const ml::Attribute& at = d.attrs[a];
            ColumnStats s = column_stats(d, a);
            if (at.numeric) {
                std::printf("    %-28s num  range [%g, %g], mean %g", at.name.c_str(), s.min, s.max,
                            s.sum / static_cast<double>(d.n - at.missing));
            } else {
                std::printf("    %-28s cat  %d values", at.name.c_str(), at.num_values());
                if (at.num_values() <= 6) {
                    std::printf(" {");
                    for (int v = 0; v < at.num_values(); ++v) std::printf("%s%s", v ? ", " : "", at.values[v].c_str());
                    std::printf("}");
                }
            }
            if (at.missing) std::printf(", %zu missing", at.missing);
            std::printf("\n");
        }
        if (holdout > 0)
            std::printf("split            : stratified holdout, %.0f%% test, seed %llu\n", 100 * holdout, seed);
        else
            std::printf("split            : stratified %d-fold, seed %llu\n", folds, seed);
        for (size_t f = 0; f < base.folds.size(); ++f)
            std::printf("    fold %-2zu train %9zu  test %9zu\n", f, base.folds[f].n_train, base.folds[f].n_test);
        std::printf("majority baseline: accuracy %.4f (+/- %.4f over folds), macro-F1 %.4f\n", base.mean_accuracy,
                    base.std_accuracy, base.pooled.macro_f1());
        std::printf("time read        : %10.3f ms\n", read_ms);
        std::printf("peak host memory : %10.2f MB\n", peak_mb);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
