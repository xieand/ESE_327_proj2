#include "runner.h"

#include <cstdio>
#include <numeric>
#include <stdexcept>

#include "util.h"

namespace ml {

Args::Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) tokens_.emplace_back(argv[i]);
}

bool Args::flag(const std::string& name) {
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (tokens_[i] == name) {
            tokens_.erase(tokens_.begin() + i);
            return true;
        }
    }
    return false;
}

bool Args::value(const std::string& name, std::string& out) {
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (tokens_[i] != name) continue;
        if (i + 1 >= tokens_.size()) throw std::invalid_argument(name + " needs a value");
        out = tokens_[i + 1];
        tokens_.erase(tokens_.begin() + i, tokens_.begin() + i + 2);
        return true;
    }
    return false;
}

namespace {

template <typename T, typename F>
bool parse_number(Args& args, const std::string& name, T& out, F convert) {
    std::string s;
    if (!args.value(name, s)) return false;
    try {
        size_t used = 0;
        out = static_cast<T>(convert(s, &used));
        if (used != s.size()) throw std::invalid_argument(s);
    } catch (const std::logic_error&) {
        throw std::invalid_argument("bad value '" + s + "' for " + name);
    }
    return true;
}

}  // namespace

bool Args::value(const std::string& name, int& out) {
    return parse_number(*this, name, out, [](const std::string& s, size_t* n) { return std::stoi(s, n); });
}
bool Args::value(const std::string& name, uint64_t& out) {
    return parse_number(*this, name, out, [](const std::string& s, size_t* n) { return std::stoull(s, n); });
}
bool Args::value(const std::string& name, double& out) {
    return parse_number(*this, name, out, [](const std::string& s, size_t* n) { return std::stod(s, n); });
}

void parse_run_options(Args& args, RunOptions& opt) {
    args.value("--folds", opt.folds);
    args.value("--holdout", opt.holdout);
    args.value("--seed", opt.seed);
    args.value("--predictions", opt.predictions);
    args.value("--dump-folds", opt.dump_folds);
    opt.json = args.flag("--json");
    opt.quiet = args.flag("--quiet");
    if (opt.folds < 0 || opt.folds == 1) throw std::invalid_argument("--folds must be 0 or at least 2");
    if (opt.holdout < 0 || opt.holdout >= 1) throw std::invalid_argument("--holdout must be in (0, 1)");
}

const char* run_options_help() {
    return "  --folds K            stratified K-fold cross-validation (default 10);\n"
           "                       0 = train and test on the whole table\n"
           "  --holdout F          one stratified split with fraction F as the test set\n"
           "  --seed S             seed of the split (default 327)\n"
           "  --predictions file   write the predicted label of every tested record\n"
           "  --dump-folds file    write the fold id of every record\n"
           "  --json               print the run report as one JSON line\n"
           "  --quiet              suppress the text report\n";
}

EvalResult run_evaluation(const Dataset& data, const RunOptions& opt, const RunInfo& info,
                          const ClassifierFactory& make) {
    const int n_classes = data.num_classes();
    std::vector<Split> splits;
    std::vector<int32_t> fold_of(data.num_rows(), 0);
    std::string split_name;
    if (opt.holdout > 0) {
        splits.push_back(stratified_holdout(data.y, n_classes, opt.holdout, opt.seed));
        for (uint32_t i : splits[0].test) fold_of[i] = 1;
        split_name = "holdout";
    } else if (opt.folds == 0) {
        Split s;
        s.train.resize(data.num_rows());
        std::iota(s.train.begin(), s.train.end(), 0u);
        s.test = s.train;
        splits.push_back(std::move(s));
        split_name = "resubstitution";
    } else {
        fold_of = stratified_fold_ids(data.y, n_classes, opt.folds, opt.seed);
        splits = stratified_kfold(data.y, n_classes, opt.folds, opt.seed);
        split_name = "kfold";
    }

    EvalResult r = evaluate(data, splits, make);

    if (!opt.dump_folds.empty()) {
        FILE* f = std::fopen(opt.dump_folds.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output file: " + opt.dump_folds);
        for (int32_t v : fold_of) std::fprintf(f, "%d\n", v);
        std::fclose(f);
    }
    if (!opt.predictions.empty()) {
        FILE* f = std::fopen(opt.predictions.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output file: " + opt.predictions);
        for (int32_t p : r.predictions) std::fprintf(f, "%s\n", p < 0 ? "-" : data.classes[p].c_str());
        std::fclose(f);
    }

    const size_t k = r.folds.size();
    const double train_ms = k ? r.train_ms / k : 0, predict_ms = k ? r.predict_ms / k : 0;
    const double mb = 1024.0 * 1024.0;
    const double host_mb = peak_host_memory_bytes() / mb, device_mb = info.peak_device_bytes / mb;

    if (opt.json) {
        std::printf("{\"algorithm\":\"%s\",\"device\":\"%s\",\"options\":\"%s\",\"dataset\":\"%s\",\"records\":%zu,"
                    "\"attributes\":%d,\"classes\":%d,\"split\":\"%s\",\"folds\":%zu,\"seed\":%llu,"
                    "\"accuracy\":%.6f,\"accuracy_std\":%.6f,\"pooled_accuracy\":%.6f,\"macro_f1\":%.6f,"
                    "\"train_ms\":%.3f,\"predict_ms\":%.3f,\"read_ms\":%.3f,\"peak_host_mb\":%.2f,"
                    "\"peak_device_mb\":%.2f",
                    json_escape(info.algorithm).c_str(), json_escape(info.device).c_str(),
                    json_escape(info.options).c_str(), json_escape(data.name).c_str(), data.num_rows(),
                    data.num_attrs(), n_classes, split_name.c_str(), k, static_cast<unsigned long long>(opt.seed),
                    r.mean_accuracy, r.std_accuracy, r.pooled.accuracy(), r.pooled.macro_f1(), train_ms, predict_ms,
                    info.read_ms, host_mb, device_mb);
        for (const auto& kv : r.info) std::printf(",\"%s\":%.6g", json_escape(kv.first).c_str(), kv.second);
        std::printf("}\n");
    } else if (!opt.quiet) {
        std::printf("== %s (%s) on %s\n", info.algorithm.c_str(), info.device.c_str(), opt.input.c_str());
        if (!info.options.empty()) std::printf("options          : %s\n", info.options.c_str());
        std::printf("records          : %zu  (%d numeric + %d categorical attributes, %d classes)\n", data.num_rows(),
                    data.num_numeric(), data.num_categorical(), n_classes);
        if (opt.holdout > 0)
            std::printf("evaluation       : stratified holdout, %.0f%% test, seed %llu\n", 100 * opt.holdout,
                        static_cast<unsigned long long>(opt.seed));
        else if (opt.folds == 0)
            std::printf("evaluation       : trained and tested on the whole table\n");
        else
            std::printf("evaluation       : stratified %d-fold cross-validation, seed %llu\n", opt.folds,
                        static_cast<unsigned long long>(opt.seed));
        std::printf("accuracy         : %.4f", r.mean_accuracy);
        if (k > 1) std::printf("  (+/- %.4f over folds)", r.std_accuracy);
        std::printf("\nmacro-F1         : %.4f\n", r.pooled.macro_f1());
        for (const auto& kv : r.info) std::printf("%-17s: %.6g\n", kv.first.c_str(), kv.second);
        std::printf("time read        : %10.3f ms\n", info.read_ms);
        std::printf("time train       : %10.3f ms  (mean per fold)\n", train_ms);
        std::printf("time predict     : %10.3f ms  (mean per fold)\n", predict_ms);
        std::printf("peak host memory : %10.2f MB\n", host_mb);
        if (info.peak_device_bytes) std::printf("peak device mem  : %10.2f MB\n", device_mb);
    }
    return r;
}

}  // namespace ml
