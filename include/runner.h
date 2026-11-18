// Command line and reporting shared by every classifier program: the common
// options (split, seed, output), the evaluation run, and the run report.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dataset.h"
#include "eval.h"

namespace ml {

// Minimal option parser: options are taken out one by one, whatever is left
// at the end is positional (or an error).
class Args {
public:
    Args(int argc, char** argv);
    bool flag(const std::string& name);
    bool value(const std::string& name, std::string& out);
    bool value(const std::string& name, int& out);
    bool value(const std::string& name, uint64_t& out);
    bool value(const std::string& name, double& out);
    const std::vector<std::string>& rest() const { return tokens_; }

private:
    std::vector<std::string> tokens_;
};

struct RunOptions {
    std::string input;
    int folds = 10;              // 0: train and test on the whole table (resubstitution)
    double holdout = 0;          // > 0: one stratified split with this test fraction
    uint64_t seed = 327;
    bool json = false;
    bool quiet = false;
    std::string predictions;     // file: predicted label of every record
    std::string dump_folds;      // file: fold id of every record
};

// Takes the common options out of `args`.
void parse_run_options(Args& args, RunOptions& opt);
// Help text for the common options.
const char* run_options_help();

struct RunInfo {
    std::string algorithm;       // e.g. "dtree"
    std::string device;          // e.g. "cpu"
    std::string options;         // classifier options, for the report
    double read_ms = 0;
    size_t peak_device_bytes = 0;
};

// Builds the split named by the options, evaluates the classifier on it,
// writes the requested files and prints the report (text or one JSON line).
// Reported train / predict times are means per fold.
EvalResult run_evaluation(const Dataset& data, const RunOptions& opt, const RunInfo& info,
                          const ClassifierFactory& make);

}  // namespace ml
