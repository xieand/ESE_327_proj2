// Small shared utilities: wall-clock timer, peak memory, a portable RNG.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml {

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }
    void reset() { start_ = std::chrono::steady_clock::now(); }

private:
    std::chrono::steady_clock::time_point start_;
};

// Peak resident memory of this process (peak working set on Windows, max RSS elsewhere).
size_t peak_host_memory_bytes();

// SplitMix64. The standard library's distributions and std::shuffle are not
// specified bit-for-bit, so splits would differ between compilers; this
// generator gives the same folds on every platform for a given seed.
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}
    uint64_t next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform integer in [0, n), n > 0, without modulo bias.
    uint64_t below(uint64_t n) {
        uint64_t limit = UINT64_MAX - UINT64_MAX % n;
        uint64_t r;
        do r = next();
        while (r >= limit);
        return r % n;
    }
    // Uniform double in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    template <typename T>
    void shuffle(std::vector<T>& v) {
        for (size_t i = v.size(); i > 1; --i) {
            size_t j = static_cast<size_t>(below(i));
            T tmp = v[i - 1];
            v[i - 1] = v[j];
            v[j] = tmp;
        }
    }

private:
    uint64_t state_;
};

std::string json_escape(const std::string& s);

}  // namespace ml
