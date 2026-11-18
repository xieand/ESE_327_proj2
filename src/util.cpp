#include "util.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace ml {

size_t peak_host_memory_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.PeakWorkingSetSize;
    return 0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<size_t>(ru.ru_maxrss);
#else
    return static_cast<size_t>(ru.ru_maxrss) * 1024;
#endif
#endif
}

std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') r.push_back('\\');
        r.push_back(c);
    }
    return r;
}

}  // namespace ml
