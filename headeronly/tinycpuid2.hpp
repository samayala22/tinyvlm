#pragma once

#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
static inline void cpuid(unsigned int info[4], unsigned int eax, unsigned int ecx) {
    __cpuidex(reinterpret_cast<int*>(info), static_cast<int>(eax), static_cast<int>(ecx));
}
static inline void cpuid(unsigned int info[4], unsigned int eax) {
    __cpuid(reinterpret_cast<int*>(info), static_cast<int>(eax));
}
#else // GCC, Clang, ICC
#include <cpuid.h>
static inline void cpuid(unsigned int info[4], unsigned int eax, unsigned int ecx) {
    __cpuid_count(eax, ecx, info[0], info[1], info[2], info[3]);
}
static inline void cpuid(unsigned int info[4], unsigned int eax) {
    __cpuid(eax, info[0], info[1], info[2], info[3]);
}
#endif // _MSC_VER

namespace tiny {

class CPUID2 {
    public:
        std::string vendor;
        std::string full_name;

        CPUID2();
        ~CPUID2() = default;
    private:
        void get_vendor() noexcept;
        void get_full_name() noexcept;
};

CPUID2::CPUID2() {
    get_vendor();
    get_full_name();
}

inline void CPUID2::get_vendor() noexcept {
    unsigned int cpu_info[4] = {0, 0, 0, 0}; // registers
    // Call to cpuid with eax = 0
    cpuid(cpu_info, 0);

    // The vendor string is composed as EBX + EDX + ECX.
    // Each reg is 4 bytes so the string is 3 x 4 + 1 (null terminator) = 13 bytes
    vendor.clear();
    vendor.reserve(12);
    vendor.append(reinterpret_cast<const char*>(&cpu_info[1]), 4);
    vendor.append(reinterpret_cast<const char*>(&cpu_info[3]), 4);
    vendor.append(reinterpret_cast<const char*>(&cpu_info[2]), 4);
}

inline void CPUID2::get_full_name() noexcept {
    unsigned int cpu_info[4] = {0, 0, 0, 0}; // registers
    full_name.clear();
    full_name.reserve(48); // 3 * 16 bytes (cpu_info)

    // Use Extended Function CPUID
    unsigned int eax_leafs[3] = {0x8000'0002, 0x8000'0003, 0x8000'0004};
    for (unsigned int i = 0; i < 3; i++) {
        cpuid(cpu_info, eax_leafs[i]);
        full_name.append(reinterpret_cast<const char*>(&cpu_info[0]), 16);
    }
}

} // tiny namespace
