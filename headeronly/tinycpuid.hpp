#pragma once

#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <map>

namespace tiny {

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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static int get_total_cpus(void) {
	SYSTEM_INFO system_info;
	GetSystemInfo(&system_info);
	return system_info.dwNumberOfProcessors;
}

static GROUP_AFFINITY savedGroupAffinity;

static bool save_cpu_affinity(void) {
	HANDLE thread = GetCurrentThread();
	return GetThreadGroupAffinity(thread, &savedGroupAffinity);
}

static bool restore_cpu_affinity(void) {
	if (!savedGroupAffinity.Mask)
		return false;

	HANDLE thread = GetCurrentThread();
	return SetThreadGroupAffinity(thread, &savedGroupAffinity, NULL);
}

static bool set_cpu_affinity(int logical_cpu) {
    // Credits to https://github.com/PolygonTek/BlueshiftEngine/blob/fbc374cbc391e1147c744649f405a66a27c35d89/Source/Runtime/Private/Platform/Windows/PlatformWinThread.cpp#L27
	int groups = GetActiveProcessorGroupCount();
	int total_processors = 0;
	int group = 0;
	int number = 0;
	int found = 0;
	HANDLE thread = GetCurrentThread();
	GROUP_AFFINITY groupAffinity;

	for (int i = 0; i < groups; i++) {
		int processors = GetActiveProcessorCount(i);
		if (total_processors + processors > logical_cpu) {
			group = i;
			number = logical_cpu - total_processors;
			found = 1;
			break;
		}
		total_processors += processors;
	}
	if (!found) return 0; // logical CPU # too large, does not exist

	memset(&groupAffinity, 0, sizeof(groupAffinity));
	groupAffinity.Group = (WORD) group;
	groupAffinity.Mask = (KAFFINITY) (1ULL << number);
	return SetThreadGroupAffinity(thread, &groupAffinity, NULL);
}

#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sched.h>

static int get_total_cpus(void) {
	return sysconf(_SC_NPROCESSORS_ONLN);
}

static cpu_set_t saved_affinity;

static bool save_cpu_affinity(void) {
	return sched_getaffinity(0, sizeof(saved_affinity), &saved_affinity) == 0;
}

static bool restore_cpu_affinity(void) {
	return sched_setaffinity(0, sizeof(saved_affinity), &saved_affinity) == 0;
}

static bool set_cpu_affinity(int logical_cpu) {
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(logical_cpu, &cpuset);
	return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
}
#else
#error "Unsupported platform"
#endif

#define EXTRACTS_BITS(reg, highbit, lowbit) (((reg) >> (lowbit)) & ((1ULL << ((highbit) - (lowbit) + 1)) - 1))

bool HWMTSupported() {
    unsigned int e_x[4];
    cpuid(e_x, 1);
    return e_x[3] & (1 << 28);
}

// template <typename V, typename... T>
// constexpr std::array< V, sizeof...(T)> array_of(T&&... t) {return {{ std::forward<T>(t)... }};};

struct Core {
    unsigned int count_logical = 0;
    // No topology here, only the cache that the core has access to
    unsigned int L1i = 0; // instruction cache
    unsigned int L1d = 0; // data cache
    unsigned int L2 = 0; // unified cache
    unsigned int L3 = 0; // unified cache
    bool hyperthreading = 0;
    unsigned int count_physical() {return count_logical / (static_cast<unsigned int>(hyperthreading + 1));};
};

class CPUID {
    public:
        std::map<std::string, Core> cores_info;
        bool hybrid = false;
        std::string vendor;
        std::string full_name;

        CPUID();
        ~CPUID() = default;
        void print_info() noexcept;
        constexpr bool has(const std::string& feature) noexcept;
    private:
        unsigned int instr_set = 0u; // bitfield of supported instruction sets

        void get_cache_size(Core& core) noexcept;
        void get_vendor() noexcept;
        void get_full_name() noexcept;
        void get_cores() noexcept;
        void get_instr_set() noexcept;
};

// Helper struct to easily print formatted bytes
struct Bytes {
    unsigned int bytes;
    Bytes(unsigned int bytes) : bytes(bytes) {};
    friend std::ostream &operator<<(std::ostream &os, Bytes &&value) {
        const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
        int i = 0;
        for (; value.bytes >= 1024 && i < 5; i++) value.bytes /= 1024;
        return (os << value.bytes << suffixes[i]);
    };
};

static auto InstrSets = std::map<std::string, unsigned int>({
    {"MMX", 1u << 0},
    {"SSE", 1u << 1},
    {"SSE2", 1u << 2},
    {"SSE3", 1u << 3},
    {"SSSE3", 1u << 4},
    {"SSE4.1", 1u << 5},
    {"SSE4.2", 1u << 6},
    {"POPCNT", 1u << 7},
    {"AVX", 1u << 8},
    {"AVX2", 1u << 9},
    {"FMA3", 1u << 10},
    {"FMA4", 1u << 11},
    {"AVX512F", 1u << 12},
    {"AVX512DQ", 1u << 13},
    {"AVX512IFMA", 1u << 14},
    {"AVX512CD", 1u << 15},
    {"AVX512BW", 1u << 16},
    {"AVX512VL", 1u << 17},
    {"AVX512_VPOPCNTDQ", 1u << 18},
    {"AVX512_VBMI", 1u << 19},
    {"AVX512_VBMI2", 1u << 20},
    {"AVX512_VNNI", 1u << 21},
    {"AVX512_BITALG", 1u << 22},
    {"AVX512_FP16", 1u << 23},
    {"AVX512_BF16", 1u << 24},
    {"AMX_BF16", 1u << 25},
    {"AMX_TILE", 1u << 26},
    {"AMX_INT8", 1u << 27},
    {"BMI1", 1u << 28},
    {"BMI2", 1u << 29}
});

CPUID::CPUID() {
    get_vendor();
    get_full_name();
    get_instr_set();
    get_cores();
}

void CPUID::get_cores() noexcept {
    save_cpu_affinity();
    unsigned int e_x[4];
    unsigned int threads = static_cast<unsigned int>(get_total_cpus());
    e_x[0] = 0x7;
    cpuid(e_x, e_x[0]);
    if ((e_x[3] & (1 << 15))) {
        hybrid = true;
        for (unsigned int i = 0; i < threads; i++) {
            if (set_cpu_affinity(i)) {
                e_x[0] = 0x1a;
                cpuid(e_x, e_x[0]);
                switch (EXTRACTS_BITS(e_x[0], 31, 24)) {
                    case 0x20: // e-core
                        if (cores_info.find("e-core") == cores_info.end()) {
                            cores_info["e-core"]; // initialize
                            get_cache_size(cores_info["e-core"]);
                        }
                        cores_info["e-core"].count_logical++;
                        break;
                    case 0x40: //p-core
                        if (cores_info.find("p-core") == cores_info.end()) {
                            cores_info["p-core"]; // initialize
                            cores_info["p-core"].hyperthreading = true; // hardcoded because getting real value is a pain
                            get_cache_size(cores_info["p-core"]);
                        }
                        cores_info["p-core"].count_logical++;
                        break;
                }
            }
        }
    } else {
        cores_info["general"];
        cores_info["general"].count_logical = threads;
        cores_info["general"].hyperthreading = HWMTSupported();
        get_cache_size(cores_info["general"]);
    }
    restore_cpu_affinity();
}

void CPUID::get_cache_size(Core& core) noexcept {
    unsigned int e_x[4];
    // Intel CPU
    // From Intel software reference manual (p846 March 2023)
    // INPUT EAX = 04H: Returns Deterministic Cache Parameters for Each Level
    for (unsigned int i = 0;; i++) {
        
        e_x[0] = 4;
        e_x[1] = 0;
        e_x[2] = i;
        e_x[3] = 0;
        cpuid(e_x, e_x[0], e_x[2]);

        // Check cache type (EAX[4:0])
        // 0 = Null, no more caches
        // 1 = Data Cache
        // 2 = Instruction Cache
        // 3 = Unified Cache
        // 4-31 = Reserved
        if ((e_x[0] & 0x1F) == 0) {
            break;
        }

        // Cache size = (Ways + 1) * (Partitions + 1) * (Line Size + 1) * (Sets + 1)
        // Cache size = (EBX[31:22] + 1) * (EBX[21:12] + 1) * (EBX[11:0] + 1) * (ECX + 1)
        uint32_t cache_size = (((e_x[1] >> 22) & 0x3ff) + 1) * (((e_x[1] >> 12) & 0x3ff) + 1) * ((e_x[1] & 0xfff) + 1) * (e_x[2] + 1);

        uint32_t cache_type = (e_x[0] & 0x1F); // 1 = Data Cache, 2 = Instruction Cache, 3 = Unified Cache
        uint32_t cache_level = (e_x[0] >> 5) & 0x7; // 1 = L1, 2 = L2, 3 = L3

        if (cache_type == 2 && cache_level == 1) {
            core.L1i = cache_size;
        } else if (cache_type == 1 && cache_level == 1) {
            core.L1d = cache_size;
        } else if (cache_type == 3 && cache_level == 2) {
            core.L2 = cache_size;
        } else if (cache_type == 3 && cache_level == 3) {
            core.L3 = cache_size;
        }
    }
}

void CPUID::get_instr_set() noexcept {
    unsigned int cpu_info[4] = {0, 0, 0, 0};
    unsigned int eax, ebx, ecx, edx;

    // CPUID with EAX = 1 returns feature information in EDX and ECX
    cpuid(cpu_info, 1);

    edx = cpu_info[3];
    ecx = cpu_info[2];

    if (edx & (1 << 23)) instr_set |= InstrSets["MMX"];
    if (edx & (1 << 25)) instr_set |= InstrSets["SSE"];
    if (edx & (1 << 26)) instr_set |= InstrSets["SSE2"];
    if (ecx & (1 << 0)) instr_set |= InstrSets["SSE3"];
    if (ecx & (1 << 9)) instr_set |= InstrSets["SSSE3"];
    if (ecx & (1 << 19)) instr_set |= InstrSets["SSE4.1"];
    if (ecx & (1 << 20)) instr_set |= InstrSets["SSE4.2"];
    if (ecx & (1 << 23)) instr_set |= InstrSets["POPCNT"];
    if (ecx & (1 << 28)) instr_set |= InstrSets["AVX"];
    if (ecx & (1 << 12)) instr_set |= InstrSets["FMA3"];

    // CPUID with EAX = 7 and ECX = 0 returns extended feature information in EBX, ECX, and EDX
    cpuid(cpu_info, 7, 0);

    ebx = cpu_info[1];
    ecx = cpu_info[2];
    edx = cpu_info[3];
    if (ebx & (1 << 3)) instr_set |= InstrSets["BMI1"];
    if (ebx & (1 << 5)) instr_set |= InstrSets["AVX2"];
    if (ebx & (1 << 8)) instr_set |= InstrSets["BMI2"];
    if (ebx & (1 << 16)) instr_set |= InstrSets["AVX512F"];
    if (ebx & (1 << 17)) instr_set |= InstrSets["AVX512DQ"];
    if (ebx & (1 << 21)) instr_set |= InstrSets["AVX512IFMA"];
    if (ebx & (1 << 28)) instr_set |= InstrSets["AVX512CD"];
    if (ebx & (1 << 30)) instr_set |= InstrSets["AVX512BW"];
    if (ebx & (1 << 31)) instr_set |= InstrSets["AVX512VL"];
    if (ecx & (1 << 14)) instr_set |= InstrSets["AVX512_VPOPCNTDQ"];
    if (ecx & (1 << 1)) instr_set |= InstrSets["AVX512_VBMI"];
    if (ecx & (1 << 6)) instr_set |= InstrSets["AVX512_VBMI2"];
    if (ecx & (1 << 11)) instr_set |= InstrSets["AVX512_VNNI"];
    if (ecx & (1 << 12)) instr_set |= InstrSets["AVX512_BITALG"];
    if (edx & (1 << 23)) instr_set |= InstrSets["AVX512_FP16"];
    if (edx & (1 << 22)) instr_set |= InstrSets["AMX_BF16"];
    if (edx & (1 << 24)) instr_set |= InstrSets["AMX_TILE"];
    if (edx & (1 << 25)) instr_set |= InstrSets["AMX_INT8"];

    cpuid(cpu_info, 7, 1);

    eax = cpu_info[0];

    if (eax & (1 << 5)) instr_set |= InstrSets["AVX512_BF16"];
}

constexpr bool CPUID::has(const std::string& feature) noexcept {
    return (instr_set & InstrSets[feature]);
}

void CPUID::get_vendor() noexcept {
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

void CPUID::get_full_name() noexcept {
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

void CPUID::print_info() noexcept {
    std::cout << "----- CPU Device information -----\n";
    std::cout << "Vendor: " << vendor << "\n";
    std::cout << "Full name: " << full_name << "\n";
    std::cout << "Cores: \n";
    uint32_t cores = 0;
    uint32_t threads = 0;
    for (auto& [key, value] : cores_info) {
        std::cout << "\t" << key << ":\n";
        std::cout << "\t\tPhysical: " << value.count_physical() << "\n";
        std::cout << "\t\tLogical: " << value.count_logical << "\n";
        std::cout << "\t\tL1i: " << Bytes(value.L1i) << "\n";
        std::cout << "\t\tL1d: " << Bytes(value.L1d) << "\n";
        std::cout << "\t\tL2: " << Bytes(value.L2) << "\n";
        std::cout << "\t\tL3: " << Bytes(value.L3) << "\n";
        cores += value.count_physical();
        threads += value.count_logical;
    }
    std::cout << "Physical cores: " << cores << "\n";
    std::cout << "Logical cores: " << threads << "\n";

    std::cout << "Instruction sets: ";
    for (auto& [key, value] : InstrSets) {
        if (instr_set & value) {
            std::cout << key << " ";
        }
    }
    std::cout << "\n-----------------------------------\n";
}

} // tiny namespace

// int main(int argc, char** argv) {
//     tiny::CPUID cpuid;
//     cpu.print_info();
//     std::cout << cpu.has("AVX2") << std::endl;
//     return 0;
// }