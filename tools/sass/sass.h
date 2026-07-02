#pragma once
#include <cstdint>
#include <cstdio>
#include <climits>
#include <string>
#include <vector>

namespace sass {

// ═══════════════════════════════════════════════════════════════════════════
// Control word fields
// ═══════════════════════════════════════════════════════════════════════════

struct CtrlFields {
    int stall;      // [3:0]   stall count (0-15)
    int yield;      // [4]     yield hint
    int wr_bar;     // [9:5]   write barrier index (0-5, 31=none)
    int rd_bar;     // [14:10] read barrier index (0-5, 31=none)
    int wait_mask;  // [20:15] wait barrier mask (6 bits)
    int reuse;      // [22:21] register reuse flags
};

CtrlFields decode_control(uint64_t ctrl);

// ═══════════════════════════════════════════════════════════════════════════
// Instruction
// ═══════════════════════════════════════════════════════════════════════════

struct Instruction {
    uint32_t addr;
    std::string opcode;
    std::string operands;
    std::string predicate;
    uint64_t encoding;
    uint64_t control;
    CtrlFields ctrl;
    std::vector<std::string> regs_read;
    std::vector<std::string> regs_written;
    const char* category;
    int latency;

    int stall() const { return ctrl.stall; }
};

// ═══════════════════════════════════════════════════════════════════════════
// Dependency graph
// ═══════════════════════════════════════════════════════════════════════════

struct DepEdge {
    int src, dst;
    std::string reg;
    enum Type { RAW, WAR, WAW } type;
    int latency;
};

// ═══════════════════════════════════════════════════════════════════════════
// Slack analysis
// ═══════════════════════════════════════════════════════════════════════════

struct SlackInfo {
    int idx;
    uint32_t addr;
    const char* opcode;      // points into Instruction.opcode
    const char* category;    // points into Instruction.category
    int actual_stall;
    int min_stall;
    int slack;
    enum SlackType { NONE, SCHEDULING, BARRIER_WAIT } type;
    int wait_mask;
    int limiting_dep_idx;       // -1 = none
    const char* limiting_dep_op; // null = none
};

// ═══════════════════════════════════════════════════════════════════════════
// Kernel container
// ═══════════════════════════════════════════════════════════════════════════

struct Kernel {
    std::string name;
    std::vector<Instruction> instrs;
};

// ═══════════════════════════════════════════════════════════════════════════
// Tables
// ═══════════════════════════════════════════════════════════════════════════

int get_latency(const char* opcode);
const char* categorize(const char* opcode);

// ═══════════════════════════════════════════════════════════════════════════
// Parse
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Kernel> parse_sass(const char* text, size_t len,
                                const char* kernel_filter = nullptr);

// ═══════════════════════════════════════════════════════════════════════════
// Analysis
// ═══════════════════════════════════════════════════════════════════════════

std::vector<DepEdge> build_dep_graph(const std::vector<Instruction>& instrs,
                                      bool raw_only = true);

std::vector<SlackInfo> compute_slack(const std::vector<Instruction>& instrs,
                                      const std::vector<DepEdge>& edges);

int critical_path(const std::vector<Instruction>& instrs,
                  const std::vector<DepEdge>& edges,
                  std::vector<int>& path);

// ═══════════════════════════════════════════════════════════════════════════
// Output
// ═══════════════════════════════════════════════════════════════════════════

void print_listing(FILE* out, const std::vector<Instruction>& instrs,
                   const std::vector<SlackInfo>* slack = nullptr,
                   uint32_t sec_lo = 0, uint32_t sec_hi = UINT32_MAX);

void print_summary(FILE* out, const std::vector<Instruction>& instrs,
                   const std::vector<SlackInfo>& slack,
                   uint32_t sec_lo = 0, uint32_t sec_hi = UINT32_MAX);

void write_json(FILE* out, const char* kernel_name,
                const std::vector<Instruction>& instrs,
                const std::vector<SlackInfo>& slack,
                int cp_cycles, const std::vector<int>& cp_path,
                uint32_t sec_lo = 0, uint32_t sec_hi = UINT32_MAX);

// ═══════════════════════════════════════════════════════════════════════════
// Calibration
// ═══════════════════════════════════════════════════════════════════════════

// Print calibration CUDA source to stdout and exit
void print_calibration(FILE* out);

// Compare SASS stall counts vs runtime measurements (runtime_text=nullptr for SASS-only)
void calibrate_compare(FILE* out,
                       const std::vector<Kernel>& kernels,
                       const char* runtime_text);

} // namespace sass
