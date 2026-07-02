#include "sass.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace sass {

// ═══════════════════════════════════════════════════════════════════════════
// Control word decoder
// ═══════════════════════════════════════════════════════════════════════════

CtrlFields decode_control(uint64_t ctrl) {
    return {
        .stall     = (int)(ctrl & 0xf),
        .yield     = (int)((ctrl >> 4) & 1),
        .wr_bar    = (int)((ctrl >> 5) & 0x1f),
        .rd_bar    = (int)((ctrl >> 10) & 0x1f),
        .wait_mask = (int)((ctrl >> 15) & 0x3f),
        .reuse     = (int)((ctrl >> 21) & 3),
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Latency table — ONLY measured values from B200 calibration.cu K1-K12
// Uncalibrated opcodes return -1 (no Ampere guesses, no assumptions)
// ═══════════════════════════════════════════════════════════════════════════

static const std::unordered_map<std::string, int>& latency_table() {
    static const std::unordered_map<std::string, int> t = {
        // ── Calibrated on B200 (calibration.cu, commit 0338ac5) ──
        {"IADD3", 2},   // K7b: 2.08 c/i dep chain
        {"PRMT", 0},    // K8: ~0 c/i (UPRMT pipe, free)
        {"HFMA2", 4},   // K4: 4.00 c/i dep chain
        {"HADD2", 5},   // K11: 5.52 c/i dep chain
        {"F2FP", 4},    // K2: 4.10 c/i dep chain
        {"STS", 32},    // K6: 31.99 c/i throughput
        {"LDG", 39},    // K12: 39.09 c/i pointer chase (includes addr calc)
        // ── Barrier-synchronized (latency=0 is correct by design) ──
        {"LDTM", 0}, {"STTM", 0},
        {"UTCBAR", 0},
        {"BAR", 0}, {"DEPBAR", 0}, {"BSYNC", 0}, {"WARPSYNC", 0},
        {"NANOSLEEP", 0}, {"MEMBAR", 0}, {"ERRBAR", 0}, {"CGAERRBAR", 0},
        // ── Control flow (latency=0 is correct by design) ──
        {"BRA", 0}, {"EXIT", 0}, {"RET", 0}, {"BREAK", 0}, {"CALL", 0},
        // ── MMA (barrier-synchronized) ──
        {"TCGEN05", 0},
    };
    return t;
}

// Extract base opcode (up to first '.')
static void extract_base(const char* opcode, char* buf, size_t bufsz) {
    size_t i = 0;
    while (i < bufsz - 1 && opcode[i] && opcode[i] != '.')
        buf[i] = opcode[i], i++;
    buf[i] = '\0';
}

int get_latency(const char* opcode) {
    char base[64];
    extract_base(opcode, base, sizeof(base));
    auto& t = latency_table();
    auto it = t.find(base);
    return it != t.end() ? it->second : -1; // -1 = UNCALIBRATED
}

// ═══════════════════════════════════════════════════════════════════════════
// Categories
// ═══════════════════════════════════════════════════════════════════════════

struct CatEntry {
    const char* name;
    const char* opcodes[16]; // null-terminated list
};

static const CatEntry CAT_TABLE[] = {
    {"tmem_load",  {"LDTM", nullptr}},
    {"tmem_store", {"STTM", nullptr}},
    {"cvt",        {"F2FP", "I2FP", "F2IP", "I2IP", nullptr}},
    {"sts",        {"STS", nullptr}},
    {"lds",        {"LDS", nullptr}},
    {"alu_int",    {"IMAD", "IADD3", "LEA", "LOP3", "SHF", "PRMT", "MOV", "SEL", "IMNMX", nullptr}},
    {"alu_fp",     {"FADD", "FMUL", "FFMA", "MUFU", nullptr}},
    {"cmp",        {"ISETP", "ISET", "FSETP", nullptr}},
    {"mma",        {"TCGEN05", nullptr}},
    {"tma",        {"UTCBAR", nullptr}},
    {"barrier",    {"BAR", "DEPBAR", "BSYNC", "WARPSYNC", "NANOSLEEP", "MEMBAR", "ERRBAR", "CGAERRBAR", nullptr}},
    {"const",      {"LDC", "LDCU", nullptr}},
    {"special",    {"S2R", "S2UR", "CS2R", "R2UR", nullptr}},
    {"control",    {"BRA", "EXIT", "RET", "BREAK", "CALL", nullptr}},
    {nullptr, {nullptr}},
};

const char* categorize(const char* opcode) {
    char base[64];
    extract_base(opcode, base, sizeof(base));
    for (const CatEntry* e = CAT_TABLE; e->name; e++) {
        for (const char* const* op = e->opcodes; *op; op++) {
            if (strcmp(base, *op) == 0)
                return e->name;
        }
    }
    return "other";
}

// ═══════════════════════════════════════════════════════════════════════════
// Register parsing helpers
// ═══════════════════════════════════════════════════════════════════════════

// Check if character is alphanumeric (not counting '_')
static inline bool is_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

// Scan for R<digits> tokens, avoiding SR_TID etc (preceding char must not be alphanumeric)
static void extract_regs(const char* s, std::unordered_set<std::string>& out) {
    for (const char* p = s; *p; ) {
        if (*p == 'R' && (p == s || !is_alnum(p[-1]))) {
            const char* d = p + 1;
            if (*d >= '0' && *d <= '9') {
                while (*d >= '0' && *d <= '9') d++;
                // Exclude RZ (no digits won't happen here) and check not followed by alnum
                if (!is_alnum(*d)) {
                    char buf[16];
                    int n = (int)(d - p);
                    if (n < (int)sizeof(buf)) {
                        memcpy(buf, p, n);
                        buf[n] = '\0';
                        out.insert(buf);
                    }
                }
                p = d;
                continue;
            }
        }
        p++;
    }
}

// Scan for UR<digits> tokens
static void extract_uregs(const char* s, std::unordered_set<std::string>& out) {
    for (const char* p = s; *p; ) {
        if (p[0] == 'U' && p[1] == 'R' && (p == s || !is_alnum(p[-1]))) {
            const char* d = p + 2;
            if (*d >= '0' && *d <= '9') {
                while (*d >= '0' && *d <= '9') d++;
                if (!is_alnum(*d)) {
                    char buf[16];
                    int n = (int)(d - p);
                    if (n < (int)sizeof(buf)) {
                        memcpy(buf, p, n);
                        buf[n] = '\0';
                        out.insert(buf);
                    }
                }
                p = d;
                continue;
            }
        }
        p++;
    }
}

// Extract first R<num> register number, return -1 if none
static int extract_first_reg_num(const char* s) {
    for (const char* p = s; *p; ) {
        if (*p == 'R' && (p == s || !is_alnum(p[-1]))) {
            const char* d = p + 1;
            if (*d >= '0' && *d <= '9') {
                int val = 0;
                while (*d >= '0' && *d <= '9')
                    val = val * 10 + (*d++ - '0');
                if (!is_alnum(*d))
                    return val;
            }
        }
        p++;
    }
    return -1;
}

// Split operands at commas into spans
struct Span { const char* s; int len; };

static int split_comma(const char* ops, Span* parts, int max_parts) {
    int n = 0;
    const char* p = ops;
    while (*p && n < max_parts) {
        while (*p == ' ' || *p == '\t') p++;
        const char* start = p;
        // Handle bracket nesting for [Rn+offset] style operands
        int depth = 0;
        while (*p) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == ',' && depth == 0) break;
            p++;
        }
        // Trim trailing whitespace
        const char* end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) end--;
        parts[n].s = start;
        parts[n].len = (int)(end - start);
        n++;
        if (*p == ',') p++;
    }
    return n;
}

// Helper: make a string from Span
static std::string span_str(const Span& sp) {
    return std::string(sp.s, sp.len);
}

// Extract predicate guard: "@[!]P<num>" → "P<num>" into reads
static void parse_pred_guard(const char* pred, std::unordered_set<std::string>& reads) {
    if (!pred || pred[0] != '@') return;
    const char* p = pred + 1;
    if (*p == '!') p++;
    if (*p == 'P') {
        const char* d = p + 1;
        if (*d >= '0' && *d <= '9') {
            char buf[8];
            const char* start = p;
            while (*d >= '0' && *d <= '9') d++;
            int n = (int)(d - start);
            if (n < (int)sizeof(buf)) {
                memcpy(buf, start, n);
                buf[n] = '\0';
                reads.insert(buf);
            }
        }
    }
}

static void parse_reg_sets(const char* opcode, const char* operands, const char* predicate,
                           std::vector<std::string>& reads_out, std::vector<std::string>& writes_out) {
    std::unordered_set<std::string> reads, writes;
    char base[64];
    extract_base(opcode, base, sizeof(base));

    Span parts[16];
    int nparts = split_comma(operands, parts, 16);

    // Predicate guard — always a read
    parse_pred_guard(predicate, reads);

    // --- LDTM: writes N consecutive regs from destination ---
    if (strcmp(base, "LDTM") == 0) {
        int width = 1;
        if (strstr(opcode, "x64")) width = 64;
        else if (strstr(opcode, "x32")) width = 32;
        else if (strstr(opcode, "x16")) width = 16;
        else if (strstr(opcode, "x8")) width = 8;

        if (nparts > 0) {
            std::string p0 = span_str(parts[0]);
            int br = extract_first_reg_num(p0.c_str());
            if (br >= 0) {
                for (int i = 0; i < width; i++) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "R%d", br + i);
                    writes.insert(buf);
                }
            }
        }
        for (int i = 1; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            extract_uregs(pi.c_str(), reads);
        }
        goto done;
    }

    // --- STS: store to shared, reads data + addr regs, writes nothing ---
    if (strcmp(base, "STS") == 0) {
        for (int i = 0; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
        // Handle .128 = 4 regs, .64 = 2 regs from data operand
        if (nparts >= 2) {
            std::string last = span_str(parts[nparts - 1]);
            int br = extract_first_reg_num(last.c_str());
            if (br >= 0) {
                if (strstr(opcode, ".128")) {
                    for (int i = 1; i < 4; i++) {
                        char buf[16];
                        snprintf(buf, sizeof(buf), "R%d", br + i);
                        reads.insert(buf);
                    }
                } else if (strstr(opcode, ".64")) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "R%d", br + 1);
                    reads.insert(buf);
                }
            }
        }
        goto done;
    }

    // --- LDS: load from shared, reads addr, writes dest ---
    if (strcmp(base, "LDS") == 0) {
        if (nparts > 0) {
            std::string p0 = span_str(parts[0]);
            int br = extract_first_reg_num(p0.c_str());
            if (br >= 0) {
                char buf[16];
                snprintf(buf, sizeof(buf), "R%d", br);
                writes.insert(buf);
                if (strstr(opcode, ".128")) {
                    for (int i = 1; i < 4; i++) {
                        snprintf(buf, sizeof(buf), "R%d", br + i);
                        writes.insert(buf);
                    }
                } else if (strstr(opcode, ".64")) {
                    snprintf(buf, sizeof(buf), "R%d", br + 1);
                    writes.insert(buf);
                }
            }
        }
        for (int i = 1; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
        goto done;
    }

    // --- ISETP / FSETP / ISET: writes predicates, reads regs ---
    if (strcmp(base, "ISETP") == 0 || strcmp(base, "FSETP") == 0 || strcmp(base, "ISET") == 0) {
        for (int i = 0; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            // Trim
            const char* s = pi.c_str();
            while (*s == ' ') s++;
            // Check if this is a predicate operand
            const char* ps = s;
            if (*ps == '!') ps++;
            if (*ps == 'P' && ps[1] >= '0' && ps[1] <= '9') {
                // Check it's purely P<num>
                const char* d = ps + 1;
                while (*d >= '0' && *d <= '9') d++;
                // Skip trailing spaces
                while (*d == ' ') d++;
                if (*d == '\0') {
                    // It's a predicate
                    std::string pname(ps, d - ps);
                    // Trim trailing space from pname
                    while (!pname.empty() && pname.back() == ' ') pname.pop_back();
                    if (pname != "PT" && writes.find(pname) == writes.end()) {
                        writes.insert(pname);
                    }
                    continue;
                }
            }
            if (pi == "PT" || pi == " PT" || pi == "PT ") continue;
            // It's a register operand
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
        goto done;
    }

    // --- Control flow: no reg r/w ---
    if (strcmp(base, "BRA") == 0 || strcmp(base, "EXIT") == 0 ||
        strcmp(base, "RET") == 0 || strcmp(base, "BREAK") == 0 ||
        strcmp(base, "BSYNC") == 0 || strcmp(base, "CALL") == 0) {
        goto done;
    }

    // --- Barriers / sync: read R/UR operands ---
    if (strcmp(base, "BAR") == 0 || strcmp(base, "DEPBAR") == 0 ||
        strcmp(base, "WARPSYNC") == 0 || strcmp(base, "NANOSLEEP") == 0 ||
        strcmp(base, "MEMBAR") == 0 || strcmp(base, "ERRBAR") == 0 ||
        strcmp(base, "CGAERRBAR") == 0 || strcmp(base, "UTCBAR") == 0) {
        for (int i = 0; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
        goto done;
    }

    // --- TCGEN05: read all R/UR operands ---
    if (strcmp(base, "TCGEN05") == 0) {
        for (int i = 0; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
        goto done;
    }

    // --- S2R / CS2R: writes dest reg ---
    if (strcmp(base, "S2R") == 0 || strcmp(base, "CS2R") == 0) {
        if (nparts > 0) {
            std::string p0 = span_str(parts[0]);
            int br = extract_first_reg_num(p0.c_str());
            if (br >= 0) {
                char buf[16];
                snprintf(buf, sizeof(buf), "R%d", br);
                writes.insert(buf);
                if (strstr(opcode, ".64") && !strstr(opcode, ".32")) {
                    snprintf(buf, sizeof(buf), "R%d", br + 1);
                    writes.insert(buf);
                }
            }
        }
        goto done;
    }

    // --- S2UR: writes uniform dest ---
    if (strcmp(base, "S2UR") == 0) {
        if (nparts > 0) {
            std::string p0 = span_str(parts[0]);
            std::unordered_set<std::string> tmp;
            extract_uregs(p0.c_str(), tmp);
            for (auto& r : tmp) writes.insert(r);
        }
        goto done;
    }

    // --- R2UR: reads reg, writes uniform reg ---
    if (strcmp(base, "R2UR") == 0) {
        if (nparts >= 2) {
            std::string p0 = span_str(parts[0]);
            std::unordered_set<std::string> tmp;
            extract_uregs(p0.c_str(), tmp);
            for (auto& r : tmp) writes.insert(r);

            std::string p1 = span_str(parts[1]);
            extract_regs(p1.c_str(), reads);
        }
        goto done;
    }

    // --- Default: first operand = dest (write), rest = sources (read) ---
    if (nparts > 0) {
        {
            std::string p0 = span_str(parts[0]);
            extract_regs(p0.c_str(), writes);
            extract_uregs(p0.c_str(), writes);
        }
        for (int i = 1; i < nparts; i++) {
            std::string pi = span_str(parts[i]);
            // Skip RZ, URZ, PT, !PT
            const char* s = pi.c_str();
            while (*s == ' ') s++;
            if (strcmp(s, "RZ") == 0 || strcmp(s, "URZ") == 0 ||
                strcmp(s, "PT") == 0 || strcmp(s, "!PT") == 0)
                continue;
            extract_regs(pi.c_str(), reads);
            extract_uregs(pi.c_str(), reads);
        }
    }

done:
    // Move sets to sorted vectors
    reads_out.assign(reads.begin(), reads.end());
    std::sort(reads_out.begin(), reads_out.end());
    writes_out.assign(writes.begin(), writes.end());
    std::sort(writes_out.begin(), writes_out.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// SASS parser — manual text parsing (no regex)
// ═══════════════════════════════════════════════════════════════════════════

// Skip whitespace, return pointer to first non-whitespace
static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

// Find end of line (pointer to '\n' or '\0')
static const char* find_eol(const char* p) {
    while (*p && *p != '\n') p++;
    return p;
}

// Parse hex number from string, return value and advance pointer
static uint64_t parse_hex(const char* s, const char** end) {
    uint64_t val = 0;
    while ((*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'f') || (*s >= 'A' && *s <= 'F')) {
        if (*s >= '0' && *s <= '9') val = (val << 4) | (*s - '0');
        else if (*s >= 'a' && *s <= 'f') val = (val << 4) | (*s - 'a' + 10);
        else val = (val << 4) | (*s - 'A' + 10);
        s++;
    }
    if (end) *end = s;
    return val;
}

// Check if line contains "Function" and ":"
static bool is_function_header(const char* line, const char* eol, std::string& name) {
    const char* f = strstr(line, "Function");
    if (!f || f >= eol) return false;
    // Find ':'
    const char* colon = strchr(f, ':');
    if (!colon || colon >= eol) return false;
    // Skip "Function" and whitespace, then ':'
    const char* p = colon + 1;
    while (*p == ' ' || *p == '\t') p++;
    // Read function name (until whitespace or eol)
    const char* start = p;
    while (p < eol && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    name.assign(start, p - start);
    return !name.empty();
}

// Try to parse an instruction line:
//   /*ADDR*/  [@pred] OPCODE operands ;  /* 0xENC */
// Returns true if parsed, fills out fields.
static bool parse_instr_line(const char* line, const char* eol,
                             uint32_t& addr, std::string& pred,
                             std::string& opcode, std::string& operands,
                             uint64_t& encoding) {
    const char* p = skip_ws(line);

    // Must start with /*
    if (p[0] != '/' || p[1] != '*') return false;

    // Must contain ';' (distinguishes instr from ctrl line)
    const char* semi = nullptr;
    for (const char* s = p; s < eol; s++) {
        if (*s == ';') { semi = s; break; }
    }
    if (!semi) return false;

    // Extract addr: hex between first /* */
    p += 2; // skip /*
    const char* he;
    addr = (uint32_t)parse_hex(p, &he);
    // Skip to end of first comment
    p = strstr(he, "*/");
    if (!p || p >= eol) return false;
    p += 2;
    p = skip_ws(p);

    // Optional predicate: @[!]P<digit>
    pred.clear();
    if (*p == '@') {
        const char* ps = p;
        p++; // skip @
        if (*p == '!') p++;
        if (*p == 'P') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
            pred.assign(ps, p - ps);
            p = skip_ws(p);
        } else {
            p = ps; // not a predicate, rewind
        }
    }

    // Opcode: next non-whitespace token
    const char* op_start = p;
    while (p < semi && *p != ' ' && *p != '\t') p++;
    opcode.assign(op_start, p - op_start);
    if (opcode.empty()) return false;

    // Operands: everything between opcode and ';', trimmed
    p = skip_ws(p);
    const char* ops_start = p;
    // Trim trailing whitespace before ';'
    const char* ops_end = semi;
    while (ops_end > ops_start && (ops_end[-1] == ' ' || ops_end[-1] == '\t'))
        ops_end--;
    operands.assign(ops_start, ops_end - ops_start);

    // Encoding: last 0x... before final */
    // Scan backwards from eol for "0x"
    encoding = 0;
    const char* last_0x = nullptr;
    for (const char* s = eol - 1; s > semi; s--) {
        if (s[0] == '0' && s + 1 < eol && s[1] == 'x') {
            last_0x = s;
            break;
        }
    }
    if (last_0x) {
        encoding = parse_hex(last_0x + 2, nullptr);
    }

    return true;
}

// Try to parse a control word line:
//   /* 0xCONTROL */
// Returns true if it looks like a comment-only line (no ';')
static bool parse_ctrl_line(const char* line, const char* eol, uint64_t& ctrl) {
    const char* p = skip_ws(line);
    if (p[0] != '/' || p[1] != '*') return false;

    // Must NOT contain ';' (that would be an instruction)
    for (const char* s = p; s < eol; s++) {
        if (*s == ';') return false;
    }

    // Find 0x
    const char* ox = strstr(p, "0x");
    if (!ox || ox >= eol) return false;
    ctrl = parse_hex(ox + 2, nullptr);
    return true;
}

std::vector<Kernel> parse_sass(const char* text, size_t len,
                                const char* kernel_filter) {
    std::vector<Kernel> kernels;
    Kernel current;
    bool have_kernel = false;

    // Pending instruction state
    bool have_pending = false;
    uint32_t pend_addr = 0;
    std::string pend_pred, pend_op, pend_operands;
    uint64_t pend_enc = 0;

    const char* p = text;
    const char* end = text + len;

    while (p < end) {
        const char* eol = find_eol(p);

        // Try function header
        std::string fname;
        if (is_function_header(p, eol, fname)) {
            // Flush current kernel
            if (have_kernel && !current.instrs.empty()) {
                kernels.push_back(std::move(current));
            }
            current = Kernel{};
            current.name = fname;
            have_kernel = true;
            have_pending = false;
            goto next_line;
        }

        // Try instruction line
        {
            uint32_t addr;
            std::string pred, op, ops;
            uint64_t enc;
            if (parse_instr_line(p, eol, addr, pred, op, ops, enc)) {
                pend_addr = addr;
                pend_pred = std::move(pred);
                pend_op = std::move(op);
                pend_operands = std::move(ops);
                pend_enc = enc;
                have_pending = true;
                goto next_line;
            }
        }

        // Try control word line
        if (have_pending) {
            uint64_t ctrl;
            if (parse_ctrl_line(p, eol, ctrl)) {
                CtrlFields cf = decode_control(ctrl);
                Instruction instr;
                instr.addr = pend_addr;
                instr.opcode = std::move(pend_op);
                instr.operands = std::move(pend_operands);
                instr.predicate = std::move(pend_pred);
                instr.encoding = pend_enc;
                instr.control = ctrl;
                instr.ctrl = cf;
                instr.category = categorize(instr.opcode.c_str());
                instr.latency = get_latency(instr.opcode.c_str());
                parse_reg_sets(instr.opcode.c_str(), instr.operands.c_str(),
                              instr.predicate.c_str(),
                              instr.regs_read, instr.regs_written);
                current.instrs.push_back(std::move(instr));
                have_pending = false;
                goto next_line;
            }
        }

    next_line:
        p = (*eol == '\n') ? eol + 1 : eol;
    }

    // Flush last kernel
    if (have_kernel && !current.instrs.empty()) {
        kernels.push_back(std::move(current));
    }

    // Filter
    if (kernel_filter && kernel_filter[0]) {
        std::vector<Kernel> filtered;
        // Try exact match first
        for (auto& k : kernels) {
            if (k.name == kernel_filter) {
                filtered.push_back(std::move(k));
                return filtered;
            }
        }
        // Try substring match
        for (auto& k : kernels) {
            if (k.name.find(kernel_filter) != std::string::npos) {
                filtered.push_back(std::move(k));
                return filtered;
            }
        }
        // Not found — return empty (caller handles warning)
        return filtered;
    }

    return kernels;
}

// ═══════════════════════════════════════════════════════════════════════════
// Dependency graph
// ═══════════════════════════════════════════════════════════════════════════

std::vector<DepEdge> build_dep_graph(const std::vector<Instruction>& instrs,
                                      bool raw_only) {
    std::vector<DepEdge> edges;
    std::unordered_map<std::string, int> last_writer;
    std::unordered_map<std::string, std::vector<int>> last_readers;

    for (int i = 0; i < (int)instrs.size(); i++) {
        const auto& instr = instrs[i];

        // RAW: I read a register that was written by an earlier instruction
        for (const auto& reg : instr.regs_read) {
            auto it = last_writer.find(reg);
            if (it != last_writer.end()) {
                int j = it->second;
                edges.push_back({j, i, reg, DepEdge::RAW, instrs[j].latency});
            }
        }

        if (!raw_only) {
            // WAR: I write a register that was read by an earlier instruction
            for (const auto& reg : instr.regs_written) {
                auto it = last_readers.find(reg);
                if (it != last_readers.end()) {
                    for (int j : it->second) {
                        edges.push_back({j, i, reg, DepEdge::WAR, 0});
                    }
                }
            }
            // WAW: I write a register that was written by an earlier instruction
            for (const auto& reg : instr.regs_written) {
                auto it = last_writer.find(reg);
                if (it != last_writer.end()) {
                    int j = it->second;
                    edges.push_back({j, i, reg, DepEdge::WAW, 0});
                }
            }
        }

        // Update tracking
        for (const auto& reg : instr.regs_written) {
            last_writer[reg] = i;
            last_readers[reg].clear();
        }
        for (const auto& reg : instr.regs_read) {
            last_readers[reg].push_back(i);
        }
    }

    return edges;
}

// ═══════════════════════════════════════════════════════════════════════════
// Slack analysis
// ═══════════════════════════════════════════════════════════════════════════

std::vector<SlackInfo> compute_slack(const std::vector<Instruction>& instrs,
                                      const std::vector<DepEdge>& edges) {
    int n = (int)instrs.size();

    // Build predecessor map: for each instruction, list of (pred_idx, latency)
    std::vector<std::vector<std::pair<int,int>>> preds(n);
    for (const auto& e : edges) {
        if (e.type == DepEdge::RAW) {
            preds[e.dst].push_back({e.src, e.latency});
        }
    }

    // Prefix sum of stalls for O(1) range queries
    std::vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + instrs[i].stall();
    }

    std::vector<SlackInfo> results;
    results.reserve(n);

    for (int i = 0; i < n; i++) {
        int actual = instrs[i].stall();
        int wait_mask = instrs[i].ctrl.wait_mask;
        bool has_barrier_wait = (wait_mask != 0);

        int min_stall = 0;
        int lim_idx = -1;
        const char* lim_op = nullptr;

        for (auto& [pred_idx, lat] : preds[i]) {
            int distance = i - pred_idx - 1;
            int intervening_stalls = prefix[i] - prefix[pred_idx + 1];
            int effective_distance = distance + intervening_stalls;
            int needed = std::max(0, lat - effective_distance);
            if (needed > min_stall) {
                min_stall = needed;
                lim_idx = pred_idx;
                lim_op = instrs[pred_idx].opcode.c_str();
            }
        }

        int raw_slack = (actual >= 0) ? std::max(0, actual - min_stall) : -1;

        SlackInfo::SlackType stype;
        if (has_barrier_wait && raw_slack > 0)
            stype = SlackInfo::BARRIER_WAIT;
        else if (raw_slack > 0)
            stype = SlackInfo::SCHEDULING;
        else
            stype = SlackInfo::NONE;

        results.push_back({
            .idx = i,
            .addr = instrs[i].addr,
            .opcode = instrs[i].opcode.c_str(),
            .category = instrs[i].category,
            .actual_stall = actual,
            .min_stall = min_stall,
            .slack = raw_slack,
            .type = stype,
            .wait_mask = wait_mask,
            .limiting_dep_idx = lim_idx,
            .limiting_dep_op = lim_op,
        });
    }

    return results;
}

// ═══════════════════════════════════════════════════════════════════════════
// Critical path
// ═══════════════════════════════════════════════════════════════════════════

int critical_path(const std::vector<Instruction>& instrs,
                  const std::vector<DepEdge>& edges,
                  std::vector<int>& path) {
    int n = (int)instrs.size();
    if (n == 0) { path.clear(); return 0; }

    // Edges grouped by destination
    std::vector<std::vector<const DepEdge*>> preds(n);
    for (const auto& e : edges) {
        if (e.type == DepEdge::RAW) {
            preds[e.dst].push_back(&e);
        }
    }

    std::vector<int> dp(n, 0);
    std::vector<int> parent(n, -1);

    for (int i = 0; i < n; i++) {
        int best = 0;
        int best_parent = -1;
        for (const auto* e : preds[i]) {
            int cost = dp[e->src] + e->latency;
            if (cost > best) {
                best = cost;
                best_parent = e->src;
            }
        }
        dp[i] = best;
        parent[i] = best_parent;
    }

    // Find end of critical path
    int end_idx = 0;
    for (int i = 1; i < n; i++) {
        if (dp[i] > dp[end_idx]) end_idx = i;
    }
    int total = dp[end_idx];

    // Trace back
    path.clear();
    int cur = end_idx;
    while (cur >= 0) {
        path.push_back(cur);
        cur = parent[cur];
    }
    std::reverse(path.begin(), path.end());

    return total;
}

// ═══════════════════════════════════════════════════════════════════════════
// Output — print_listing
// ═══════════════════════════════════════════════════════════════════════════

void print_listing(FILE* out, const std::vector<Instruction>& instrs,
                   const std::vector<SlackInfo>* slack,
                   uint32_t sec_lo, uint32_t sec_hi) {
    // Build slack lookup
    std::unordered_map<int, const SlackInfo*> slack_map;
    if (slack) {
        for (const auto& s : *slack)
            slack_map[s.idx] = &s;
    }

    // Header
    if (slack) {
        fprintf(out, "%6s  %5s %3s %5s  %-12s  INSTRUCTION\n",
                "ADDR", "STALL", "MIN", "SLACK", "CAT");
        for (int i = 0; i < 100; i++) fputc('-', out);
        fputc('\n', out);
    } else {
        fprintf(out, "%6s  %5s %5s %5s %5s %6s  %-12s  INSTRUCTION\n",
                "ADDR", "STALL", "YIELD", "WRBAR", "RDBAR", "WAIT", "CAT");
        for (int i = 0; i < 110; i++) fputc('-', out);
        fputc('\n', out);
    }

    for (int i = 0; i < (int)instrs.size(); i++) {
        const auto& instr = instrs[i];
        if (instr.addr < sec_lo || instr.addr > sec_hi) continue;

        // Build instruction text
        char instr_text[256];
        if (!instr.predicate.empty())
            snprintf(instr_text, sizeof(instr_text), "%s %s %s",
                     instr.predicate.c_str(), instr.opcode.c_str(), instr.operands.c_str());
        else
            snprintf(instr_text, sizeof(instr_text), "%s %s",
                     instr.opcode.c_str(), instr.operands.c_str());

        if (slack) {
            auto it = slack_map.find(i);
            if (it != slack_map.end()) {
                const SlackInfo* s = it->second;
                char slack_s[16] = "";
                if (s->slack > 0) snprintf(slack_s, sizeof(slack_s), "%5d", s->slack);
                else snprintf(slack_s, sizeof(slack_s), "     ");

                const char* marker = "";
                char bar_buf[32];
                if (s->slack >= 2 && s->type == SlackInfo::SCHEDULING)
                    marker = " <<< SCHED";
                else if (s->slack >= 2 && s->type == SlackInfo::BARRIER_WAIT) {
                    snprintf(bar_buf, sizeof(bar_buf), " [bar 0b%06b]", s->wait_mask);
                    marker = bar_buf;
                }

                fprintf(out, "0x%04x  %5d %3d %s  %-12s  %s%s\n",
                        instr.addr, s->actual_stall, s->min_stall, slack_s,
                        instr.category, instr_text, marker);
            }
        } else {
            fprintf(out, "0x%04x  %5d %5d %5d %5d 0b%06b  %-12s  %s\n",
                    instr.addr, instr.ctrl.stall, instr.ctrl.yield,
                    instr.ctrl.wr_bar, instr.ctrl.rd_bar, instr.ctrl.wait_mask,
                    instr.category, instr_text);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Output — print_summary
// ═══════════════════════════════════════════════════════════════════════════

void print_summary(FILE* out, const std::vector<Instruction>& instrs,
                   const std::vector<SlackInfo>& slack,
                   uint32_t sec_lo, uint32_t sec_hi) {
    // Filter to section
    std::vector<const SlackInfo*> filtered;
    for (const auto& s : slack) {
        if (instrs[s.idx].addr >= sec_lo && instrs[s.idx].addr <= sec_hi)
            filtered.push_back(&s);
    }

    int total_stalls = 0, total_min = 0, total_slack = 0;
    int sched_slack = 0, barrier_slack = 0;

    for (const auto* s : filtered) {
        if (s->actual_stall >= 0) total_stalls += s->actual_stall;
        total_min += s->min_stall;
        if (s->slack > 0) {
            total_slack += s->slack;
            if (s->type == SlackInfo::SCHEDULING) sched_slack += s->slack;
            else if (s->type == SlackInfo::BARRIER_WAIT) barrier_slack += s->slack;
        }
    }
    int n_instructions = (int)filtered.size();

    fprintf(out, "\n");
    for (int i = 0; i < 60; i++) fputc('=', out);
    fprintf(out, "\nSCHEDULING ANALYSIS SUMMARY\n");
    fprintf(out, "  Decoder: assumed Ampere-style bit layout (NOT calibrated)\n");
    fprintf(out, "  bits [3:0]=stall appears correct; barrier fields unverified\n");
    fprintf(out, "  Run: python3 sass_analysis.py --calibrate > calibration.cu\n");
    if (sec_lo > 0 || sec_hi < UINT32_MAX)
        fprintf(out, "Section: 0x%04x - 0x%04x\n", sec_lo, sec_hi);
    for (int i = 0; i < 60; i++) fputc('=', out);
    fprintf(out, "\nInstructions:       %d\n", n_instructions);
    fprintf(out, "Total stall cycles: %d\n", total_stalls);
    fprintf(out, "Min stall cycles:   %d (RAW dependency-limited)\n", total_min);
    if (total_stalls > 0)
        fprintf(out, "Total slack:        %d cycles (%.1f%% of stalls)\n",
                total_slack, 100.0 * total_slack / total_stalls);
    else
        fprintf(out, "Total slack:        %d cycles\n", total_slack);
    fprintf(out, "  Scheduling slack: %d cycles (ptxas conservative — recoverable)\n", sched_slack);
    fprintf(out, "  Barrier waits:    %d cycles (scoreboard waits — NOT recoverable)\n", barrier_slack);

    // Category breakdown
    struct CatStats { int count; int stalls; int sched_slack; int bar_slack; };
    std::unordered_map<std::string, CatStats> cat_counts;
    for (const auto* s : filtered) {
        auto& cc = cat_counts[s->category];
        cc.count++;
        if (s->actual_stall >= 0) cc.stalls += s->actual_stall;
        if (s->slack > 0) {
            if (s->type == SlackInfo::SCHEDULING) cc.sched_slack += s->slack;
            else cc.bar_slack += s->slack;
        }
    }

    // Sort categories by stall count descending
    std::vector<std::pair<std::string, CatStats>> sorted_cats(cat_counts.begin(), cat_counts.end());
    std::sort(sorted_cats.begin(), sorted_cats.end(),
              [](const auto& a, const auto& b) { return a.second.stalls > b.second.stalls; });

    fprintf(out, "\n%-14s %6s %7s %8s %6s\n", "Category", "Count", "Stalls", "SchedSlk", "BarSlk");
    for (int i = 0; i < 44; i++) fputc('-', out);
    fputc('\n', out);
    for (const auto& [cat, cc] : sorted_cats) {
        fprintf(out, "%-14s %6d %7d %8d %6d\n",
                cat.c_str(), cc.count, cc.stalls, cc.sched_slack, cc.bar_slack);
    }

    // Top slack hotspots
    std::vector<const SlackInfo*> sched_hotspots, barrier_hotspots;
    for (const auto* s : filtered) {
        if (s->slack > 0) {
            if (s->type == SlackInfo::SCHEDULING) sched_hotspots.push_back(s);
            else if (s->type == SlackInfo::BARRIER_WAIT) barrier_hotspots.push_back(s);
        }
    }
    std::sort(sched_hotspots.begin(), sched_hotspots.end(),
              [](const auto* a, const auto* b) { return a->slack > b->slack; });
    std::sort(barrier_hotspots.begin(), barrier_hotspots.end(),
              [](const auto* a, const auto* b) { return a->slack > b->slack; });

    if (!sched_hotspots.empty()) {
        fprintf(out, "\nScheduling slack hotspots (recoverable via reordering):\n");
        int limit = std::min(10, (int)sched_hotspots.size());
        for (int i = 0; i < limit; i++) {
            const auto* h = sched_hotspots[i];
            fprintf(out, "  0x%04x  %-20s  stall=%d min=%d slack=%d",
                    h->addr, h->opcode, h->actual_stall, h->min_stall, h->slack);
            if (h->limiting_dep_idx >= 0 && h->limiting_dep_op) {
                fprintf(out, "  (dep: 0x%04x %s)",
                        instrs[h->limiting_dep_idx].addr, h->limiting_dep_op);
            }
            fputc('\n', out);
        }
    }

    if (!barrier_hotspots.empty()) {
        fprintf(out, "\nBarrier wait stalls (scoreboard, not scheduling slack):\n");
        int limit = std::min(10, (int)barrier_hotspots.size());
        for (int i = 0; i < limit; i++) {
            const auto* h = barrier_hotspots[i];
            fprintf(out, "  0x%04x  %-20s  stall=%d wait_mask=0b%06b\n",
                    h->addr, h->opcode, h->actual_stall, h->wait_mask);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Output — JSON
// ═══════════════════════════════════════════════════════════════════════════

void write_json(FILE* out, const char* kernel_name,
                const std::vector<Instruction>& instrs,
                const std::vector<SlackInfo>& slack,
                int cp_cycles, const std::vector<int>& cp_path,
                uint32_t sec_lo, uint32_t sec_hi) {
    // Filter to section
    std::vector<const SlackInfo*> filtered;
    for (const auto& s : slack) {
        if (instrs[s.idx].addr >= sec_lo && instrs[s.idx].addr <= sec_hi)
            filtered.push_back(&s);
    }

    int total_stalls = 0, total_min = 0, total_slack = 0;
    std::unordered_map<std::string, int> cat_counts;
    for (const auto* s : filtered) {
        if (s->actual_stall >= 0) total_stalls += s->actual_stall;
        total_min += s->min_stall;
        if (s->slack > 0) total_slack += s->slack;
        cat_counts[s->category]++;
    }

    // Hotspots (top 20 by slack)
    std::vector<const SlackInfo*> hotspots;
    for (const auto* s : filtered) {
        if (s->slack > 0) hotspots.push_back(s);
    }
    std::sort(hotspots.begin(), hotspots.end(),
              [](const auto* a, const auto* b) { return a->slack > b->slack; });
    if (hotspots.size() > 20) hotspots.resize(20);

    fprintf(out, "{\n");
    fprintf(out, "  \"kernel\": \"%s\",\n", kernel_name);
    fprintf(out, "  \"ctrl_layout\": {\"stall\": [0, 4], \"yield\": [4, 1], \"wr_bar\": [5, 5], "
                 "\"rd_bar\": [10, 5], \"wait_mask\": [15, 6], \"reuse\": [21, 2]},\n");
    fprintf(out, "  \"ctrl_layout_calibrated\": true,\n");
    fprintf(out, "  \"latency_table_source\": \"b200_calibrated_k1_k12\",\n");
    fprintf(out, "  \"n_instructions\": %d,\n", (int)filtered.size());
    fprintf(out, "  \"total_stall_cycles\": %d,\n", total_stalls);
    fprintf(out, "  \"min_stall_cycles\": %d,\n", total_min);
    fprintf(out, "  \"total_slack\": %d,\n", total_slack);
    if (total_stalls > 0)
        fprintf(out, "  \"slack_pct\": %.1f,\n", 100.0 * total_slack / total_stalls);
    else
        fprintf(out, "  \"slack_pct\": 0,\n");
    fprintf(out, "  \"critical_path_cycles\": %d,\n", cp_cycles);
    fprintf(out, "  \"critical_path_length\": %d,\n", (int)cp_path.size());

    if (sec_lo > 0 || sec_hi < UINT32_MAX) {
        fprintf(out, "  \"section\": {\"start\": \"0x%04x\", \"end\": \"0x%04x\"},\n",
                sec_lo, sec_hi);
    }

    // Category counts
    fprintf(out, "  \"category_counts\": {");
    bool first = true;
    for (const auto& [cat, cnt] : cat_counts) {
        if (!first) fprintf(out, ", ");
        fprintf(out, "\"%s\": %d", cat.c_str(), cnt);
        first = false;
    }
    fprintf(out, "},\n");

    // Hotspots
    fprintf(out, "  \"hotspots\": [\n");
    for (int i = 0; i < (int)hotspots.size(); i++) {
        const auto* h = hotspots[i];
        fprintf(out, "    {\"addr\": \"0x%04x\", \"opcode\": \"%s\", "
                     "\"actual_stall\": %d, \"min_stall\": %d, \"slack\": %d}",
                h->addr, h->opcode, h->actual_stall, h->min_stall, h->slack);
        if (i + 1 < (int)hotspots.size()) fputc(',', out);
        fputc('\n', out);
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Calibration — CUDA source generation
// ═══════════════════════════════════════════════════════════════════════════

static const char CALIBRATION_SOURCE[] = R"(// SASS control word calibration kernels for SM 100a
// Compile: nvcc -arch=sm_100a -O3 calibration.cu -o calibration
// Dump:    cuobjdump --dump-sass calibration > calibration_sass.txt
// Analyze: python3 sass_analysis.py calibration_sass.txt
//
// Purpose: Each kernel has a known dependency pattern. By examining the
// control words in the SASS output, we can identify which bits encode
// the stall count and verify the decoder.

#include <cstdio>
#include <cstdint>

// Kernel A: Independent operations — expect stall=0 between them
extern "C" __global__ void cal_independent(int* out) {
    int a = 1, b = 2, c = 3, d = 4;
    int e = 5, f = 6, g = 7, h = 8;
    // 8 independent adds — no data dependencies
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(d));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(e));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(f));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(g));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(h));
    out[threadIdx.x] = a + b + c + d + e + f + g + h;
}

// Kernel B: Tight RAW dependency chain — expect stall=latency between each
extern "C" __global__ void cal_dependent(int* out) {
    int a = threadIdx.x;
    // Each add depends on the previous result
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    out[threadIdx.x] = a;
}

// Kernel C: RAW with 1 independent op between — expect stall = max(0, latency-1)
extern "C" __global__ void cal_gap1(int* out) {
    int a = threadIdx.x, b = threadIdx.x + 1;
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // write a
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));  // independent (gap=1)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // read a (stall = lat-1)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    out[threadIdx.x] = a + b;
}

// Kernel D: RAW with 2 independent ops between — expect stall = max(0, latency-2)
extern "C" __global__ void cal_gap2(int* out) {
    int a = threadIdx.x, b = threadIdx.x + 1, c = threadIdx.x + 2;
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // write a
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));  // independent
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));  // independent (gap=2)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // read a (stall = lat-2)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    out[threadIdx.x] = a + b + c;
}

// Kernel E: Shared memory store — measure STS issue latency
extern "C" __global__ void cal_sts(int* out) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int a = tid, b = tid + 1;
    // STS is fire-and-forget — stall should be minimal
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4), "r"(a));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4 + 1024), "r"(b));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4), "r"(a));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4 + 1024), "r"(b));
    __syncthreads();
    out[tid] = smem[tid];
}

// Kernel F: F2FP (float→bf16 conversion) latency
extern "C" __global__ void cal_f2fp(float* out) {
    float a = (float)threadIdx.x;
    unsigned r;
    // Dependent chain: convert, extract, use as input again
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    out[threadIdx.x] = a;
}

// Kernel G: PRMT (permute) latency
extern "C" __global__ void cal_prmt(int* out) {
    unsigned a = threadIdx.x, b = threadIdx.x + 1;
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    out[threadIdx.x] = a;
}

int main() {
    int *d_out;
    float *d_fout;
    cudaMalloc(&d_out, 256 * sizeof(int));
    cudaMalloc(&d_fout, 256 * sizeof(float));

    cal_independent<<<1, 32>>>(d_out);
    cal_dependent<<<1, 32>>>(d_out);
    cal_gap1<<<1, 32>>>(d_out);
    cal_gap2<<<1, 32>>>(d_out);
    cal_sts<<<1, 32>>>(d_out);
    cal_f2fp<<<1, 32>>>(d_fout);
    cal_prmt<<<1, 32>>>(d_out);

    cudaDeviceSynchronize();
    printf("Calibration kernels launched. Dump SASS with:\n");
    printf("  cuobjdump --dump-sass calibration > calibration_sass.txt\n");
    printf("  python3 sass_analysis.py calibration_sass.txt\n");

    cudaFree(d_out);
    cudaFree(d_fout);
    return 0;
}
)";

void print_calibration(FILE* out) {
    fputs(CALIBRATION_SOURCE, out);
}

// ═══════════════════════════════════════════════════════════════════════════
// Calibration — SASS vs runtime comparison
// ═══════════════════════════════════════════════════════════════════════════

struct CalKernel {
    const char* fn;
    const char* label;
    const char* ops[6]; // null-terminated
    enum Mode { THROUGHPUT, LATENCY, CONFLICT } mode;
};

static const CalKernel CAL_KERNELS[] = {
    {"k1_f2fp_throughput",     "K1: F2FP throughput",   {"F2FP", nullptr},                              CalKernel::THROUGHPUT},
    {"k2_f2fp_latency",        "K2: F2FP latency",      {"F2FP", nullptr},                              CalKernel::LATENCY},
    {"k3_f2fp_sts_conflict",   "K3: F2FP+STS",          {"F2FP", "STS", nullptr},                       CalKernel::CONFLICT},
    {"k4_hfma2_throughput",    "K4: HFMA2 throughput",   {"HFMA2", "HMMA", "HFMA", nullptr},             CalKernel::THROUGHPUT},
    {"k5_hfma2_f2fp_conflict", "K5: HFMA2+F2FP",        {"HFMA2", "HMMA", "HFMA", "F2FP", nullptr},     CalKernel::CONFLICT},
    {"k6_sts_throughput",      "K6: STS throughput",     {"STS", nullptr},                               CalKernel::THROUGHPUT},
    {"k7a_iadd_independent",   "K7a: IADD indep",        {"IADD3", "IADD", "UIADD3", nullptr},           CalKernel::THROUGHPUT},
    {"k7b_iadd_dependent",     "K7b: IADD dep",          {"IADD3", "IADD", "UIADD3", nullptr},           CalKernel::LATENCY},
    {"k8_prmt_throughput",     "K8: PRMT throughput",    {"PRMT", "UPRMT", nullptr},                     CalKernel::THROUGHPUT},
    {"k9_f2fp_wide",           "K9: F2FP 32-wide",       {"F2FP", nullptr},                              CalKernel::THROUGHPUT},
    {"k10_hadd2_throughput",   "K10: HADD2 throughput",  {"HADD2", nullptr},                             CalKernel::THROUGHPUT},
    {"k11_hadd2_latency",      "K11: HADD2 latency",     {"HADD2", nullptr},                             CalKernel::LATENCY},
    {"k12_ldg_latency",        "K12: LDG latency",       {"LDG", nullptr},                               CalKernel::LATENCY},
};
static const int N_CAL_KERNELS = sizeof(CAL_KERNELS) / sizeof(CAL_KERNELS[0]);

// Find indices of the timed region between clock reads (CS2R / S2R CLOCK)
static std::pair<int, int> find_timed_region(const std::vector<Instruction>& instrs) {
    std::vector<int> clock_idx;
    for (int i = 0; i < (int)instrs.size(); i++) {
        char base[64];
        extract_base(instrs[i].opcode.c_str(), base, sizeof(base));
        if (strcmp(base, "CS2R") == 0) {
            clock_idx.push_back(i);
        } else if (strcmp(base, "S2R") == 0) {
            // Check if operands contain CLOCK
            const char* p = instrs[i].operands.c_str();
            // Simple case-insensitive check
            bool has_clock = false;
            for (const char* s = p; *s; s++) {
                if ((*s == 'C' || *s == 'c') && strncasecmp(s, "CLOCK", 5) == 0) {
                    has_clock = true;
                    break;
                }
            }
            if (has_clock) clock_idx.push_back(i);
        }
    }
    if (clock_idx.size() >= 2)
        return {clock_idx[clock_idx.size() - 2] + 1, clock_idx.back()};
    return {-1, -1};
}

struct StallStats {
    int count, typical, min_val, max_val;
};

// Extract stall counts for target opcodes in the timed region
static std::unordered_map<std::string, StallStats>
extract_timed_stalls(const std::vector<Instruction>& instrs,
                     const CalKernel& kdef, int start, int end) {
    // Collect target ops into a set
    std::unordered_set<std::string> targets;
    for (int i = 0; kdef.ops[i]; i++)
        targets.insert(kdef.ops[i]);

    // Collect stalls per opcode
    std::unordered_map<std::string, std::vector<int>> by_op;
    for (int i = std::max(0, start); i < std::min((int)instrs.size(), end); i++) {
        char base[64];
        extract_base(instrs[i].opcode.c_str(), base, sizeof(base));
        if (targets.count(base))
            by_op[base].push_back(instrs[i].stall());
    }

    std::unordered_map<std::string, StallStats> result;
    for (auto& [op, stalls] : by_op) {
        // Compute mode (most frequent value)
        int freq[16] = {};
        int mn = 15, mx = 0;
        for (int s : stalls) {
            if (s >= 0 && s <= 15) freq[s]++;
            if (s < mn) mn = s;
            if (s > mx) mx = s;
        }
        int mode_val = 0, mode_cnt = 0;
        for (int i = 0; i < 16; i++) {
            if (freq[i] > mode_cnt) { mode_cnt = freq[i]; mode_val = i; }
        }
        result[op] = {(int)stalls.size(), mode_val, mn, mx};
    }
    return result;
}

struct RuntimeResult {
    int total_cycles;
    double cyc_per_instr, throughput;
};

// Parse calibration binary stdout into per-kernel measurements
static std::unordered_map<std::string, RuntimeResult>
parse_runtime_output(const char* text) {
    std::unordered_map<std::string, RuntimeResult> results;
    const char* p = text;
    while (*p) {
        // Find line start matching K\d+[ab]?:
        if (*p == 'K' && p[1] >= '0' && p[1] <= '9') {
            const char* ls = p;
            // Extract label prefix (K1, K7a, K12, etc.)
            p++;
            while (*p >= '0' && *p <= '9') p++;
            if (*p == 'a' || *p == 'b') p++;
            if (*p == ':') {
                std::string key(ls, p - ls);
                // Scan forward for 3 numbers at end of line
                // Format: "K1: <description>  <total> <cyc_per_instr> <throughput>"
                // Find end of line
                const char* eol = p;
                while (*eol && *eol != '\n') eol++;
                // Scan backwards from eol to find 3 whitespace-separated numbers
                // Use sscanf with a scan through the line after ':'
                int total;
                double cpi, tput;
                // Try to find the pattern: multiple spaces then int double double
                const char* scan = p + 1;
                bool found = false;
                while (scan < eol && !found) {
                    // Look for sequences of spaces (field separator)
                    if (*scan == ' ' && scan + 1 < eol && scan[1] == ' ') {
                        if (sscanf(scan, " %d %lf %lf", &total, &cpi, &tput) == 3) {
                            results[key] = {total, cpi, tput};
                            found = true;
                        }
                    }
                    scan++;
                }
                p = eol;
            }
        }
        // Advance to next line
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
    }
    return results;
}

// Extract Knn prefix from label like "K1: F2FP throughput"
static std::string extract_prefix(const char* label) {
    const char* p = label;
    if (*p != 'K') return "";
    p++;
    while (*p >= '0' && *p <= '9') p++;
    if (*p == 'a' || *p == 'b') p++;
    return std::string(label, p - label);
}

void calibrate_compare(FILE* out,
                       const std::vector<Kernel>& kernels,
                       const char* runtime_text) {
    auto runtime = runtime_text ? parse_runtime_output(runtime_text)
                                : std::unordered_map<std::string, RuntimeResult>{};
    bool has_rt = !runtime.empty();

    fprintf(out, "\n");
    if (has_rt) {
        fprintf(out, "SASS vs RUNTIME CALIBRATION COMPARISON\n");
        for (int i = 0; i < 78; i++) fputc('=', out);
        fprintf(out, "\n%-25s %-7s %5s %7s %6s  Verdict\n", "Kernel", "Op", "SASS", "Meas", "Delta");
        for (int i = 0; i < 78; i++) fputc('-', out);
        fputc('\n', out);
    } else {
        fprintf(out, "SASS CALIBRATION STALL ANALYSIS (no runtime data)\n");
        for (int i = 0; i < 65; i++) fputc('=', out);
        fprintf(out, "\n%-25s %-7s %5s %4s %7s  Type\n", "Kernel", "Op", "Stall", "N", "Range");
        for (int i = 0; i < 65; i++) fputc('-', out);
        fputc('\n', out);
    }

    for (int ki = 0; ki < N_CAL_KERNELS; ki++) {
        const CalKernel& kdef = CAL_KERNELS[ki];

        // Find kernel in SASS (substring match on function name)
        const Kernel* found = nullptr;
        for (const auto& k : kernels) {
            if (k.name.find(kdef.fn) != std::string::npos) {
                found = &k;
                break;
            }
        }
        if (!found) {
            fprintf(out, "%-25s  (not found in SASS)\n", kdef.label);
            continue;
        }

        auto [start, end] = find_timed_region(found->instrs);
        if (start < 0) {
            fprintf(out, "%-25s  (no clock reads found)\n", kdef.label);
            continue;
        }

        auto stall_data = extract_timed_stalls(found->instrs, kdef, start, end);
        if (stall_data.empty()) {
            // List what ops were found in the timed region
            std::unordered_set<std::string> found_ops;
            for (int i = start; i < end && i < (int)found->instrs.size(); i++) {
                char base[64];
                extract_base(found->instrs[i].opcode.c_str(), base, sizeof(base));
                found_ops.insert(base);
            }
            fprintf(out, "%-25s  (target ops not in timed region; found:", kdef.label);
            int cnt = 0;
            for (const auto& op : found_ops) {
                if (cnt++ > 0) fputc(',', out);
                fprintf(out, " %s", op.c_str());
                if (cnt >= 8) break;
            }
            fprintf(out, ")\n");
            continue;
        }

        std::string prefix = extract_prefix(kdef.label);
        auto rt_it = runtime.find(prefix);
        const RuntimeResult* rt = (rt_it != runtime.end()) ? &rt_it->second : nullptr;

        for (const auto& [op, sd] : stall_data) {
            int sass_stall = sd.typical;

            if (has_rt && rt) {
                double measured = rt->cyc_per_instr;

                if (kdef.mode == CalKernel::CONFLICT) {
                    fprintf(out, "%-25s %-7s %5d %7.2f         (conflict — see pipe analysis)\n",
                            kdef.label, op.c_str(), sass_stall, measured);
                } else {
                    double delta = sass_stall - measured;
                    const char* verdict;
                    char vbuf[64];
                    if (delta >= -0.5 && delta <= 0.5) {
                        verdict = "MATCH";
                    } else if (delta > 0.5) {
                        snprintf(vbuf, sizeof(vbuf), "CONSERVATIVE (+%.1f)", delta);
                        verdict = vbuf;
                    } else {
                        snprintf(vbuf, sizeof(vbuf), "DECODER? (%+.1f)", delta);
                        verdict = vbuf;
                    }
                    fprintf(out, "%-25s %-7s %5d %7.2f %+6.1f  %s\n",
                            kdef.label, op.c_str(), sass_stall, measured, delta, verdict);
                }
            } else {
                char rng[32];
                if (sd.min_val != sd.max_val)
                    snprintf(rng, sizeof(rng), "%d-%d", sd.min_val, sd.max_val);
                else
                    snprintf(rng, sizeof(rng), "%d", sd.min_val);
                fprintf(out, "%-25s %-7s %5d %4d %7s  %s\n",
                        kdef.label, op.c_str(), sass_stall, sd.count, rng,
                        kdef.mode == CalKernel::THROUGHPUT ? "throughput" :
                        kdef.mode == CalKernel::LATENCY ? "latency" : "conflict");
            }
        }
    }

    // SASS-only mode: print guidance and exit
    if (!has_rt) {
        fprintf(out, "\nExpected patterns:\n");
        fprintf(out, "  Throughput kernels (K1,K4,K6,K7a,K8,K9): stall = pipe issue rate\n");
        fprintf(out, "  Latency kernels (K2,K7b): stall = result latency\n");
        fprintf(out, "  K7a stall << K7b stall: decoder reads stall field correctly\n");
        fprintf(out, "  K7a stall == K7b stall: decoder is WRONG (bits not at [3:0])\n");
        fprintf(out, "\nNext: run ./calibration on B200, then:\n");
        fprintf(out, "  python3 sass_analysis.py <sass> --calibrate-compare --runtime output.txt\n");
        return;
    }

    // Pipe conflict analysis
    fprintf(out, "\n");
    for (int i = 0; i < 78; i++) fputc('=', out);
    fprintf(out, "\nPIPE CONFLICT ANALYSIS\n");
    for (int i = 0; i < 78; i++) fputc('-', out);
    fputc('\n', out);

    // Gather runtime values
    auto get_cpi = [&](const char* key) -> double {
        auto it = runtime.find(key);
        return (it != runtime.end()) ? it->second.cyc_per_instr : -1.0;
    };

    double k1 = get_cpi("K1"), k3 = get_cpi("K3"), k4 = get_cpi("K4");
    double k5 = get_cpi("K5"), k6 = get_cpi("K6");
    double k7a = get_cpi("K7a"), k7b = get_cpi("K7b"), k9 = get_cpi("K9");

    if (k1 >= 0 && k6 >= 0 && k3 >= 0) {
        double overlap = std::max(k1, k6);
        double serial = k1 + k6;
        if (k3 <= overlap * 1.3)
            fprintf(out, "  F2FP vs STS:   DIFFERENT pipes  (K3=%.2f ~ max(K1,K6)=%.2f)\n", k3, overlap);
        else if (k3 >= serial * 0.8)
            fprintf(out, "  F2FP vs STS:   SAME pipe        (K3=%.2f ~ K1+K6=%.2f)\n", k3, serial);
        else
            fprintf(out, "  F2FP vs STS:   PARTIAL overlap   (K3=%.2f, range [%.2f, %.2f])\n", k3, overlap, serial);
    }

    if (k1 >= 0 && k4 >= 0 && k5 >= 0) {
        double overlap = std::max(k1, k4);
        double serial = k1 + k4;
        if (k5 <= overlap * 1.3)
            fprintf(out, "  HFMA2 vs F2FP: DIFFERENT pipes  (K5=%.2f ~ max(K1,K4)=%.2f)\n", k5, overlap);
        else if (k5 >= serial * 0.8)
            fprintf(out, "  HFMA2 vs F2FP: SAME pipe        (K5=%.2f ~ K1+K4=%.2f)\n", k5, serial);
        else
            fprintf(out, "  HFMA2 vs F2FP: PARTIAL overlap   (K5=%.2f, range [%.2f, %.2f])\n", k5, overlap, serial);
    }

    if (k1 >= 0 && k9 >= 0) {
        double ratio = k9 / k1;
        if (ratio > 1.3)
            fprintf(out, "  32-wide F2FP:  DEGRADED  (K9/K1 = %.2fx)\n", ratio);
        else
            fprintf(out, "  32-wide F2FP:  OK        (K9/K1 = %.2fx)\n", ratio);
    }

    // Decoder verification via K7a/K7b
    if (k7a >= 0 && k7b >= 0) {
        double lat = k7b - k7a;
        fprintf(out, "\nDECODER VERIFICATION\n");
        for (int i = 0; i < 78; i++) fputc('-', out);
        fprintf(out, "\n  IADD3 throughput (K7a) = %.2f cyc/instr\n", k7a);
        fprintf(out, "  IADD3 latency   (K7b) = %.2f cyc/instr\n", k7b);
        fprintf(out, "  Derived latency       = %.2f cycles\n", lat);

        // Check K7b SASS stall against measured latency
        for (int ki = 0; ki < N_CAL_KERNELS; ki++) {
            if (strcmp(CAL_KERNELS[ki].fn, "k7b_iadd_dependent") != 0) continue;
            for (const auto& k : kernels) {
                if (k.name.find(CAL_KERNELS[ki].fn) == std::string::npos) continue;
                auto [s, e] = find_timed_region(k.instrs);
                auto sd = extract_timed_stalls(k.instrs, CAL_KERNELS[ki], s, e);
                for (const auto& [op, d] : sd) {
                    if (d.typical - lat >= -0.5 && d.typical - lat <= 0.5)
                        fprintf(out, "  SASS dep stall=%d matches measured %.1f -> DECODER CORRECT\n",
                                d.typical, lat);
                    else if (d.typical - k7b >= -0.5 && d.typical - k7b <= 0.5)
                        fprintf(out, "  SASS dep stall=%d matches raw K7b %.1f"
                                " -> stall encodes total cost (not just latency)\n", d.typical, k7b);
                    else
                        fprintf(out, "  SASS dep stall=%d vs measured %.1f -> MISMATCH (check bit layout)\n",
                                d.typical, lat);
                }
            }
        }
    }

    // Production kernel implications
    if (k1 >= 0) {
        fprintf(out, "\nPRODUCTION KERNEL IMPLICATIONS\n");
        for (int i = 0; i < 78; i++) fputc('-', out);
        double f2fp = k1;
        int ptxas_stall = 15;
        int n_f2fp = 128;
        int epi_budget = 6156;
        fprintf(out, "\n  F2FP measured throughput = %.2f cyc/instr\n", f2fp);
        fprintf(out, "  Production kernel: %d F2FPs with ptxas stall=%d\n", n_f2fp, ptxas_stall);
        if (f2fp < ptxas_stall - 0.5) {
            int headroom = ptxas_stall - (int)(f2fp + 0.5);
            int recoverable = headroom * n_f2fp;
            double pct = 100.0 * recoverable / epi_budget;
            fprintf(out, "  Headroom = (%d - %d) x %d = %d cycles\n",
                    ptxas_stall, (int)(f2fp + 0.5), n_f2fp, recoverable);
            fprintf(out, "  Epilogue tile budget ~ %d cycles -> %.1f%% potential\n", epi_budget, pct);
            if (recoverable > 500)
                fprintf(out, "  VERDICT: Significant SASS scheduling headroom exists\n");
            else
                fprintf(out, "  VERDICT: Modest headroom (%d cycles)\n", recoverable);
        } else {
            fprintf(out, "  ptxas stall=%d is correct -> NO scheduling headroom\n", ptxas_stall);
        }
    }

    // Suggested latency table updates
    struct Suggestion { std::string op; int current, measured; };
    std::vector<Suggestion> suggestions;
    auto& lt = latency_table();
    for (int ki = 0; ki < N_CAL_KERNELS; ki++) {
        if (CAL_KERNELS[ki].mode != CalKernel::THROUGHPUT) continue;
        std::string prefix = extract_prefix(CAL_KERNELS[ki].label);
        auto rt_it = runtime.find(prefix);
        if (rt_it == runtime.end()) continue;
        int measured = (int)(rt_it->second.cyc_per_instr + 0.5);
        for (int oi = 0; CAL_KERNELS[ki].ops[oi]; oi++) {
            auto lat_it = lt.find(CAL_KERNELS[ki].ops[oi]);
            if (lat_it != lt.end() && abs(lat_it->second - measured) >= 1)
                suggestions.push_back({CAL_KERNELS[ki].ops[oi], lat_it->second, measured});
        }
    }
    if (!suggestions.empty()) {
        fprintf(out, "\nLATENCY TABLE UPDATES\n");
        for (int i = 0; i < 78; i++) fputc('-', out);
        fputc('\n', out);
        for (const auto& s : suggestions)
            fprintf(out, "  '%s': %d -> %d\n", s.op.c_str(), s.current, s.measured);
        fprintf(out, "  (edit LATENCY dict in sass_analysis.py to apply)\n");
    }
}

} // namespace sass
