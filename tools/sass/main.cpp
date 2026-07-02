#include "sass.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>
#include <unordered_map>
#include <vector>

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS] [INPUT]\n\n"
        "SASS scheduling analysis for SM 100a (Blackwell)\n\n"
        "Positional:\n"
        "  INPUT                  SASS dump file (from cuobjdump --dump-sass)\n\n"
        "Options:\n"
        "  --cubin PATH           Run cuobjdump on PATH, parse stdout\n"
        "  -k, --kernel NAME      Kernel name filter (substring match)\n"
        "  -s, --section START END  Address range (hex)\n"
        "  -d, --deps             Dependency + slack analysis\n"
        "  -j, --json FILE        Write JSON output to file\n"
        "  --list-kernels         List kernel names and exit\n"
        "  --calibrate            Print calibration CUDA source and exit\n"
        "  --calibrate-compare    Compare SASS stall counts vs runtime\n"
        "  --runtime FILE         Runtime output from ./calibration\n"
        "  -h, --help             Show this help\n\n"
        "Examples:\n"
        "  %s sass_dump.txt\n"
        "  %s sass_dump.txt --deps\n"
        "  %s sass_dump.txt --section 0x1300 0x1a70\n"
        "  %s --cubin ./siglip_vision --deps\n"
        "  %s --calibrate > calibration.cu\n"
        "  %s cal_sass.txt --calibrate-compare --runtime output.txt\n",
        prog, prog, prog, prog, prog, prog, prog);
}

static std::string read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

static std::string run_cuobjdump(const char* path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "cuobjdump --dump-sass %s", path);
    FILE* p = popen(cmd, "r");
    if (!p) {
        fprintf(stderr, "Error: cannot run cuobjdump\n");
        exit(1);
    }
    std::string result;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof(buf), p))
        result.append(buf, n);
    int status = pclose(p);
    if (status != 0) {
        fprintf(stderr, "Error: cuobjdump failed (status %d)\n", status);
        exit(1);
    }
    return result;
}

int main(int argc, char** argv) {
    static struct option long_opts[] = {
        {"cubin",            required_argument, nullptr, 'C'},
        {"kernel",           required_argument, nullptr, 'k'},
        {"section",          required_argument, nullptr, 's'},
        {"deps",             no_argument,       nullptr, 'd'},
        {"json",             required_argument, nullptr, 'j'},
        {"list-kernels",     no_argument,       nullptr, 'L'},
        {"calibrate",        no_argument,       nullptr, 'A'},
        {"calibrate-compare", no_argument,      nullptr, 'B'},
        {"runtime",          required_argument, nullptr, 'R'},
        {"help",             no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    const char* cubin_path = nullptr;
    const char* kernel_filter = nullptr;
    const char* json_path = nullptr;
    const char* runtime_path = nullptr;
    uint32_t sec_lo = 0, sec_hi = UINT32_MAX;
    bool do_deps = false;
    bool list_kernels = false;
    bool do_calibrate = false;
    bool do_cal_compare = false;
    int opt;
    while ((opt = getopt_long(argc, argv, "k:s:dj:hL", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'C': cubin_path = optarg; break;
        case 'k': kernel_filter = optarg; break;
        case 's':
            sec_lo = (uint32_t)strtoul(optarg, nullptr, 16);
            // Next argument is END
            if (optind < argc) {
                sec_hi = (uint32_t)strtoul(argv[optind++], nullptr, 16);
            } else {
                fprintf(stderr, "Error: --section requires START and END\n");
                return 1;
            }
            break;
        case 'd': do_deps = true; break;
        case 'j': json_path = optarg; break;
        case 'L': list_kernels = true; break;
        case 'A': do_calibrate = true; break;
        case 'B': do_cal_compare = true; break;
        case 'R': runtime_path = optarg; break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 1;
        }
    }

    // --calibrate: print CUDA source and exit (no SASS input needed)
    if (do_calibrate) {
        sass::print_calibration(stdout);
        return 0;
    }

    // Get SASS text
    std::string sass_text;
    if (cubin_path) {
        sass_text = run_cuobjdump(cubin_path);
    } else if (optind < argc) {
        sass_text = read_file(argv[optind]);
    } else if (do_cal_compare) {
        fprintf(stderr, "Error: --calibrate-compare needs a SASS input (positional file or --cubin)\n");
        return 1;
    } else {
        usage(argv[0]);
        return 1;
    }

    // Parse (calibrate-compare uses all kernels, ignoring kernel_filter)
    auto kernels = sass::parse_sass(sass_text.c_str(), sass_text.size(),
                                     do_cal_compare ? nullptr : kernel_filter);
    if (kernels.empty()) {
        fprintf(stderr, "No kernels found in input.\n");
        return 1;
    }

    // --calibrate-compare: compare SASS stalls vs runtime
    if (do_cal_compare) {
        std::string runtime_text;
        if (runtime_path)
            runtime_text = read_file(runtime_path);
        sass::calibrate_compare(stdout, kernels,
                                runtime_path ? runtime_text.c_str() : nullptr);
        return 0;
    }

    if (list_kernels) {
        for (const auto& k : kernels)
            printf("%s: %zu instructions\n", k.name.c_str(), k.instrs.size());
        return 0;
    }

    // Process each kernel
    for (const auto& k : kernels) {
        printf("\n");
        for (int i = 0; i < 60; i++) putchar('=');
        printf("\nKernel: %s  (%zu instructions)\n", k.name.c_str(), k.instrs.size());
        for (int i = 0; i < 60; i++) putchar('=');
        printf("\n\n");

        if (do_deps) {
            auto edges = sass::build_dep_graph(k.instrs);
            auto slack_data = sass::compute_slack(k.instrs, edges);
            std::vector<int> cp_path;
            int cp_cycles = sass::critical_path(k.instrs, edges, cp_path);

            sass::print_listing(stdout, k.instrs, &slack_data, sec_lo, sec_hi);
            sass::print_summary(stdout, k.instrs, slack_data, sec_lo, sec_hi);

            printf("\nCritical path: %d cycles, %zu instructions\n",
                   cp_cycles, cp_path.size());
            if (cp_path.size() <= 20) {
                for (int idx : cp_path) {
                    const auto& instr = k.instrs[idx];
                    printf("  0x%04x  ", instr.addr);
                    if (!instr.predicate.empty())
                        printf("%s ", instr.predicate.c_str());
                    printf("%s %s\n", instr.opcode.c_str(), instr.operands.c_str());
                }
            }

            if (json_path) {
                FILE* jf = fopen(json_path, "w");
                if (!jf) {
                    fprintf(stderr, "Error: cannot open '%s' for writing\n", json_path);
                    return 1;
                }
                sass::write_json(jf, k.name.c_str(), k.instrs, slack_data,
                                cp_cycles, cp_path, sec_lo, sec_hi);
                fclose(jf);
                printf("\nJSON written to %s\n", json_path);
            }
        } else {
            sass::print_listing(stdout, k.instrs, nullptr, sec_lo, sec_hi);

            // Basic stats even without --deps
            std::unordered_map<std::string, int> cat_counts;
            int total_stall = 0;
            for (const auto& instr : k.instrs) {
                if (instr.addr < sec_lo || instr.addr > sec_hi) continue;
                cat_counts[instr.category]++;
                if (instr.stall() >= 0) total_stall += instr.stall();
            }

            printf("\nTotal stall cycles: %d\n", total_stall);
            printf("\nInstruction mix:\n");
            // Sort by count descending
            std::vector<std::pair<std::string, int>> sorted_cats(cat_counts.begin(), cat_counts.end());
            std::sort(sorted_cats.begin(), sorted_cats.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            for (const auto& [cat, cnt] : sorted_cats)
                printf("  %-14s %4d\n", cat.c_str(), cnt);
        }
    }

    return 0;
}
