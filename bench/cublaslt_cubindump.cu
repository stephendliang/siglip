/*
  Capture the module image(s) the driver actually loads for cuBLASLt's
  nvjet kernels, plus the launched kernel names, via CUPTI callbacks.

  Why: the sm_100 nvjet kernels have NO ELF symbols in any .so on disk
  (JIT / obfuscated fatbin — see tools/dump_cublaslt_sass.sh), and Modal
  has no ncu. CUPTI's RESOURCE_MODULE_LOADED callback hands us the exact
  image the driver receives — plain cubin (or PTX) by that point — with
  no perf counters involved, so it works on counter-locked shared GPUs.

  Writes cubin_<id>.bin into CWD, prints @@MODULE / @@LAUNCH / @@ALGO
  lines. A runner then cuobjdump --dump-sass --function's the cubin that
  contains the launched nvjet symbol.

  Usage: ./cublaslt-cubindump <M> <N> <K> <epi>   (PER_TENSOR only)
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cupti.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include "fc_problem.cuh"

#define CUPTI_CHECK(x) do {                                             \
    CUptiResult _r = (x);                                               \
    if (_r != CUPTI_SUCCESS) {                                          \
        const char* s = "?";                                            \
        cuptiGetResultString(_r, &s);                                   \
        std::fprintf(stderr, "CUPTI %s at %s:%d\n", s, __FILE__, __LINE__); \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA %s at %s:%d\n",                      \
                     cudaGetErrorString(_e), __FILE__, __LINE__);       \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

static void CUPTIAPI dump_cb(void*, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const void* cbdata) {
    if (domain == CUPTI_CB_DOMAIN_RESOURCE &&
        cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
        const CUpti_ResourceData* rd = (const CUpti_ResourceData*)cbdata;
        const CUpti_ModuleResourceData* md =
            (const CUpti_ModuleResourceData*)rd->resourceDescriptor;
        char fn[64];
        std::snprintf(fn, sizeof fn, "cubin_%u.bin", md->moduleId);
        if (FILE* f = std::fopen(fn, "wb")) {
            std::fwrite(md->pCubin, 1, md->cubinSize, f);
            std::fclose(f);
        }
        const unsigned char* p = (const unsigned char*)md->pCubin;
        std::printf("@@MODULE id=%u size=%zu magic=%02x%02x%02x%02x file=%s\n",
                    md->moduleId, (size_t)md->cubinSize,
                    p[0], p[1], p[2], p[3], fn);
        std::fflush(stdout);
    } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        const CUpti_CallbackData* cd = (const CUpti_CallbackData*)cbdata;
        if (cd->callbackSite == CUPTI_API_ENTER && cd->symbolName &&
            cd->symbolName[0]) {
            std::printf("@@LAUNCH sym=%s\n", cd->symbolName);
            std::fflush(stdout);
        }
    }
}

static int get_algo_int(const cublasLtMatmulAlgo_t* a,
                        cublasLtMatmulAlgoConfigAttributes_t at) {
    union { uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64; } b;
    size_t w = 0;
    for (size_t sz : {sizeof(uint32_t), sizeof(uint16_t), sizeof(uint64_t),
                      sizeof(uint8_t)}) {
        b.u64 = 0;
        if (cublasLtMatmulAlgoConfigGetAttribute(a, at, &b, sz, &w) ==
                CUBLAS_STATUS_SUCCESS && w > 0) {
            switch (w) {
                case 1: return (int)b.u8;
                case 2: return (int)b.u16;
                case 4: return (int)b.u32;
                case 8: return (int)b.u64;
            }
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 928256;
    int N = argc > 2 ? std::atoi(argv[2]) : 3072;
    int K = argc > 3 ? std::atoi(argv[3]) : 768;
    int epi_i = argc > 4 ? std::atoi(argv[4]) : 2;
    fc::Epilogue epi = (fc::Epilogue)epi_i;
    std::printf("@@CONFIG M=%d N=%d K=%d epi=%s\n", M, N, K, fc::epi_name(epi));

    CUpti_SubscriberHandle sub;
    CUPTI_CHECK(cuptiSubscribe(&sub, (CUpti_CallbackFunc)dump_cb, nullptr));
    CUPTI_CHECK(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE));
    const CUpti_CallbackId launch_ids[] = {
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    };
    for (CUpti_CallbackId id : launch_ids)
        CUPTI_CHECK(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, id));

    size_t szA = (size_t)K * N, szB = (size_t)K * M, szD = (size_t)N * M * 2;
    void *dA, *dB, *dD, *dBias = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dB, szB));
    CUDA_CHECK(cudaMalloc(&dD, szD));
    CUDA_CHECK(cudaMemset(dA, 0, szA));
    CUDA_CHECK(cudaMemset(dB, 0, szB));
    if (fc::epi_uses_bias(epi)) {
        CUDA_CHECK(cudaMalloc(&dBias, fc::bias_bytes(N)));
        CUDA_CHECK(cudaMemset(dBias, 0, fc::bias_bytes(N)));
    }

    cublasLtHandle_t lt;
    FC_CUBLAS_CHECK(cublasLtCreate(&lt));
    fc::Layouts lay = fc::make_layouts(M, N, K);
    fc::DescConfig cfg;
    cfg.epi = epi;
    cfg.d_bias = dBias;
    cublasLtMatmulDesc_t desc = fc::make_desc(cfg);

    void* ws;
    size_t ws_size = 256ull << 20;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));
    cublasLtMatmulPreference_t pref;
    FC_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    FC_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof ws_size));

    cublasLtMatmulHeuristicResult_t heur[8];
    int nres = 0;
    FC_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt, desc, lay.A, lay.B, lay.D, lay.D, pref, 8, heur, &nres));
    if (nres == 0) {
        std::fprintf(stderr, "no heuristic algos\n");
        return 1;
    }
    std::printf("@@ALGO id=%d tile=%d stages=%d cluster=%d nres=%d\n",
                get_algo_int(&heur[0].algo, CUBLASLT_ALGO_CONFIG_ID),
                get_algo_int(&heur[0].algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
                get_algo_int(&heur[0].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID),
                get_algo_int(&heur[0].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID),
                nres);

    float alpha = 1.f, beta = 0.f;
    for (int i = 0; i < 3; i++) {
        FC_CUBLAS_CHECK(cublasLtMatmul(lt, desc, &alpha, dA, lay.A, dB, lay.B,
                                       &beta, dD, lay.D, dD, lay.D,
                                       &heur[0].algo, ws, ws_size, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("@@DONE\n");
    return 0;
}
