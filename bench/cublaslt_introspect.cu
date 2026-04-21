/*
cuBLASLt algo-config introspection.

For a given (M, N, K, epilogue) shape, this enumerates every algo that
cublasLtMatmulAlgoGetHeuristic returns for per-tensor FP8, times each one,
and dumps all the CUBLASLT_ALGO_{CAP,CONFIG}_* attributes.  Also prints the
top-3 actually-used algos ranked by measured time (not heuristic).

Output is both human-readable and a TSV header+rows so the shell script can
grep/awk it.

Usage: ./cublaslt-introspect <M> <N> <K> <epi>
  epi: 0=none, 2=gelu_bias, 3=bias_only
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(x)   do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CUBLAS_CHECK(x) do { auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static constexpr size_t WS = 256ULL*1024*1024;
static constexpr int    N_TIME = 10;
static constexpr int    N_WARM = 3;

struct AlgoInfo {
    int heur_idx;
    float ms;
    int algo_id, tile_id, stages_id, cluster_id;
    int splitk, reduction, swizzle, custom;
    size_t ws_size;
    float waves;
};

static const char* epi_name(int e) {
    switch(e){case 0:return"none";case 2:return"gelu_bias";case 3:return"bias_only";}
    return "?";
}

static int get_algo_int(const cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoConfigAttributes_t at) {
    int v = -1; size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(a, at, &v, sizeof(v), &w) != CUBLAS_STATUS_SUCCESS)
        return -1;
    return v;
}

static void cap_dump(const cublasLtMatmulAlgo_t* a, const char* prefix) {
    int v=0; size_t w=0;
    #define CAP_INT(attr) do { \
        if (cublasLtMatmulAlgoCapGetAttribute(a, attr, &v, sizeof(v), &w)==CUBLAS_STATUS_SUCCESS) \
            printf("%s  %-44s = %d\n", prefix, #attr, v); \
    } while(0)
    CAP_INT(CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
    CAP_INT(CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    CAP_INT(CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT);
    CAP_INT(CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT);
    CAP_INT(CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
    CAP_INT(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK);
    CAP_INT(CUBLASLT_ALGO_CAP_EPILOGUE_MASK);

    std::vector<int> buf(128);
    #define CAP_LIST(attr) do { \
        if (cublasLtMatmulAlgoCapGetAttribute(a, attr, buf.data(), buf.size()*sizeof(int), &w)==CUBLAS_STATUS_SUCCESS) { \
            int n = (int)(w/sizeof(int)); \
            printf("%s  %-44s = [", prefix, #attr); \
            for (int i=0;i<n && i<24;i++) printf("%s%d", i?",":"", buf[i]); \
            if (n>24) printf(",...+%d", n-24); \
            printf("] (n=%d)\n", n); \
        } \
    } while(0)
    CAP_LIST(CUBLASLT_ALGO_CAP_TILE_IDS);
    CAP_LIST(CUBLASLT_ALGO_CAP_STAGES_IDS);
}

int main(int argc, char** argv) {
#ifdef DEFAULT_M
    int M   = (argc >= 2) ? atoi(argv[1]) : DEFAULT_M;
    int N   = (argc >= 3) ? atoi(argv[2]) : DEFAULT_N;
    int K   = (argc >= 4) ? atoi(argv[3]) : DEFAULT_K;
    int EPI = (argc >= 5) ? atoi(argv[4]) : DEFAULT_EPI;
#else
    if (argc < 5) { fprintf(stderr,"usage: %s <M> <N> <K> <epi:0|2|3>\n",argv[0]); return 1; }
    int M   = atoi(argv[1]);
    int N   = atoi(argv[2]);
    int K   = atoi(argv[3]);
    int EPI = atoi(argv[4]);
#endif

    cublasLtHandle_t lt; CUBLAS_CHECK(cublasLtCreate(&lt));

    // Buffers: A [K,M] col-major fp8, B [K,N] col-major fp8, D [M,N] bf16
    size_t szA = (size_t)K*M, szB = (size_t)K*N, szD = (size_t)M*N*2;
    void *dA,*dB,*dD,*dBias,*dSA,*dSB,*dWS;
    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dB, szB));
    CUDA_CHECK(cudaMalloc(&dD, szD));
    CUDA_CHECK(cudaMalloc(&dBias, N*2));
    CUDA_CHECK(cudaMalloc(&dSA, 4));
    CUDA_CHECK(cudaMalloc(&dSB, 4));
    CUDA_CHECK(cudaMalloc(&dWS, WS));
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpy(dSA, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSB, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dA,0,szA));
    CUDA_CHECK(cudaMemset(dB,0,szB));
    CUDA_CHECK(cudaMemset(dBias,0,N*2));

    cublasLtMatrixLayout_t layA,layB,layD;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layA, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layB, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layD, CUDA_R_16BF,    M, N, M));

    cublasLtMatmulDesc_t desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSA, sizeof(dSA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSB, sizeof(dSB)));
    if (EPI == 2 || EPI == 3) {
        cublasLtEpilogue_t e = (EPI==2) ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias)));
        cudaDataType_t btype = CUDA_R_16BF;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
    }

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));

    constexpr int MAX = 128;
    cublasLtMatmulHeuristicResult_t heur[MAX]; int n = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, desc, layA, layB, layD, layD, pref, MAX, heur, &n));

    printf("### CUBLASLT INTROSPECT M=%d N=%d K=%d epi=%s\n", M, N, K, epi_name(EPI));
    printf("### Heuristics returned: %d\n\n", n);

    float alpha=1.0f, beta=0.0f;
    cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    std::vector<AlgoInfo> infos;
    for (int i = 0; i < n; i++) {
        // warm + time
        for (int w=0;w<N_WARM;w++) {
            auto s = cublasLtMatmul(lt, desc, &alpha, dA, layA, dB, layB, &beta,
                                    dD, layD, dD, layD, &heur[i].algo, dWS, WS, 0);
            if (s != CUBLAS_STATUS_SUCCESS) { break; }
        }
        auto s = cublasLtMatmul(lt, desc, &alpha, dA, layA, dB, layB, &beta,
                                dD, layD, dD, layD, &heur[i].algo, dWS, WS, 0);
        if (s != CUBLAS_STATUS_SUCCESS) continue;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r=0;r<N_TIME;r++) {
            cublasLtMatmul(lt, desc, &alpha, dA, layA, dB, layB, &beta,
                           dD, layD, dD, layD, &heur[i].algo, dWS, WS, 0);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= N_TIME;

        AlgoInfo a{};
        a.heur_idx = i;
        a.ms       = ms;
        a.ws_size  = heur[i].workspaceSize;
        a.waves    = heur[i].wavesCount;
        a.algo_id    = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_ID);
        a.tile_id    = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
        a.stages_id  = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
        a.cluster_id = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
        a.splitk     = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        a.reduction  = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
        a.swizzle    = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
        a.custom     = get_algo_int(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
        infos.push_back(a);
    }

    std::sort(infos.begin(), infos.end(), [](const AlgoInfo& a, const AlgoInfo& b){ return a.ms < b.ms; });

    // TSV for easy awk'ing
    printf("# TSV\n");
    printf("rank\theur\tms\talgoId\ttile\tstages\tcluster\tsplitk\tredux\tswizzle\tcustom\twaves\tws_MB\n");
    for (size_t r = 0; r < infos.size(); r++) {
        const auto& a = infos[r];
        printf("%zu\t%d\t%.4f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%.1f\n",
               r+1, a.heur_idx, a.ms, a.algo_id, a.tile_id, a.stages_id,
               a.cluster_id, a.splitk, a.reduction,
               a.swizzle, a.custom, a.waves, a.ws_size / (1024.0*1024.0));
    }

    printf("\n# Top-3 caps:\n");
    int top = std::min<int>(3, (int)infos.size());
    for (int r = 0; r < top; r++) {
        const auto& a = infos[r];
        printf("\n## rank=%d  algoId=%d  tile=%d  stages=%d  cluster=%d  ms=%.4f\n",
               r+1, a.algo_id, a.tile_id, a.stages_id, a.cluster_id, a.ms);
        cap_dump(&heur[a.heur_idx].algo, "  ");
    }

    if (infos.empty()) {
        printf("\n@@RESULT ms=ERR tflops=0.00 checksum=0.000000 valid=0 c0=0.0\n");
        return 1;
    }
    printf("\n# Winner:  rank=1  tile=%d  stages=%d  cluster=%d  splitk=%d  swizzle=%d  ms=%.4f\n",
           infos[0].tile_id, infos[0].stages_id, infos[0].cluster_id,
           infos[0].splitk, infos[0].swizzle, infos[0].ms);
    double tflops = 2.0 * (double)M * (double)N * (double)K / ((double)infos[0].ms * 1e9);
    printf("\n@@RESULT ms=%.4f tflops=%.2f checksum=0.000000 valid=1 c0=0.0\n",
           infos[0].ms, tflops);
    return 0;
}
