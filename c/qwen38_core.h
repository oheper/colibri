#include "qwen36_tier.h"   /* CUDA VRAM expert tier, shared with qwen36; CPU-only builds get the inline no-op stubs */
/* Native Qwen3.8-Flash-Next text core.
 *
 * This header is included once by qwen38.c after its tokenizer and protocol
 * helpers.  It intentionally consumes the official HF safetensors layout:
 * the multimodal wrapper's `model.language_model` namespace and the standalone
 * text model's `model` namespace are both accepted.  Vision and MTP tensors are
 * indexed by st.h but are never read.
 */
#ifndef COLI_QWEN38_CORE_H
#define COLI_QWEN38_CORE_H
#include "kv_prefix.h"
#include "expert_ffn.h"   /* BABA_V7 packed q4 routed experts */
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>   /* q38_ehit_mark publishes the lazy HITS table under a lock */

#define Q38_MAX_LAYERS 512
#define Q38_MAX_EXPERTS 1024
#define Q38_MAX_TOPK 256
#define Q38_MAX_PLE_PARTS 512
#define Q38_PREFILL_BATCH_ROWS 32
#define Q38_PREFILL_WORKSPACE_BYTES (64u << 20)

typedef struct {
    int hidden, layers, vocab, max_positions, eos_id;
    float eps, theta;
    int hc_count, hc_rank, hc_width;
    int q_heads, kv_heads, head_dim, rotary_dim;
    int idx_qheads, idx_kheads, idx_dim, idx_budget, idx_ratio;
    int experts, topk, inter, shared_inter, norm_topk;
    int dn_kheads, dn_vheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int ple_layer, ple_dim, ple_convk, ngram_size, heads_per_ngram;
    int ngram_heads, ngram_head_dim, ngram_parts;
    uint8_t *is_attn;
    /* vision: 0 = checkpoint di solo testo, o torre non caricata */
    int image_token;
    int vis_depth, vis_hidden, vis_heads, vis_inter, vis_patch;
    int vis_merge, vis_temporal, vis_in_ch, vis_out_hidden, vis_num_pos;
} Cfg;

typedef enum {
    Q38_WEIGHT_NONE = 0,
    Q38_WEIGHT_F32,
    Q38_WEIGHT_BF16,
    Q38_WEIGHT_FP8
} Q38WeightKind;

typedef struct {
    void *data;
    float *scales;                 /* block-FP8 only */
    int rows, cols;
    int64_t elements, scale_count;
    Q38WeightKind kind;
    unsigned owns_data:1, owns_scales:1;
    int gpu;                       /* 0 = CPU; else 1 + tier handle of an int8 copy resident in VRAM (decode, S == 1) */
    int8_t *q8; float *q8sc;       /* the trunk's int8 rows on the CPU (default; Q38_TRUNK_CPU_INT8=0 keeps BF16): the same rows the GPU holds, met by an int8 activation in idot.h */
} Q38Weight;

typedef struct { float *norm; Q38Weight down, up, inject; } GatedResidual;

/* Persistent wall-clock counters.  They live on Model rather than in process
 * globals so Segment sessions and future multi-model serving cannot leak phase
 * time into one another.  Some categories intentionally overlap: architecture
 * phases (DeltaNet/QSA/PLE) contain resident matmuls, while the matmul counter
 * answers the orthogonal question "how much time is in dense kernels?". */
typedef enum {
    Q38_TM_EXPERT_READ = 0,
    Q38_TM_FP8_EXPAND,
    Q38_TM_ROUTED_EXPERT,
    Q38_TM_SHARED_EXPERT,
    Q38_TM_DENSE_MATMUL,
    Q38_TM_DELTANET,
    Q38_TM_QSA_INDEX,
    Q38_TM_QSA_ATTENTION,
    Q38_TM_PLE,
    Q38_TM_LM_HEAD,
    Q38_TM_COUNT
} Q38Timer;

typedef struct {
    double seconds[Q38_TM_COUNT];
    uint64_t forwards;
} Q38Timers;

typedef struct {
    GatedResidual attn_gr, mlp_gr;
    Q38Weight router, sh_g, sh_u, sh_d;
    float *sh_gate;
    Q38Weight q, k, v, o;
    float *qn, *kn;
    Q38Weight idx_qk;
    float *idx_qn, *idx_kn;
    Q38Weight dn_qkv, dn_z, dn_b, dn_a, dn_out;
    float *dn_conv;
    float *dn_dtbias, *dn_alog, *dn_norm;
    Q38Weight ple_key, ple_value;
    float *ple_norm_key, *ple_norm_query;
    float *ple_norm_conv, *ple_conv;
} Layer;

typedef struct {
    int eid;
    Q38Weight gate, up, down;
    uint64_t used;
    void *fp8_slab;
    int64_t fp8_slab_bytes;
    void *q4_slab;
    int64_t q4_slab_bytes;
    XfExpert q4;
    unsigned q4_active:1;
} Slot;
/* BABA_CPU_HOTPATH_V5
 * freq[eid] survives eviction. With Q38_CACHE_POLICY=lfu this is an
 * aging LFU/TinyLFU-ish victim signal; default/lru preserves old behavior. */
typedef struct {
    Slot *slots;
    int *by_expert, n, cap;
    uint16_t *freq;
    uint32_t freq_touches;
} LCache;

typedef struct {
    float *values;                 /* [expert][gate,up,down][block] */
    int64_t scale_count;
    int ready;                     /* 0 unknown, 1 resident, -1 incompatible */
} Q38ExpertScaleCache;

typedef enum {
    Q38_EXPERT_BATCH_FALLBACK_NONE = 0,
    Q38_EXPERT_BATCH_FALLBACK_DISABLED,
    Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY,
    Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK,
    Q38_EXPERT_BATCH_FALLBACK_DUPLICATE,
    Q38_EXPERT_BATCH_FALLBACK_LAYOUT,
} Q38ExpertBatchFallback;

typedef struct {
    Cfg c;
    shards S;
    char prefix[32];
    Q38Weight embed, lm_head;
    GatedResidual final_gr;
    Layer *L;
    LCache *cache;
    uint8_t **ehit;                    /* experts routed this turn, for HITS (dashboard Brain) */
    Q38ExpertScaleCache *expert_scales;
    uint64_t clock, hits, miss;
    uint64_t expert_weight_reads, expert_scale_reads, expert_pair_reads;
    uint64_t q4_cache_hits, q4_cache_misses;
    uint64_t expert_prefetch_ranges, expert_parallel_batches;
    uint64_t expert_scale_bytes;
    float **DN_rec, **DN_conv;
    float **K, **V, **IK;
    int kv_len, kv_cap, max_t;
    kv_prefix kvp; /* token identity of the live attention rows, not a snapshot */
    st_tensor *ple_parts[Q38_MAX_PLE_PARTS];
    char ple_part_names[Q38_MAX_PLE_PARTS][320];
    int64_t ple_part_start[Q38_MAX_PLE_PARTS + 1];
    int ple_part_count;
    float ple_weight_scale;
    int64_t ple_multipliers[3], ple_head_vocab[64], ple_head_offset[64];
    int64_t *ple_history;
    float *PLE_conv_state;
    int ple_history_len;
    int range_begin, range_end;
    int native_fp8, native_bf16, expert_prefetch, expert_parallel_reads;
    Q38ExpertBatchFallback expert_batch_fallback;
    int prefill_batch;
    uint64_t resident_weight_bytes;
    int trunk_table_built;         /* q38_trunk_offer_all ran for this load (the table is process-wide, the model is not) */
    double dense_load_s;
    /* vision. `vis_map` mappa la posizione ASSOLUTA nella sequenza alla riga di
     * `vis_rows`, oppure -1. Assoluta e non relativa al chunk: il prefill arriva
     * a pezzi, e un indice relativo darebbe l'immagine sbagliata al secondo
     * pezzo senza che niente protesti. */
    /* PLE prefetto: le righe gia' lette per il chunk in corso, o NULL. */
    float *ple_pref; int ple_pref_rows;
    Q38Vision vis;
    int vis_ready;
    float *vis_rows;
    int *vis_map, vis_map_len, vis_rows_n;
    Q38Timers timers;
} Model;

static float *g_last_logit;
static int g_capture_last_logit;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static inline void q38_tm_add(Model *m,Q38Timer timer,double started) {
    m->timers.seconds[timer] += now_s() - started;
}

static Q38Timers q38_tm_delta(const Q38Timers *after,const Q38Timers *before) {
    Q38Timers delta={0};
    for(int i=0;i<Q38_TM_COUNT;i++)
        delta.seconds[i]=after->seconds[i]-before->seconds[i];
    delta.forwards=after->forwards-before->forwards;
    return delta;
}

#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1073741824.0; }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576.0; }
#endif

static float *falloc(int64_t n) {
    if (n < 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "invalid float allocation: %lld\n", (long long)n); exit(1);
    }
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p && n) { fprintf(stderr, "OOM allocating %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* W is row-major [O,I], y=x@W^T. */
static void q38_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * w[i];
            y[(int64_t)s * O + o] = a;
        }
    }
}

static void q38_weight_free(Q38Weight *weight) {
    if(!weight)return;
    if(weight->owns_data)free(weight->data);
    if(weight->owns_scales)free(weight->scales);
    free(weight->q8); free(weight->q8sc);
    memset(weight,0,sizeof(*weight));
}

static void q38_weight_reserve(Q38Weight *weight,Q38WeightKind kind,int rows,int cols) {
    if(rows<=0||cols<=0||(int64_t)rows>INT64_MAX/cols){
        fprintf(stderr,"invalid weight geometry [%d,%d]\n",rows,cols);exit(1);
    }
    int64_t elements=(int64_t)rows*cols;
    int64_t scales=kind==Q38_WEIGHT_FP8?fp8_nblk(rows)*fp8_nblk(cols):0;
    if(weight->kind==kind&&weight->rows==rows&&weight->cols==cols&&
       weight->owns_data&&weight->data&&
       (!scales||(weight->owns_scales&&weight->scales)))return;
    q38_weight_free(weight);
    size_t element_size=kind==Q38_WEIGHT_F32?sizeof(float):
                        kind==Q38_WEIGHT_BF16?sizeof(uint16_t):
                        kind==Q38_WEIGHT_FP8?sizeof(uint8_t):0;
    if(!element_size||(uint64_t)elements>SIZE_MAX/element_size||
       (scales&&(uint64_t)scales>SIZE_MAX/sizeof(float))){
        fprintf(stderr,"unsupported or oversized weight geometry [%d,%d] kind=%d\n",
                rows,cols,(int)kind);exit(1);
    }
    weight->data=malloc((size_t)elements*element_size);
    if(scales)weight->scales=(float*)malloc((size_t)scales*sizeof(float));
    if(!weight->data||(scales&&!weight->scales)){
        fprintf(stderr,"OOM allocating weight [%d,%d] kind=%d\n",rows,cols,(int)kind);exit(1);
    }
    weight->rows=rows;weight->cols=cols;weight->elements=elements;
    weight->scale_count=scales;weight->kind=kind;
    weight->owns_data=1;weight->owns_scales=scales?1u:0u;
}

static uint64_t q38_weight_bytes(const Q38Weight *weight) {
    uint64_t element_size=weight->kind==Q38_WEIGHT_F32?sizeof(float):
                          weight->kind==Q38_WEIGHT_BF16?sizeof(uint16_t):
                          weight->kind==Q38_WEIGHT_FP8?sizeof(uint8_t):0;
    return (uint64_t)weight->elements*element_size+
           (uint64_t)weight->scale_count*sizeof(float);
}

/* Native BF16 storage with FP32 activations and accumulation.  This deliberately
 * does not round activations to BF16 or use BF16 dot-product instructions: it is
 * the storage-equivalent form of the existing st_read_f32 reference. */
static void q38_matmul_bf16(float *y,const float *x,const uint16_t *W,
                            int S,int I,int O) {
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;float a=0.f;
            for(int i=0;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* The routed experts as the checkpoint ships them: e4m3 bytes with one f32
 * scale per 128x128 block. quant.h's matmul_fp8 decodes every byte through a
 * 256-entry table, one gather per weight; this kernel decodes eight bytes at a
 * time in registers (quant.h e4m3_decode8: a shift and one multiply, NaNs
 * kept) and multiplies them with FMA. The block scale still applies once per
 * block and the blocks still add in double, so the result differs from the
 * scalar kernel only by the float summation order inside a block. For a
 * batch of rows (prefill) the block is decoded once and held while every row
 * runs through it: the matrix streams past once, not once per row.
 * Q38_FP8_KERNEL=scalar restores the table kernel (bisecting a difference). */
static int q38_fp8_vector_on(void) {
    static int v=-1;
    if(v<0){ const char *e=getenv("Q38_FP8_KERNEL"); v=!(e&&!strcmp(e,"scalar")); }
    return v;
}
#ifdef __AVX2__
#define Q38_FP8_ROWS 8
static void q38_matmul_fp8_vec(float *y,const float *x,const uint8_t *q8,
                               const float *bscale,int S,int I,int O) {
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q8+(int64_t)o*I;
        const float *scl=bscale+((int64_t)o/FP8_BLOCK)*nblkI;
        if(S==1){
            /* decode: the weight bytes are the traffic, so decode and multiply in one pass */
            double a=0;
            for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
                int base=(int)(bi*FP8_BLOCK),blen=I-base<FP8_BLOCK?I-base:FP8_BLOCK,i=0;
                __m256 acc=_mm256_setzero_ps();
                for(;i+8<=blen;i+=8)
                    acc=_mm256_fmadd_ps(e4m3_decode8(w+base+i),_mm256_loadu_ps(x+base+i),acc);
                float part=hsum256(acc);
                for(;i<blen;i++)part+=e4m3_decode(w[base+i])*x[base+i];
                a+=(double)part*scl[bi];
            }
            y[o]=(float)a; continue;
        }
        for(int s0=0;s0<S;s0+=Q38_FP8_ROWS){
            int ns=S-s0<Q38_FP8_ROWS?S-s0:Q38_FP8_ROWS; double a[Q38_FP8_ROWS]={0};
            for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
                int base=(int)(bi*FP8_BLOCK),blen=I-base<FP8_BLOCK?I-base:FP8_BLOCK,i=0;
                float wf[FP8_BLOCK];
                for(;i+8<=blen;i+=8)_mm256_storeu_ps(wf+i,e4m3_decode8(w+base+i));
                for(;i<blen;i++)wf[i]=e4m3_decode(w[base+i]);
                float sc=scl[bi];
                for(int r=0;r<ns;r++){
                    const float *xs=x+(int64_t)(s0+r)*I+base; int k=0;
                    __m256 acc=_mm256_setzero_ps();
                    for(;k+8<=blen;k+=8)
                        acc=_mm256_fmadd_ps(_mm256_loadu_ps(wf+k),_mm256_loadu_ps(xs+k),acc);
                    float part=hsum256(acc);
                    for(;k<blen;k++)part+=wf[k]*xs[k];
                    a[r]+=(double)part*sc;
                }
            }
            for(int r=0;r<ns;r++)y[(int64_t)(s0+r)*O+o]=(float)a[r];
        }
    }
}
#endif
static void q38_matmul_fp8(float *y,const float *x,const uint8_t *q8,const float *bscale,
                           int S,int I,int O) {
#ifdef __AVX2__
    if(q38_fp8_vector_on()){ q38_matmul_fp8_vec(y,x,q8,bscale,S,I,O); return; }
#endif
    matmul_fp8(y,x,q8,bscale,S,I,O);
}

/* BABA_CPU_HOTPATH_V3_REPAIR
 * Local tree already has same-input multi-GEMV. This patch only adds:
 *   - reusable q8 activation scratch in q38_weight_matmul
 *   - GR decode workspace
 *   - MoE decode workspace
 *   - QSA per-head workspace
 * Q38_CPU_V3=1 enables these additions. */
static inline int q38_cpu_v3(void) {
    const char *e=getenv("Q38_CPU_V3");
    return e&&e[0]=='1'&&!e[1];
}
typedef struct { float *p; size_t cap; } Q38TlsF32;
static _Thread_local Q38TlsF32 g_q38_gr_ws, g_q38_moe_ws;
static _Thread_local Q38TlsF32 g_q38_qsa_qh_ws, g_q38_qsa_score_ws;
static inline float *q38_tls_f32(Q38TlsF32 *w,size_t n) {
    if(n>w->cap){
        float *p=(float*)realloc(w->p,n*sizeof(float));
        if(!p){fprintf(stderr,"OOM qwen38 TLS workspace\n");exit(1);}
        w->p=p;w->cap=n;
    }
    return w->p;
}
/* BABA_CPU_HOTPATH_V4
 * Big-iron decode experiments:
 * Q38_DN_AVX2=1          vectorize DeltaNet recurrence across value lanes
 * Q38_Q4_ASYNC_SHARED=1  overlap missing-expert disk I/O with shared-expert CPU work
 * Q38_Q4_TLS_SCRATCH=1   reuse xf_moe_run scratch instead of malloc/free per layer
 * Q38_Q4_PROFILE=1       print aggregate Q4 async overlap counters at shutdown/report */
static inline int q38_v4_on(const char *name){
    const char *e=getenv(name);return e&&e[0]=='1'&&!e[1];
}
static _Thread_local unsigned char *g_q38_xf_scratch;
static _Thread_local size_t g_q38_xf_scratch_cap;
static inline void *q38_xf_scratch_get(size_t n){
    if(n>g_q38_xf_scratch_cap){
        void *p=realloc(g_q38_xf_scratch,n);
        if(!p){fprintf(stderr,"OOM q4 persistent scratch\n");exit(1);}
        g_q38_xf_scratch=(unsigned char*)p;g_q38_xf_scratch_cap=n;
    }
    return g_q38_xf_scratch;
}
typedef struct {
    Model *m; int layer; const int *experts; int count; Slot **selected; int result;
} Q38AsyncExpertLoad;
static void *q38_async_expert_load_main(void *arg);

static double g_q38_v4_async_wall_s;
static double g_q38_v4_async_hidden_s;
static uint64_t g_q38_v4_async_batches,g_q38_v4_async_jobs;
static void q38_v4_report(void);

#ifdef __AVX2__
static inline void q38_dn_head_avx2(float *state,const float *qh,const float *kh,
                                    const float *vh,float *delta,float *out,
                                    int KD,int VD,float alpha,float beta){
    int64_t cells=(int64_t)KD*VD;
    __m256 va=_mm256_set1_ps(alpha);
    int64_t cell=0;
    for(;cell+8<=cells;cell+=8)
        _mm256_storeu_ps(state+cell,_mm256_mul_ps(_mm256_loadu_ps(state+cell),va));
    for(;cell<cells;cell++)state[cell]*=alpha;

    int v=0;
    __m256 vb=_mm256_set1_ps(beta);
    for(;v+8<=VD;v+=8){
        __m256 prev=_mm256_setzero_ps();
        for(int d=0;d<KD;d++){
            __m256 kk=_mm256_set1_ps(kh[d]);
            prev=_mm256_fmadd_ps(kk,_mm256_loadu_ps(state+(int64_t)d*VD+v),prev);
        }
        __m256 de=_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vh+v),prev),vb);
        _mm256_storeu_ps(delta+v,de);
    }
    for(;v<VD;v++){
        float prev=0.f;
        for(int d=0;d<KD;d++)prev+=kh[d]*state[(int64_t)d*VD+v];
        delta[v]=(vh[v]-prev)*beta;
    }

    for(int d=0;d<KD;d++){
        __m256 kk=_mm256_set1_ps(kh[d]); int j=0;
        float *st=state+(int64_t)d*VD;
        for(;j+8<=VD;j+=8)
            _mm256_storeu_ps(st+j,_mm256_fmadd_ps(kk,_mm256_loadu_ps(delta+j),_mm256_loadu_ps(st+j)));
        for(;j<VD;j++)st[j]+=kh[d]*delta[j];
    }

    v=0;
    for(;v+8<=VD;v+=8){
        __m256 cur=_mm256_setzero_ps();
        for(int d=0;d<KD;d++){
            __m256 qq=_mm256_set1_ps(qh[d]);
            cur=_mm256_fmadd_ps(qq,_mm256_loadu_ps(state+(int64_t)d*VD+v),cur);
        }
        _mm256_storeu_ps(out+v,cur);
    }
    for(;v<VD;v++){
        float cur=0.f;
        for(int d=0;d<KD;d++)cur+=qh[d]*state[(int64_t)d*VD+v];
        out[v]=cur;
    }
}
#endif
static void q38_weight_matmul(float *y,const float *x,const Q38Weight *weight,
                              int S,int I,int O) {
    /* A matrix the tier placed in VRAM (q38_trunk_place) answers a decode
     * GEMV from there; prefill rows and any failure take the CPU path below,
     * so the BF16 copy stays the reference for everything but S == 1. */
    if(S==1&&weight&&weight->gpu&&weight->rows==O&&weight->cols==I&&
       qt_dense_matmul(weight->gpu-1,y,x,I,O))return;
    if(weight&&weight->q8&&weight->rows==O&&weight->cols==I){
        /* the trunk's int8 rows (the same the GPU holds) meet an int8
         * activation in the integer kernel: x quantized once per row with one
         * scale, then maddubs / vpdpbusd dot products (idot.h). Decode and
         * prefill take the same path, so GPU or not the trunk quantization is
         * the only thing that separates the output from the BF16 run. */
        int8_t *xq=NULL; float *sx=NULL;
        if(q38_cpu_v3()) quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        else {
            xq=(int8_t*)malloc((size_t)S*I);
            sx=(float*)malloc((size_t)S*sizeof(float));
            if(!xq||!sx){fprintf(stderr,"OOM activation quantization\n");exit(1);}
        }
        for(int s=0;s<S;s++)sx[s]=dense_act_i8(x+(int64_t)s*I,I,xq+(int64_t)s*I,NULL);
        matmul_q_idot(y,xq,sx,weight->q8,weight->q8sc,S,I,O);
        if(!q38_cpu_v3()){free(xq);free(sx);}
        return;
    }
    if(!weight||weight->rows!=O||weight->cols!=I||!weight->data){
        fprintf(stderr,"invalid matmul weight: have [%d,%d] kind=%d, need [%d,%d]\n",
                weight?weight->rows:0,weight?weight->cols:0,
                weight?(int)weight->kind:0,O,I);exit(1);
    }
    if(weight->kind==Q38_WEIGHT_F32)
        q38_matmul(y,x,(const float*)weight->data,S,I,O);
    else if(weight->kind==Q38_WEIGHT_BF16)
        q38_matmul_bf16(y,x,(const uint16_t*)weight->data,S,I,O);
    else if(weight->kind==Q38_WEIGHT_FP8&&weight->scales)
        q38_matmul_fp8(y,x,(const uint8_t*)weight->data,weight->scales,S,I,O);
    else {fprintf(stderr,"unsupported matmul weight kind %d\n",(int)weight->kind);exit(1);}
}

static void q38_weight_row(const Q38Weight *weight,int row,float *out) {
    if(!weight||row<0||row>=weight->rows||!weight->data){
        fprintf(stderr,"invalid weight row %d\n",row);exit(1);
    }
    if(weight->kind==Q38_WEIGHT_F32)
        memcpy(out,(const float*)weight->data+(int64_t)row*weight->cols,
               (size_t)weight->cols*sizeof(float));
    else if(weight->kind==Q38_WEIGHT_BF16){
        const uint16_t *src=(const uint16_t*)weight->data+(int64_t)row*weight->cols;
        for(int i=0;i<weight->cols;i++)out[i]=bf16_to_f32(src[i]);
    } else {fprintf(stderr,"weight kind %d has no dense row view\n",(int)weight->kind);exit(1);}
}

static void q38_dense_matmul(Model *m,float *y,const float *x,const Q38Weight *weight,
                             int S,int I,int O) {
    double started=now_s();
    q38_weight_matmul(y,x,weight,S,I,O);
    q38_tm_add(m,Q38_TM_DENSE_MATMUL,started);
}

/* BABA_Q38_CUDA_PIPE_CORE: decode-only resident projection fan-out.
 * No prefill semantics change. The CPU path remains the exact fallback. */
static int q38_cuda_pipe_on(void){
    const char *e=getenv("Q38_CUDA_PIPE");
    return e&&e[0]=='1'&&!e[1];
}
static int q38_cuda_pipe_shared_on(void){
    const char *e=getenv("Q38_CUDA_PIPE_SHARED");
    return q38_cuda_pipe_on() && !(e&&e[0]=='0'&&!e[1]);
}
/* BABA_V7_MEGA_TOURNAMENT
 * Decode-only same-X projection fan-out.
 * CUDA keeps its existing path. On CPU, one float activation is quantized once
 * and reused by all compatible resident q8 matrices. */
static int q38_mega_cpu_multi_on(void){
    const char *e=getenv("Q38_MEGA_CPU_MULTI");
    return e&&e[0]=='1'&&!e[1];
}
static int q38_dense_multi_same_x(Model *m,const float *x,Q38Weight **weights,
                                  float **ys,const int *Os,int count,int I){
    int h[7];
    if(count<2||count>7)return 0;

    if(q38_cuda_pipe_on()){
        for(int i=0;i<count;i++){
            if(!weights[i]||!weights[i]->gpu||weights[i]->cols!=I||weights[i]->rows!=Os[i])
                return 0;
            h[i]=weights[i]->gpu-1;
        }
        double started=now_s();
        int ok=qt_dense_multi_same_x(h,ys,Os,count,x,I);
        if(ok)q38_tm_add(m,Q38_TM_DENSE_MATMUL,started);
        return ok;
    }

    if(!q38_mega_cpu_multi_on())return 0;
    for(int i=0;i<count;i++)
        if(!weights[i]||!weights[i]->q8||!weights[i]->q8sc||
           weights[i]->cols!=I||weights[i]->rows!=Os[i])
            return 0;

    int8_t *xq=NULL; float *sx=NULL;
    if(q38_cpu_v3()) quant_scratch((size_t)I,1,&xq,&sx);
    else {
        xq=(int8_t*)malloc((size_t)I);
        sx=(float*)malloc(sizeof(float));
        if(!xq||!sx){fprintf(stderr,"OOM mega same-X quant scratch\n");exit(1);}
    }

    double started=now_s();
    sx[0]=dense_act_i8(x,I,xq,NULL);
    for(int i=0;i<count;i++)
        matmul_q_idot(ys[i],xq,sx,weights[i]->q8,weights[i]->q8sc,1,I,Os[i]);
    q38_tm_add(m,Q38_TM_DENSE_MATMUL,started);

    if(!q38_cpu_v3()){free(xq);free(sx);}
    return 1;
}
static int q38_shared_mlp_cuda(float *y,const float *x,Q38Weight *g,Q38Weight *u,Q38Weight *d){
    if(!q38_cuda_pipe_shared_on()||!g||!u||!d||!g->gpu||!u->gpu||!d->gpu)return 0;
    return qt_dense_mlp3(g->gpu-1,u->gpu-1,d->gpu-1,y,x,1);
}

static inline float q38_sigmoid(float x) {
    if (x >= 0.f) { float z=expf(-x); return 1.f/(1.f+z); }
    float z=expf(x); return z/(1.f+z);
}
static inline float q38_silu(float x) { return x * q38_sigmoid(x); }
static inline float q38_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

/* Qwen4-Exp RMSNorms are zero-centered: the learned scale is 1+weight. */
static void q38_rms0(float *out, const float *x, const float *w, int n, float eps) {
    double ss=0.0; for (int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for (int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}

/* DeltaNet's RMSNormGated is inherited from Qwen3-Next and is not zero-centered. */
static void q38_rmsg(float *out,const float *x,const float *gate,const float *w,
                     int n,float eps,int sigmoid_gate) {
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*w[i]*(sigmoid_gate?q38_sigmoid(gate[i]):q38_silu(gate[i]));
}

static void q38_rope(float *x, int dim, int rotary_dim, int pos, float theta) {
    int half=rotary_dim/2;
    for(int i=0;i<half;i++) {
        float ang=(float)pos/powf(theta,(float)(2*i)/rotary_dim);
        float co=cosf(ang), si=sinf(ang), a=x[i], b=x[i+half];
        x[i]=a*co-b*si; x[i+half]=b*co+a*si;
    }
    (void)dim;
}

static jval *q38_obj(jval *o,const char *key) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_OBJ){fprintf(stderr,"config.json: %s must be an object\n",key);exit(1);}
    return v;
}
static double q38_num(jval *o,const char *key,double def,int required) {
    jval *v=json_get(o,key);
    if(v&&v->t==J_NUM) return v->num;
    if(v){fprintf(stderr,"config.json: %s must be numeric\n",key);exit(1);}
    if(required){fprintf(stderr,"config.json: missing numeric %s\n",key);exit(1);} return def;
}
/* JSON numbers are doubles in the small parser.  Never narrow an unchecked
 * value to an int: besides accepting fractional dimensions, a huge finite
 * value can become implementation-defined before q38_validate_cfg sees it. */
static int q38_num_int(jval *o,const char *key,double def,int required,
                       int min_value,int max_value) {
    double value=q38_num(o,key,def,required);
    if(!isfinite(value)||floor(value)!=value||value<(double)min_value||
       value>(double)max_value){
        fprintf(stderr,"config.json: %s must be an integer in [%d,%d]\n",
                key,min_value,max_value);exit(1);
    }
    return (int)value;
}
static int q38_derived_product(const char *name,int left,int right) {
    if(left<0||right<0||(uint64_t)left*(uint64_t)right>(uint64_t)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left*right;
}
static int q38_derived_sum(const char *name,int left,int right) {
    if(left<0||right<0||left>INT_MAX-right){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left+right;
}
static int q38_bool(jval *o,const char *key,int def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_BOOL){fprintf(stderr,"config.json: %s must be boolean\n",key);exit(1);}
    return v?v->boolean:def;
}
static const char *q38_string(jval *o,const char *key,const char *def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_STR){fprintf(stderr,"config.json: %s must be a string\n",key);exit(1);}
    return v?v->str:def;
}
static void q38_require_string(jval *o,const char *key,const char *expected) {
    const char *value=q38_string(o,key,expected);
    if(strcmp(value,expected)){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",
                key,value,expected);exit(1);
    }
}
static void q38_require_present_string(jval *o,const char *key,
                                        const char *expected) {
    jval *value=json_get(o,key);
    if(!value||value->t!=J_STR||strcmp(value->str,expected)){
        fprintf(stderr,"config.json: %s must be explicitly set to %s\n",
                key,expected);exit(1);
    }
}
static void q38_require_bool(jval *o,const char *key,int expected) {
    int value=q38_bool(o,key,expected);
    if(value!=expected){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",key,
                value?"true":"false",expected?"true":"false");exit(1);
    }
}

static void q38_load_cfg(Cfg *c,const char *snap) {
    memset(c,0,sizeof(*c));
    char path[2048]; snprintf(path,sizeof path,"%s/config.json",snap);
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END);
    long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0||n>(256L<<20)){fprintf(stderr,"invalid config size\n");exit(1);}
    char *buf=(char*)malloc((size_t)n+1),*arena=NULL;
    if(!buf||fread(buf,1,(size_t)n,f)!=(size_t)n){fprintf(stderr,"cannot read %s\n",path);exit(1);}
    fclose(f); buf[n]=0; jval *root=json_parse(buf,&arena);
    if(!root||root->t!=J_OBJ){fprintf(stderr,"invalid %s\n",path);exit(1);}
    jval *tc=q38_obj(root,"text_config"); if(!tc) tc=root;
    const char *mt=jstr(tc,"model_type");
    if(!mt||strcmp(mt,"qwen4_exp_text")){fprintf(stderr,"unsupported text model_type: %s\n",mt?mt:"(missing)");exit(1);}
    q38_require_string(tc,"hidden_act","silu");
    /* Upstream's omitted/None default resolves to hidden_act (SiLU), whereas
     * this implementation deliberately uses the released sigmoid gate. */
    q38_require_present_string(tc,"output_gate_type","sigmoid");
    q38_require_bool(tc,"attention_bias",0);
    q38_require_bool(tc,"tie_word_embeddings",0);
    c->hidden=q38_num_int(tc,"hidden_size",0,1,0,INT_MAX);
    c->layers=q38_num_int(tc,"num_hidden_layers",0,1,0,Q38_MAX_LAYERS);
    c->vocab=q38_num_int(tc,"vocab_size",0,1,0,INT_MAX);
    c->max_positions=q38_num_int(tc,"max_position_embeddings",0,1,0,INT_MAX);
    c->eos_id=q38_num_int(tc,"eos_token_id",-1,1,INT_MIN,INT_MAX);
    double eps=q38_num(tc,"rms_norm_eps",1e-6,0);
    if(!isfinite(eps)||eps<=0.0||eps>FLT_MAX){fprintf(stderr,"invalid rms_norm_eps\n");exit(1);}
    c->eps=(float)eps;
    jval *rp=q38_obj(tc,"rope_parameters");
    if(rp)q38_require_string(rp,"rope_type","default");
    double theta=rp?q38_num(rp,"rope_theta",10000,0):q38_num(tc,"rope_theta",10000,0);
    if(!isfinite(theta)||theta<=0.0||theta>FLT_MAX){fprintf(stderr,"invalid rope_theta\n");exit(1);}
    c->theta=(float)theta;
    c->hc_count=q38_num_int(tc,"hc_count",4,0,0,INT_MAX);
    c->hc_rank=q38_num_int(tc,"hc_lowrank",320,0,0,INT_MAX);
    c->hc_width=q38_derived_product("hc_width",c->hc_count,c->hidden);
    c->q_heads=q38_num_int(tc,"num_attention_heads",0,1,0,INT_MAX);
    c->kv_heads=q38_num_int(tc,"num_key_value_heads",0,1,0,INT_MAX);
    c->head_dim=q38_num_int(tc,"head_dim",0,1,0,INT_MAX);
    double partial=rp?q38_num(rp,"partial_rotary_factor",q38_num(tc,"partial_rotary_factor",1,0),0):q38_num(tc,"partial_rotary_factor",1,0);
    double rotary_product=(double)c->head_dim*partial;
    if(!isfinite(partial)||!isfinite(rotary_product)||rotary_product<0||
       rotary_product>(double)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived rotary_dim overflows int\n");exit(1);
    }
    /* Upstream Qwen4ExpTextConfig derives this with Python int(), i.e. truncation
     * toward zero rather than rounding to the nearest dimension. */
    c->rotary_dim=(int)rotary_product;
    c->idx_qheads=q38_num_int(tc,"indexer_n_heads",0,1,0,INT_MAX);
    c->idx_kheads=q38_num_int(tc,"indexer_kv_heads",0,1,0,INT_MAX);
    c->idx_dim=q38_num_int(tc,"indexer_head_dim",0,1,0,INT_MAX);
    c->idx_budget=q38_num_int(tc,"indexer_budget",0,1,0,INT_MAX);
    c->idx_ratio=q38_num_int(tc,"indexer_compress_ratio",0,1,0,INT_MAX);
    c->experts=q38_num_int(tc,"num_experts",0,1,0,INT_MAX);
    c->topk=q38_num_int(tc,"num_experts_per_tok",0,1,0,INT_MAX);
    c->inter=q38_num_int(tc,"moe_intermediate_size",0,1,0,INT_MAX);
    c->shared_inter=q38_num_int(tc,"shared_expert_intermediate_size",0,1,0,INT_MAX);
    c->norm_topk=q38_bool(tc,"norm_topk_prob",1);
    c->dn_kheads=q38_num_int(tc,"linear_num_key_heads",0,1,0,INT_MAX);
    c->dn_vheads=q38_num_int(tc,"linear_num_value_heads",0,1,0,INT_MAX);
    c->dn_kdim=q38_num_int(tc,"linear_key_head_dim",0,1,0,INT_MAX);
    c->dn_vdim=q38_num_int(tc,"linear_value_head_dim",0,1,0,INT_MAX);
    c->dn_convk=q38_num_int(tc,"linear_conv_kernel_dim",0,1,0,INT_MAX);
    int dn_qk=q38_derived_product("deltanet_qk",c->dn_kheads,c->dn_kdim);
    dn_qk=q38_derived_product("deltanet_qk_twice",2,dn_qk);
    c->dn_conv_dim=q38_derived_sum("dn_conv_dim",dn_qk,
                                    q38_derived_product("deltanet_v",c->dn_vheads,c->dn_vdim));
    c->ple_dim=q38_num_int(tc,"ple_embed_dim",c->hidden,0,0,INT_MAX);
    c->ple_convk=q38_num_int(tc,"ple_conv_kernel_size",4,0,0,INT_MAX);
    c->ngram_size=q38_num_int(tc,"ngram_size",3,0,0,INT_MAX);
    c->heads_per_ngram=q38_num_int(tc,"heads_per_ngram",8,0,0,INT_MAX);
    c->ngram_heads=(c->ngram_size>0)?q38_derived_product("ngram_heads",c->ngram_size-1,c->heads_per_ngram):0;
    if(c->ngram_heads<=0){fprintf(stderr,"[qwen38 config] derived ngram_heads invalid\n");exit(1);}
    c->ngram_head_dim=c->ple_dim/c->ngram_heads;
    c->ngram_parts=q38_num_int(tc,"split_ngram_parts",1,0,0,INT_MAX);
    c->ple_layer=-1;
    jval *pl=json_get(tc,"ple_layer_ids");
    if(pl&&pl->t==J_ARR&&pl->len){
        if(pl->len!=1||pl->kids[0]->t!=J_NUM){fprintf(stderr,"only one PLE layer is supported\n");exit(1);}
        double raw=pl->kids[0]->num;
        if(!isfinite(raw)||floor(raw)!=raw||raw<1||raw>(double)INT_MAX){
            fprintf(stderr,"config.json: ple_layer_ids[0] must be a positive integer\n");exit(1);
        }
        int one_based=(int)raw;
        c->ple_layer=one_based-1;
    }
    c->is_attn=(uint8_t*)calloc((size_t)c->layers,1);
    jval *lt=json_get(tc,"layer_types");
    if(!lt||lt->t!=J_ARR||lt->len!=c->layers){fprintf(stderr,"config layer_types must have %d entries\n",c->layers);exit(1);}
    for(int i=0;i<c->layers;i++){
        const char *s=lt->kids[i]->t==J_STR?lt->kids[i]->str:NULL;
        if(!s){fprintf(stderr,"invalid layer_types[%d]\n",i);exit(1);}
        if(!strcmp(s,"linear_attention")) c->is_attn[i]=0;
        else if(!strcmp(s,"full_attention")||!strcmp(s,"qwen_sparse_attention")) c->is_attn[i]=1;
        else {fprintf(stderr,"unsupported layer type %s\n",s);exit(1);}
    }
    /* Vision: opzionale. Un checkpoint di solo testo non ha vision_config, e in
     * quel caso la torre resta spenta invece di rifiutare il modello. La
     * geometria viene dal file, non da costanti qui: una torre di misura diversa
     * deve fallire dicendo cosa non torna, non leggere pesi della misura
     * sbagliata. */
    {
        jval *vc = q38_obj(root, "vision_config");
        if (vc) {
            c->vis_depth      = q38_num_int(vc,"depth",0,1,1,1024);
            c->vis_hidden     = q38_num_int(vc,"hidden_size",0,1,1,65536);
            c->vis_heads      = q38_num_int(vc,"num_heads",0,1,1,1024);
            c->vis_inter      = q38_num_int(vc,"intermediate_size",0,1,1,262144);
            c->vis_patch      = q38_num_int(vc,"patch_size",16,0,1,512);
            c->vis_merge      = q38_num_int(vc,"spatial_merge_size",2,0,1,8);
            c->vis_temporal   = q38_num_int(vc,"temporal_patch_size",2,0,1,8);
            c->vis_in_ch      = q38_num_int(vc,"in_channels",3,0,1,8);
            c->vis_out_hidden = q38_num_int(vc,"out_hidden_size",c->hidden,0,1,65536);
            c->vis_num_pos    = q38_num_int(vc,"num_position_embeddings",0,1,1,1<<20);
            c->image_token    = q38_num_int(root,"image_token_id",-1,0,0,INT_MAX);
        }
    }
    json_free(root); free(buf); free(arena);
}

#define Q38_NEED(x,...) do{if(!(x)){fprintf(stderr,"[qwen38 config] ");fprintf(stderr,__VA_ARGS__);fprintf(stderr," -- refusing\n");exit(1);}}while(0)
static void q38_validate_cfg(const Cfg *c) {
    Q38_NEED(c->hidden>0&&c->hidden<=65536,"hidden_size=%d",c->hidden);
    if (c->vis_depth) {
        /* La torre proietta direttamente nello spazio del testo: se le due
         * dimensioni non coincidono i token immagine finirebbero nel posto
         * giusto con i numeri sbagliati, che e' peggio di un rifiuto. */
        Q38_NEED(c->vis_out_hidden==c->hidden,
                 "vision out_hidden_size=%d but text hidden_size=%d",
                 c->vis_out_hidden,c->hidden);
        Q38_NEED(c->vis_hidden%c->vis_heads==0,
                 "vision hidden_size=%d not divisible by num_heads=%d",
                 c->vis_hidden,c->vis_heads);
        Q38_NEED(c->image_token>=0&&c->image_token<c->vocab,
                 "image_token_id=%d outside vocabulary",c->image_token);
    }
    Q38_NEED(c->layers>0&&c->layers<=Q38_MAX_LAYERS,"layers=%d",c->layers);
    Q38_NEED(c->vocab>0&&c->max_positions>0&&
             c->max_positions<=QWEN38_ATTN_MAX_CTX,"vocab/context invalid");
    Q38_NEED(isfinite(c->eps)&&c->eps>0.f&&isfinite(c->theta)&&c->theta>0.f,"RoPE/norm constants invalid");
    Q38_NEED(c->eos_id>=0&&c->eos_id<c->vocab,"eos=%d",c->eos_id);
    Q38_NEED(c->hc_count>1&&c->hc_count<=16&&c->hc_rank>0&&
             c->hc_rank<=65536&&c->hc_width>0,"gated residual dimensions invalid");
    Q38_NEED(c->q_heads>0&&c->kv_heads>0&&c->q_heads%c->kv_heads==0,"attention heads invalid");
    Q38_NEED(c->head_dim>0&&c->rotary_dim>0&&!(c->rotary_dim&1)&&c->rotary_dim<=c->head_dim,"RoPE dimensions invalid");
    Q38_NEED(c->head_dim<=INT_MAX/2&&c->q_heads<=INT_MAX/(2*c->head_dim)&&
             c->kv_heads<=INT_MAX/c->head_dim,"attention projection dimensions overflow");
    Q38_NEED(c->idx_qheads>0&&c->idx_kheads==1&&c->idx_dim>=c->rotary_dim,"indexer dimensions invalid");
    Q38_NEED(c->idx_ratio>0&&c->idx_budget>0&&c->idx_budget%c->idx_ratio==0&&
             c->idx_budget<=INT_MAX-c->idx_ratio+1,"indexer budget invalid");
    Q38_NEED(c->idx_qheads<INT_MAX&&c->idx_qheads<=INT_MAX/c->idx_dim-c->idx_kheads,
             "indexer projection dimensions overflow");
    Q38_NEED(c->experts>0&&c->experts<=Q38_MAX_EXPERTS,"experts=%d",c->experts);
    Q38_NEED(c->topk>0&&c->topk<=Q38_MAX_TOPK&&c->topk<=c->experts,"topk=%d",c->topk);
    Q38_NEED(c->inter<=INT_MAX/2,"expert projection dimensions overflow");
    Q38_NEED(c->inter>0&&c->shared_inter>0,"MoE widths invalid");
    Q38_NEED(c->dn_vheads>0&&c->dn_kheads>0&&c->dn_vheads%c->dn_kheads==0,"DeltaNet heads invalid");
    Q38_NEED(c->dn_kdim>0&&c->dn_vdim>0&&c->dn_vdim<=512&&c->dn_convk>=2&&c->dn_conv_dim>0&&
             c->dn_vdim<=INT_MAX/c->dn_vheads&&c->dn_kdim<=INT_MAX/c->dn_vdim,
             "DeltaNet dimensions invalid");
    Q38_NEED((uint64_t)c->dn_vheads<=SIZE_MAX/sizeof(float)/(uint64_t)c->dn_kdim/(uint64_t)c->dn_vdim&&
             (uint64_t)c->dn_conv_dim<=SIZE_MAX/sizeof(float)/(uint64_t)(c->dn_convk-1),
             "DeltaNet state dimensions overflow");
    Q38_NEED(c->ngram_size==3&&c->ngram_heads>0&&c->ngram_heads<=64&&
             c->ple_dim>0&&c->ple_dim%c->ngram_heads==0&&
             c->ngram_head_dim>0&&c->ngram_head_dim<=512&&
             c->ple_convk>=2&&c->ple_convk<=INT_MAX/c->ngram_size,
             "PLE dimensions invalid");
    Q38_NEED((uint64_t)c->hc_width<=SIZE_MAX/sizeof(float)/(uint64_t)(c->ple_convk-1)/(uint64_t)c->ngram_size,
             "PLE state dimensions overflow");
    Q38_NEED(c->ngram_parts>0&&c->ngram_parts<=Q38_MAX_PLE_PARTS,"PLE parts=%d",c->ngram_parts);
    Q38_NEED(c->ple_layer>=0&&c->ple_layer<c->layers,"PLE layer=%d",c->ple_layer);
}

static float *q38_load_tensor(Model *m,const char *name,int64_t want) {
    st_tensor *t=st_find(&m->S,name);
    if(!t){fprintf(stderr,"missing %s\n",name);exit(1);}
    if(t->numel!=want){fprintf(stderr,"%s: %lld elements, expected %lld\n",name,(long long)t->numel,(long long)want);exit(1);}
    float *p=falloc(want); st_read_f32(&m->S,name,p,1); return p;
}

static int q38_env_bool(const char *name,int default_value) {
    const char *value=getenv(name);if(!value||!*value)return default_value;
    if(value[0]=='0'&&!value[1])return 0;
    if(value[0]=='1'&&!value[1])return 1;
    fprintf(stderr,"%s must be exactly 0 or 1\n",name);exit(1);
}

static Q38Weight q38_load_weight(Model *m,const char *name,int rows,int cols) {
    st_tensor *tensor=st_find(&m->S,name);Q38Weight weight={0};
    if(!tensor){fprintf(stderr,"missing %s\n",name);exit(1);}
    if(tensor->rank!=2||tensor->shape[0]!=rows||tensor->shape[1]!=cols||
       tensor->numel!=(int64_t)rows*cols){
        fprintf(stderr,"%s: invalid matrix shape, expected [%d,%d]\n",name,rows,cols);exit(1);
    }
    if(tensor->dtype==0&&m->native_bf16){
        q38_weight_reserve(&weight,Q38_WEIGHT_BF16,rows,cols);
        if(tensor->nbytes!=weight.elements*(int64_t)sizeof(uint16_t)){
            fprintf(stderr,"%s: invalid BF16 byte count\n",name);exit(1);
        }
        st_read_raw_cap(&m->S,name,weight.data,tensor->nbytes,1);
    } else {
        if(tensor->dtype<0||tensor->dtype>2){
            fprintf(stderr,"%s: unsupported resident dtype %s\n",name,st_dtype_name(tensor->dtype));exit(1);
        }
        q38_weight_reserve(&weight,Q38_WEIGHT_F32,rows,cols);
        st_read_f32(&m->S,name,(float*)weight.data,1);
    }
    m->resident_weight_bytes+=q38_weight_bytes(&weight);
    return weight;
}

static void q38_name(Model *m,char *out,size_t cap,int layer,const char *suffix) {
    snprintf(out,cap,"%s.layers.%d.%s",m->prefix,layer,suffix);
}

static void q38_load_gr(Model *m,GatedResidual *g,int layer,const char *kind,int inject) {
    Cfg *c=&m->c; char nm[320],base[180];
    if(layer>=0) snprintf(base,sizeof base,"layers.%d.%s",layer,kind); else snprintf(base,sizeof base,"hyper_connection_mixer");
    snprintf(nm,sizeof nm,"%s.%s.hc_norm.weight",m->prefix,base); g->norm=q38_load_tensor(m,nm,c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_down.weight",m->prefix,base); g->down=q38_load_weight(m,nm,c->hc_rank,c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_up.weight",m->prefix,base); g->up=q38_load_weight(m,nm,c->hc_width,c->hc_rank);
    if(inject){snprintf(nm,sizeof nm,"%s.%s.block_inject_weight.weight",m->prefix,base);g->inject=q38_load_weight(m,nm,c->hc_count,c->hc_width);}
}

static void q38_load_ple(Model *m,Layer *l) {
    Cfg *c=&m->c; int i=c->ple_layer; char nm[320];
    q38_name(m,nm,sizeof nm,i,"ple.key_proj.weight");l->ple_key=q38_load_weight(m,nm,c->hc_width,c->ple_dim);
    q38_name(m,nm,sizeof nm,i,"ple.value_proj.weight");l->ple_value=q38_load_weight(m,nm,c->hidden,c->ple_dim);
    #define PL(field,suf,n) q38_name(m,nm,sizeof nm,i,"ple." suf); l->field=q38_load_tensor(m,nm,(n))
    PL(ple_norm_key,"norm_key.weight",c->hc_width);PL(ple_norm_query,"norm_query.weight",c->hc_width);
    PL(ple_norm_conv,"norm_conv.weight",c->hc_width);PL(ple_conv,"conv1d.weight",(int64_t)c->hc_width*c->ple_convk);
    #undef PL
    const char *bufs[]={"layer_multipliers","ngram_heads_vocab_sizes","ngram_heads_offsets"};
    int64_t *dsts[]={m->ple_multipliers,m->ple_head_vocab,m->ple_head_offset};
    int counts[]={c->ngram_size,c->ngram_heads,c->ngram_heads};
    for(int b=0;b<3;b++){
        q38_name(m,nm,sizeof nm,i,"ple.ple_embedding."); strncat(nm,bufs[b],sizeof(nm)-strlen(nm)-1);
        st_tensor *t=st_find(&m->S,nm); if(!t||t->dtype!=6||t->numel!=counts[b]){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_raw_cap(&m->S,nm,dsts[b],(int64_t)counts[b]*8,1);
    }
    for(int h=0;h<c->ngram_heads;h++)
        Q38_NEED(m->ple_head_vocab[h]>0&&m->ple_head_offset[h]>=0&&
                 m->ple_head_vocab[h]<=INT64_MAX-m->ple_head_offset[h],
                 "invalid PLE head range %d",h);
    m->ple_part_count=0; m->ple_part_start[0]=0;
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(!t||t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim){fprintf(stderr,"invalid PLE table %s\n",nm);exit(1);}
        snprintf(m->ple_part_names[0],sizeof m->ple_part_names[0],"%s",nm);
        m->ple_parts[0]=t; m->ple_part_count=1;
        m->ple_part_start[1]=m->ple_parts[0]->shape[0];
    } else {
        for(int p=0;p<c->ngram_parts;p++){
            q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.");
            size_t z=strlen(nm); snprintf(nm+z,sizeof(nm)-z,"shard_%d.weight",p);
            st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing PLE shard %s\n",nm);exit(1);}
            if(t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim||
               m->ple_part_start[p]>INT64_MAX-t->shape[0]){fprintf(stderr,"invalid PLE shard %s\n",nm);exit(1);}
            snprintf(m->ple_part_names[p],sizeof m->ple_part_names[p],"%s",nm);
            m->ple_parts[p]=t; m->ple_part_start[p+1]=m->ple_part_start[p]+t->shape[0];
            m->ple_part_count++;
        }
    }
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight_scale");
    m->ple_weight_scale=1.f;
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm); if(t->numel!=1){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_f32(&m->S,nm,&m->ple_weight_scale,1);
    }
    int64_t need=0; for(int h=0;h<c->ngram_heads;h++) if(m->ple_head_offset[h]+m->ple_head_vocab[h]>need) need=m->ple_head_offset[h]+m->ple_head_vocab[h];
    Q38_NEED(m->ple_part_start[m->ple_part_count]>=need,"PLE table rows %lld < required %lld",(long long)m->ple_part_start[m->ple_part_count],(long long)need);
}

static void q38_alloc_state(Model *m) {
    Cfg *c=&m->c;
    m->DN_rec=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->DN_conv=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->V=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
    if(!m->DN_rec||!m->DN_conv||!m->K||!m->V||!m->IK){fprintf(stderr,"OOM model state metadata\n");exit(1);}
    for(int i=m->range_begin;i<m->range_end;i++) if(!c->is_attn[i]) {
        m->DN_rec[i]=(float*)calloc((size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim,sizeof(float));
        m->DN_conv[i]=(float*)calloc((size_t)c->dn_conv_dim*(c->dn_convk-1),sizeof(float));
        if(!m->DN_rec[i]||!m->DN_conv[i]){fprintf(stderr,"OOM DeltaNet state\n");exit(1);}
    }
    if(c->ple_layer>=m->range_begin&&c->ple_layer<m->range_end){
        m->ple_history=(int64_t*)calloc(2,sizeof(int64_t));
        m->PLE_conv_state=(float*)calloc((size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size,sizeof(float));
        if(!m->ple_history||!m->PLE_conv_state){fprintf(stderr,"OOM PLE state\n");exit(1);}
    }
}

/* ---- vision ------------------------------------------------------------- */

/* La torre e' in F32 residente: 27 blocchi da 1152 sono ~0.6 GB, che accanto ai
 * 9.2 GiB dei pesi densi non cambia la classe di macchina. Gli esperti restano
 * su disco; la torre no, perche' si usa una volta per immagine e non per token. */
static const float *q38_vis_tensor(Model *m,const char *suffix,int64_t expect) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.visual.%s",m->prefix,suffix);
    if(!st_has(&m->S,nm)){
        snprintf(nm,sizeof nm,"model.visual.%s",suffix);
        if(!st_has(&m->S,nm)){fprintf(stderr,"vision tensor missing: %s\n",suffix);exit(1);}
    }
    st_tensor *t=st_find(&m->S,nm);
    if(expect>0&&t->numel!=expect){
        fprintf(stderr,"vision tensor %s has %lld values, expected %lld -- refusing\n",
                nm,(long long)t->numel,(long long)expect);exit(1);
    }
    float *buf=(float*)malloc((size_t)t->numel*sizeof(float));
    if(!buf){fprintf(stderr,"OOM loading %s\n",nm);exit(1);}
    st_read_f32(&m->S,nm,buf,t->numel);
    m->resident_weight_bytes+=(uint64_t)t->numel*sizeof(float);
    return buf;
}

static void q38_vis_linear(Model *m,Q38Linear *l,const char *stem,int out,int in) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.weight",stem); l->w=q38_vis_tensor(m,nm,(int64_t)out*in);
    snprintf(nm,sizeof nm,"%s.bias",stem);   l->b=q38_vis_tensor(m,nm,out);
    l->out=out; l->in=in;
}

static void q38_vis_norm(Model *m,Q38Norm *n,const char *stem,int width) {
    char nm[512];
    snprintf(nm,sizeof nm,"%s.weight",stem); n->w=q38_vis_tensor(m,nm,width);
    snprintf(nm,sizeof nm,"%s.bias",stem);   n->b=q38_vis_tensor(m,nm,width);
}

static void q38_load_vision(Model *m) {
    Cfg *c=&m->c;
    if(!c->vis_depth) return;
    char probe[512];
    snprintf(probe,sizeof probe,"%s.visual.pos_embed.weight",m->prefix);
    if(!st_has(&m->S,probe)&&!st_has(&m->S,"model.visual.pos_embed.weight")){
        /* config multimodale ma pesi assenti: e' un export solo-testo di un
         * checkpoint multimodale. Spegnere la torre e dirlo e' meglio che
         * rifiutare un modello che per il testo funziona benissimo. */
        fprintf(stderr,"[qwen38] vision_config present but no visual weights; text only\n");
        c->vis_depth=0; return;
    }
    Q38Vision *v=&m->vis;
    memset(v,0,sizeof *v);
    v->depth=c->vis_depth; v->hidden=c->vis_hidden; v->heads=c->vis_heads;
    v->head_dim=c->vis_hidden/c->vis_heads; v->inter=c->vis_inter;
    v->patch=c->vis_patch; v->merge=c->vis_merge; v->temporal=c->vis_temporal;
    v->in_ch=c->vis_in_ch; v->out_hidden=c->vis_out_hidden;
    v->num_pos=c->vis_num_pos; v->side=(int)(sqrt((double)c->vis_num_pos)+0.5);
    v->eps=1e-6f;
    if(v->side*v->side!=v->num_pos){
        fprintf(stderr,"[qwen38] num_position_embeddings=%d is not a square grid -- refusing\n",
                v->num_pos);exit(1);
    }
    int features=v->in_ch*v->temporal*v->patch*v->patch;
    q38_vis_linear(m,&v->patch_embed,"patch_embed.proj",v->hidden,features);
    v->pos_embed=q38_vis_tensor(m,"pos_embed.weight",(int64_t)v->num_pos*v->hidden);
    v->blocks=(Q38VBlock*)calloc((size_t)v->depth,sizeof(Q38VBlock));
    if(!v->blocks){fprintf(stderr,"OOM vision blocks\n");exit(1);}
    for(int i=0;i<v->depth;i++){
        char stem[160];
        snprintf(stem,sizeof stem,"blocks.%d.norm1",i);          q38_vis_norm(m,&v->blocks[i].norm1,stem,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.norm2",i);          q38_vis_norm(m,&v->blocks[i].norm2,stem,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.attn.qkv",i);       q38_vis_linear(m,&v->blocks[i].qkv,stem,3*v->hidden,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.attn.proj",i);      q38_vis_linear(m,&v->blocks[i].proj,stem,v->hidden,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.mlp.linear_fc1",i); q38_vis_linear(m,&v->blocks[i].fc1,stem,v->inter,v->hidden);
        snprintf(stem,sizeof stem,"blocks.%d.mlp.linear_fc2",i); q38_vis_linear(m,&v->blocks[i].fc2,stem,v->hidden,v->inter);
    }
    int wide=v->hidden*v->merge*v->merge;
    q38_vis_norm(m,&v->merger_norm,"merger.norm",v->hidden);
    q38_vis_linear(m,&v->merger_fc1,"merger.linear_fc1",wide,wide);
    q38_vis_linear(m,&v->merger_fc2,"merger.linear_fc2",v->out_hidden,wide);
    m->vis_ready=1;
    fprintf(stderr,"[qwen38] vision tower: %d blocks, hidden %d, %d heads, patch %d, merge %d\n",
            v->depth,v->hidden,v->heads,v->patch,v->merge);
}

/* Esegue la torre su un'immagine gia' preprocessata e prepara la mappa
 * posizione->riga. `ids`/`n` sono i token del prompt: le righe vengono
 * assegnate ai token immagine nell'ordine in cui compaiono. */
static int q38_vision_attach(Model *m,const float *patches,int grid_h,int grid_w,
                             const int *ids,int n) {
    Cfg *c=&m->c;
    if(!m->vis_ready) return -1;
    int tokens=(grid_h*grid_w)/(c->vis_merge*c->vis_merge);
    int slots=0;
    for(int i=0;i<n;i++) if(ids[i]==c->image_token) slots++;
    if(slots!=tokens){
        /* Il numero di segnaposto nel prompt DEVE essere quello che la griglia
         * produce. Se non lo e', il template e il preprocessing hanno visto due
         * immagini diverse, e proseguire vorrebbe dire mettere i vettori giusti
         * nelle posizioni sbagliate. */
        fprintf(stderr,"[qwen38] prompt has %d image placeholders but the grid gives %d tokens\n",
                slots,tokens);
        return -1;
    }
    free(m->vis_rows); free(m->vis_map);
    m->vis_rows=(float*)calloc((size_t)tokens*c->hidden,sizeof(float));
    m->vis_map=(int*)malloc((size_t)n*sizeof(int));
    if(!m->vis_rows||!m->vis_map){free(m->vis_rows);free(m->vis_map);
        m->vis_rows=NULL;m->vis_map=NULL;return -1;}
    if(q38_vision_forward(&m->vis,patches,grid_h,grid_w,m->vis_rows)!=tokens){
        free(m->vis_rows);free(m->vis_map);m->vis_rows=NULL;m->vis_map=NULL;return -1;}
    int next=0;
    for(int i=0;i<n;i++) m->vis_map[i]=(ids[i]==c->image_token)?next++:-1;
    m->vis_map_len=n; m->vis_rows_n=tokens;
    return tokens;
}

static void q38_vision_detach(Model *m) {
    free(m->vis_rows); free(m->vis_map);
    m->vis_rows=NULL; m->vis_map=NULL; m->vis_map_len=0; m->vis_rows_n=0;
}

static void model_init_range(Model *m,const char *snap,int cap,int bits,
                             int layer_begin,int layer_end,int load_boundaries,
                             int allocate_state) {
    (void)bits; memset(m,0,sizeof(*m)); double t0=now_s();
    m->native_fp8=q38_env_bool("Q38_NATIVE_FP8",1);
    m->native_bf16=q38_env_bool("Q38_NATIVE_BF16",1);
    m->expert_prefetch=q38_env_bool("Q38_EXPERT_PREFETCH",1);
    m->expert_parallel_reads=q38_env_bool("Q38_EXPERT_PARALLEL_READS",1);
    m->prefill_batch=q38_env_bool("Q38_PREFILL_BATCH",1);
    q38_load_cfg(&m->c,snap); q38_validate_cfg(&m->c); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[320];
    if(st_has(&m->S,"model.language_model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model.language_model");
    else if(st_has(&m->S,"model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model");
    else {fprintf(stderr,"checkpoint has no Qwen4-Exp text embedding\n");exit(1);}
    if(layer_end==0) layer_end=c->layers;
    if(layer_begin<0||layer_begin>=layer_end||layer_end>c->layers){fprintf(stderr,"invalid Qwen3.8 layer range [%d,%d)\n",layer_begin,layer_end);exit(1);}
    m->range_begin=layer_begin; m->range_end=layer_end;
    if(load_boundaries){
        snprintf(nm,sizeof nm,"%s.embed_tokens.weight",m->prefix); m->embed=q38_load_weight(m,nm,c->vocab,c->hidden);
        m->lm_head=q38_load_weight(m,"lm_head.weight",c->vocab,c->hidden);
        q38_load_gr(m,&m->final_gr,-1,NULL,0);
    }
    m->L=(Layer*)calloc((size_t)c->layers,sizeof(Layer));
    m->cache=(LCache*)calloc((size_t)c->layers,sizeof(LCache));
    m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers,
                                                  sizeof(*m->expert_scales));
    if(!m->L||!m->cache||!m->expert_scales){fprintf(stderr,"OOM model metadata\n");exit(1);}
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i]; q38_load_gr(m,&l->attn_gr,i,"attn_hyper_connection",1); q38_load_gr(m,&l->mlp_gr,i,"mlp_hyper_connection",1);
        #define WLD(field,suf,o,in) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_weight(m,nm,(o),(in))
        #define VLD(field,suf,n) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_tensor(m,nm,(n))
        WLD(router,"mlp.gate.weight",c->experts,c->hidden);
        WLD(sh_g,"mlp.shared_expert.gate_proj.weight",c->shared_inter,c->hidden);
        WLD(sh_u,"mlp.shared_expert.up_proj.weight",c->shared_inter,c->hidden);
        WLD(sh_d,"mlp.shared_expert.down_proj.weight",c->hidden,c->shared_inter);
        VLD(sh_gate,"mlp.shared_expert_gate.weight",c->hidden);
        if(c->is_attn[i]){
            WLD(q,"self_attn.q_proj.weight",c->q_heads*c->head_dim*2,c->hidden);
            WLD(k,"self_attn.k_proj.weight",c->kv_heads*c->head_dim,c->hidden);
            WLD(v,"self_attn.v_proj.weight",c->kv_heads*c->head_dim,c->hidden);
            WLD(o,"self_attn.o_proj.weight",c->hidden,c->q_heads*c->head_dim);
            VLD(qn,"self_attn.q_norm.weight",c->head_dim);VLD(kn,"self_attn.k_norm.weight",c->head_dim);
            WLD(idx_qk,"self_attn.indexer.index_qk_proj.weight",(c->idx_qheads+c->idx_kheads)*c->idx_dim,c->hidden);
            VLD(idx_qn,"self_attn.indexer.q_layernorm.weight",c->idx_dim);VLD(idx_kn,"self_attn.indexer.k_layernorm.weight",c->idx_dim);
        } else {
            int vd=c->dn_vheads*c->dn_vdim;
            WLD(dn_qkv,"linear_attn.in_proj_qkv.weight",c->dn_conv_dim,c->hidden);
            WLD(dn_z,"linear_attn.in_proj_z.weight",vd,c->hidden);
            WLD(dn_b,"linear_attn.in_proj_b.weight",c->dn_vheads,c->hidden);
            WLD(dn_a,"linear_attn.in_proj_a.weight",c->dn_vheads,c->hidden);
            VLD(dn_conv,"linear_attn.conv1d.weight",(int64_t)c->dn_conv_dim*c->dn_convk);
            VLD(dn_dtbias,"linear_attn.dt_bias",c->dn_vheads);VLD(dn_alog,"linear_attn.A_log",c->dn_vheads);
            VLD(dn_norm,"linear_attn.norm.weight",c->dn_vdim);
            WLD(dn_out,"linear_attn.out_proj.weight",c->hidden,vd);
        }
        #undef WLD
        #undef VLD
        LCache *lc=&m->cache[i]; lc->cap=cap; lc->slots=(Slot*)calloc((size_t)cap,sizeof(Slot)); lc->by_expert=(int*)malloc((size_t)c->experts*sizeof(int)); lc->freq=(uint16_t*)calloc((size_t)c->experts,sizeof(uint16_t));
        if(!lc->slots||!lc->by_expert||!lc->freq){fprintf(stderr,"OOM expert cache\n");exit(1);} for(int e=0;e<c->experts;e++)lc->by_expert[e]=-1;
    }
    if(c->ple_layer>=layer_begin&&c->ple_layer<layer_end) q38_load_ple(m,&m->L[c->ple_layer]);
    /* La torre solo quando il motore possiede la sequenza intera: uno shard che
     * ospita solo alcuni layer non ha da fare niente con le immagini, e
     * caricarla la' sarebbe mezzo giga per nulla. */
    if(load_boundaries&&q38_env_bool("Q38_VISION",1)) q38_load_vision(m);
    if(allocate_state) q38_alloc_state(m);
    m->dense_load_s=now_s()-t0;
    fprintf(stderr,"[qwen38] native text weights: prefix=%s, %d layers, PLE=%d, cache=%d/layer, "
                   "FP8=%s, BF16=%s (resident matrices %.2f GiB)\n",m->prefix,c->layers,c->ple_layer,cap,
                   m->native_fp8?"native":"expanded-f32",m->native_bf16?"native":"expanded-f32",
                   m->resident_weight_bytes/1073741824.0);
}

static void model_init(Model *m,const char *snap,int cap,int bits) {
    model_init_range(m,snap,cap,bits,0,0,1,1);
}

static void q38_gr_read(Model *m,const GatedResidual *g,const float *hyper,
                        int S,float *mixed,float *inject) {
    const Cfg *c=&m->c;
    int H=c->hidden,W=c->hc_width,C=c->hc_count,R=c->hc_rank;
    if(C<=0||H<=0||W!=C*H){fprintf(stderr,"invalid gated-residual width\n");exit(1);}
    float *norm,*low,*mix,*gr_owned=NULL; if(q38_cpu_v3()&&S==1){float *ws=q38_tls_f32(&g_q38_gr_ws,(size_t)(2*W+R));norm=ws;low=norm+W;mix=low+R;}else{gr_owned=falloc((int64_t)S*(2*W+R));norm=gr_owned;low=norm+(int64_t)S*W;mix=low+(int64_t)S*R;}
    /* Makes the initialized-input invariant visible to aggressive interprocedural
     * warning analysis; every element is overwritten by q38_rms0 below. */
    memset(norm,0,(size_t)S*W*sizeof(float));
    for(int s=0;s<S;s++) for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)s*W+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,g->norm+(int64_t)b*H,H,c->eps);
    q38_dense_matmul(m,low,norm,&g->down,S,W,R);
    for(int64_t z=0;z<(int64_t)S*R;z++) low[z]=q38_silu(low[z]/C);
    q38_dense_matmul(m,mix,low,&g->up,S,R,W);
    for(int s=0;s<S;s++) for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)s*W+(int64_t)b*H+d])*norm[(int64_t)s*W+(int64_t)b*H+d];
        mixed[(int64_t)s*H+d]=v/C;
    }
    if(inject){
        q38_dense_matmul(m,inject,norm,&g->inject,S,W,C);
        for(int64_t z=0;z<(int64_t)S*C;z++) inject[z]=2.f*q38_sigmoid(inject[z]/C);
    }
    if(gr_owned)free(gr_owned);
}

static void q38_gr_apply(const Cfg *c,float *hyper,const float *block,const float *inject,int S) {
    int H=c->hidden,C=c->hc_count,W=c->hc_width;
    for(int s=0;s<S;s++)for(int b=0;b<C;b++){
        float a=inject[(int64_t)s*C+b];
        for(int d=0;d<H;d++)hyper[(int64_t)s*W+(int64_t)b*H+d]+=a*block[(int64_t)s*H+d];
    }
}

typedef struct {
    st_tensor *tensor;
    int expert, projection;
} Q38ScaleDesc;

static int q38_scale_desc_cmp(const void *left,const void *right) {
    const Q38ScaleDesc *a=(const Q38ScaleDesc*)left;
    const Q38ScaleDesc *b=(const Q38ScaleDesc*)right;
    if(a->tensor->off<b->tensor->off)return -1;
    if(a->tensor->off>b->tensor->off)return 1;
    return 0;
}

static int q38_scale_group_span(Q38ScaleDesc *desc,int count,int *fd,
                                int64_t *begin,int64_t *nbytes) {
    if(!desc||count<1||!fd||!begin||!nbytes)return -1;
    qsort(desc,(size_t)count,sizeof(*desc),q38_scale_desc_cmp);
    int group_fd=desc[0].tensor->fd;
    int64_t first=desc[0].tensor->off,cursor=first;
    for(int i=0;i<count;i++){
        st_tensor *tensor=desc[i].tensor;
        if(tensor->fd!=group_fd||tensor->off!=cursor||tensor->nbytes<=0||
           tensor->nbytes>INT64_MAX-cursor)return -1;
        cursor+=tensor->nbytes;
    }
    *fd=group_fd;*begin=first;*nbytes=cursor-first;return 0;
}

static void q38_decode_scale_tensor(float *out,const unsigned char *raw,
                                     const st_tensor *tensor) {
    if(tensor->dtype==2)memcpy(out,raw,(size_t)tensor->nbytes);
    else for(int64_t i=0;i<tensor->numel;i++){
        uint16_t half;memcpy(&half,raw+(size_t)i*sizeof(half),sizeof(half));
        out[i]=tensor->dtype==0?bf16_to_f32(half):f16_to_f32(half);
    }
}

/* The official checkpoint stores every layer's gate/up scale sidecars as one
 * compact range and every down sidecar as another.  Normalize both ranges to
 * an expert-indexed F32 bank once, scattering by the numeric expert id rather
 * than the checkpoint's lexical tensor order (0, 1, 10, ...).  Variants that
 * do not satisfy the compact-range invariant simply keep the established
 * per-matrix loader. */
static int q38_prepare_expert_scale_bank(Model *m,int layer) {
    Cfg *c=&m->c;
    if(!m->native_fp8||layer<0||layer>=c->layers)return 0;
    if(!m->expert_scales){
        m->expert_scales=(Q38ExpertScaleCache*)calloc((size_t)c->layers,
                                                       sizeof(*m->expert_scales));
        if(!m->expert_scales){fprintf(stderr,"OOM expert scale metadata\n");exit(1);}
    }
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    if(cache->ready)return cache->ready>0;
    int64_t scale_count=fp8_nblk(c->inter)*fp8_nblk(c->hidden);
    if(scale_count<1||(uint64_t)c->experts>SIZE_MAX/(3u*(uint64_t)scale_count*sizeof(float))){
        fprintf(stderr,"invalid expert scale-bank geometry\n");exit(1);
    }
    Q38ScaleDesc *gate_up=(Q38ScaleDesc*)malloc((size_t)c->experts*2*sizeof(*gate_up));
    Q38ScaleDesc *down=(Q38ScaleDesc*)malloc((size_t)c->experts*sizeof(*down));
    if(!gate_up||!down){fprintf(stderr,"OOM expert scale descriptors\n");exit(1);}
    int ngu=0,ndown=0;char suffix[192],name[320];
    const char *projection[3]={"gate_proj","up_proj","down_proj"};
    for(int expert=0;expert<c->experts;expert++)for(int kind=0;kind<3;kind++){
        int rows=kind==2?c->hidden:c->inter;
        int cols=kind==2?c->inter:c->hidden;
        int length=snprintf(suffix,sizeof suffix,
            "mlp.experts.%d.%s.weight_scale_inv",expert,projection[kind]);
        if(length<0||(size_t)length>=sizeof suffix)goto incompatible;
        q38_name(m,name,sizeof name,layer,suffix);
        st_tensor *tensor=st_find(&m->S,name);
        int64_t block_rows=fp8_nblk(rows),block_cols=fp8_nblk(cols);
        if(!tensor||tensor->dtype<0||tensor->dtype>2||tensor->rank!=2||
           tensor->shape[0]!=block_rows||tensor->shape[1]!=block_cols||
           tensor->numel!=scale_count||
           tensor->numel>INT64_MAX/st_dtype_esz(tensor->dtype)||
           tensor->nbytes!=tensor->numel*st_dtype_esz(tensor->dtype))
            goto incompatible;
        Q38ScaleDesc item={tensor,expert,kind};
        if(kind<2)gate_up[ngu++]=item;else down[ndown++]=item;
    }
    int gu_fd,down_fd;int64_t gu_begin,down_begin,gu_bytes,down_bytes;
    if(q38_scale_group_span(gate_up,ngu,&gu_fd,&gu_begin,&gu_bytes)||
       q38_scale_group_span(down,ndown,&down_fd,&down_begin,&down_bytes)||
       (uint64_t)gu_bytes>SIZE_MAX||(uint64_t)down_bytes>SIZE_MAX)
        goto incompatible;
    float *values=(float*)malloc((size_t)c->experts*3*(size_t)scale_count*sizeof(float));
    unsigned char *gu_raw=(unsigned char*)malloc((size_t)gu_bytes);
    unsigned char *down_raw=(unsigned char*)malloc((size_t)down_bytes);
    if(!values||!gu_raw||!down_raw){fprintf(stderr,"OOM resident expert scales\n");exit(1);}
    double started=now_s();
    st_read_range_raw_cap(&m->S,gu_fd,gu_begin,gu_bytes,gu_raw,gu_bytes,1,
                          "pread Qwen3.8 gate/up scales");
    st_read_range_raw_cap(&m->S,down_fd,down_begin,down_bytes,down_raw,down_bytes,1,
                          "pread Qwen3.8 down scales");
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_scale_reads+=2;
    for(int i=0;i<ngu;i++){
        Q38ScaleDesc *item=&gate_up[i];
        float *dst=values+((int64_t)item->expert*3+item->projection)*scale_count;
        q38_decode_scale_tensor(dst,gu_raw+(item->tensor->off-gu_begin),item->tensor);
    }
    for(int i=0;i<ndown;i++){
        Q38ScaleDesc *item=&down[i];
        float *dst=values+((int64_t)item->expert*3+item->projection)*scale_count;
        q38_decode_scale_tensor(dst,down_raw+(item->tensor->off-down_begin),item->tensor);
    }
    free(gu_raw);free(down_raw);free(gate_up);free(down);
    cache->values=values;cache->scale_count=scale_count;cache->ready=1;
    m->expert_scale_bytes+=(uint64_t)c->experts*3*(uint64_t)scale_count*sizeof(float);
    return 1;
incompatible:
    free(gate_up);free(down);cache->ready=-1;return 0;
}

static void q38_bind_borrowed_fp8(Q38Weight *weight,void *data,float *scales,
                                  int rows,int cols) {
    q38_weight_free(weight);
    weight->data=data;weight->scales=scales;weight->rows=rows;weight->cols=cols;
    weight->elements=(int64_t)rows*cols;
    weight->scale_count=fp8_nblk(rows)*fp8_nblk(cols);
    weight->kind=Q38_WEIGHT_FP8;
}

static void q38_bind_fp8_slot(Slot *slot,float *scales,int scale_count,
                              int hidden,int intermediate) {
    int64_t matrix_bytes=(int64_t)hidden*intermediate;
    if(matrix_bytes<1||matrix_bytes>INT64_MAX/3||
       (uint64_t)(matrix_bytes*3)>SIZE_MAX||
       scale_count!=fp8_nblk(hidden)*fp8_nblk(intermediate)){
        fprintf(stderr,"invalid native FP8 expert slab geometry\n");exit(1);
    }
    q38_weight_free(&slot->gate);q38_weight_free(&slot->up);
    q38_weight_free(&slot->down);
    int64_t slab_bytes=matrix_bytes*3;
    if(!slot->fp8_slab||slot->fp8_slab_bytes!=slab_bytes){
        void *replacement=realloc(slot->fp8_slab,(size_t)slab_bytes);
        if(!replacement){fprintf(stderr,"OOM native FP8 expert slab\n");exit(1);}
        slot->fp8_slab=replacement;slot->fp8_slab_bytes=slab_bytes;
    }
    unsigned char *raw=(unsigned char*)slot->fp8_slab;
    q38_bind_borrowed_fp8(&slot->gate,raw,scales,intermediate,hidden);
    q38_bind_borrowed_fp8(&slot->up,raw+matrix_bytes,scales+scale_count,
                          intermediate,hidden);
    q38_bind_borrowed_fp8(&slot->down,raw+2*matrix_bytes,scales+2*scale_count,
                          hidden,intermediate);
}


/* ========================== BABA_V7_Q4_BOMB =========================== */
typedef struct { int fd; uint32_t layers,experts,H,F,group; uint64_t record_bytes; int ready; } Q38Q4Sidecar;
static Q38Q4Sidecar g_q38_q4={-2,0,0,0,0,0,0,0};
static unsigned char g_q38_q4_reset[Q38_MAX_LAYERS];
static int g_q38_q4_decode_scope;
/* ======================== BABA_V9_Q4_MMAP_BOMB =========================
 * Zero-copy view of the complete Q4 sidecar.
 *
 * Q38_Q4_MMAP=0 : old ReadFile/pread path
 * Q38_Q4_MMAP=1 : mapped records, demand-paged by Windows
 * Q38_Q4_MMAP=2 : mapped records + PrefetchVirtualMemory hint per miss batch
 * ====================================================================== */
#ifdef _WIN32
static HANDLE g_q38_q4_mmap_file=INVALID_HANDLE_VALUE;
static HANDLE g_q38_q4_mmap_handle=NULL;
static unsigned char *g_q38_q4_mmap_base=NULL;
static uint64_t g_q38_q4_mmap_size=0;
static uint64_t g_q38_q4_mmap_binds=0;
static uint64_t g_q38_q4_mmap_prefetches=0;
static uint64_t g_q38_q4_mmap_prefetch_bytes=0;
static int g_q38_q4_mmap_report_registered=0;

typedef struct {
    PVOID VirtualAddress;
    SIZE_T NumberOfBytes;
} Q38MemoryRangeEntry;

typedef BOOL (WINAPI *Q38PrefetchVirtualMemoryFn)(
    HANDLE, ULONG_PTR, Q38MemoryRangeEntry *, ULONG);
static Q38PrefetchVirtualMemoryFn g_q38_prefetch_vm=NULL;

static int q38_q4_mmap_mode_v9(void){
    const char *e=getenv("Q38_Q4_MMAP");
    if(!e||!*e||e[0]=='0')return 0;
    int m=atoi(e);
    return m>=2?2:1;
}

static void q38_q4_mmap_report_v9(void){
    if(!g_q38_q4_mmap_base)return;
    fprintf(stderr,
        "[BABA v9 mmap] binds=%llu prefetch_calls=%llu prefetch=%.1f MiB\n",
        (unsigned long long)g_q38_q4_mmap_binds,
        (unsigned long long)g_q38_q4_mmap_prefetches,
        (double)g_q38_q4_mmap_prefetch_bytes/1048576.0);
}

static int q38_q4_mmap_open_v9(void){
    if(!q38_q4_mmap_mode_v9())return 0;
    if(g_q38_q4_mmap_base)return 1;

    const char *path=getenv("Q38_Q4_SIDECAR");
    if(!path||!*path)return 0;

    g_q38_q4_mmap_file=CreateFileA(
        path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS,NULL);
    if(g_q38_q4_mmap_file==INVALID_HANDLE_VALUE){
        fprintf(stderr,"[BABA v9 mmap] CreateFile failed: %lu\n",
                (unsigned long)GetLastError());
        return 0;
    }

    LARGE_INTEGER sz;
    if(!GetFileSizeEx(g_q38_q4_mmap_file,&sz)||sz.QuadPart<=0){
        fprintf(stderr,"[BABA v9 mmap] GetFileSizeEx failed: %lu\n",
                (unsigned long)GetLastError());
        CloseHandle(g_q38_q4_mmap_file);
        g_q38_q4_mmap_file=INVALID_HANDLE_VALUE;
        return 0;
    }
    g_q38_q4_mmap_size=(uint64_t)sz.QuadPart;

    g_q38_q4_mmap_handle=CreateFileMappingA(
        g_q38_q4_mmap_file,NULL,PAGE_READONLY,0,0,NULL);
    if(!g_q38_q4_mmap_handle){
        fprintf(stderr,"[BABA v9 mmap] CreateFileMapping failed: %lu\n",
                (unsigned long)GetLastError());
        CloseHandle(g_q38_q4_mmap_file);
        g_q38_q4_mmap_file=INVALID_HANDLE_VALUE;
        return 0;
    }

    g_q38_q4_mmap_base=(unsigned char*)MapViewOfFile(
        g_q38_q4_mmap_handle,FILE_MAP_READ,0,0,0);
    if(!g_q38_q4_mmap_base){
        fprintf(stderr,"[BABA v9 mmap] MapViewOfFile failed: %lu\n",
                (unsigned long)GetLastError());
        CloseHandle(g_q38_q4_mmap_handle);g_q38_q4_mmap_handle=NULL;
        CloseHandle(g_q38_q4_mmap_file);g_q38_q4_mmap_file=INVALID_HANDLE_VALUE;
        return 0;
    }

    HMODULE k32=GetModuleHandleA("kernel32.dll");
    if(k32)g_q38_prefetch_vm=(Q38PrefetchVirtualMemoryFn)
        GetProcAddress(k32,"PrefetchVirtualMemory");

    if(!g_q38_q4_mmap_report_registered){
        atexit(q38_q4_mmap_report_v9);
        g_q38_q4_mmap_report_registered=1;
    }

    fprintf(stderr,
        "[BABA v9 mmap] ON: %.2f GiB mapped, mode=%d, PrefetchVirtualMemory=%s\n",
        (double)g_q38_q4_mmap_size/1073741824.0,
        q38_q4_mmap_mode_v9(),g_q38_prefetch_vm?"yes":"no");
    return 1;
}

static unsigned char *q38_q4_mmap_record_v9(int layer,int expert){
    if(!q38_q4_mmap_open_v9())return NULL;
    uint64_t off=64ull+
        ((uint64_t)layer*g_q38_q4.experts+(uint64_t)expert)*
        g_q38_q4.record_bytes;
    if(off>g_q38_q4_mmap_size ||
       g_q38_q4.record_bytes>g_q38_q4_mmap_size-off)return NULL;
    return g_q38_q4_mmap_base+off;
}

static void q38_q4_bind_mapped_v9(Model *m,Slot *s,unsigned char *p){
    /* ReadFile mode owns q4_slab; mmap mode borrows directly from file view. */
    free(s->q4_slab);s->q4_slab=NULL;s->q4_slab_bytes=0;
    q38_weight_free(&s->gate);q38_weight_free(&s->up);q38_weight_free(&s->down);
    free(s->fp8_slab);s->fp8_slab=NULL;s->fp8_slab_bytes=0;

    uint64_t mb=(uint64_t)m->c.hidden*m->c.inter/2;
    uint64_t sgu=(uint64_t)m->c.inter*(m->c.hidden/64)*4;
    s->q4.g4=p;
    s->q4.u4=p+mb;
    s->q4.d4=p+2*mb;
    s->q4.gs=(float*)(p+3*mb);
    s->q4.us=(float*)(p+3*mb+sgu);
    s->q4.ds=(float*)(p+3*mb+2*sgu);
    s->q4_active=1;
    g_q38_q4_mmap_binds++;
}
#endif
/* ====================================================================== */

/* ====================== BABA_V8_Q4_VICTIM_BOMB ========================
 * A second-level Q4 cache shared by all layers.
 *
 * The normal cache remains the fast L1 (cap experts per layer).  When L1
 * evicts a Q4 expert, ownership of its already-decoded Q4 slab is moved into
 * this global victim cache instead of being overwritten immediately.
 *
 * A victim hit moves the slab back into the active slot by POINTER SWAP:
 * no 2.6 MiB memcpy and no sidecar read.  Empty victim entries can retain a
 * spare slab, so after warm-up misses also reuse buffers instead of malloc.
 *
 * Enable with Q38_Q4_VICTIM_SLOTS=N.  N=0 disables it.
 * ====================================================================== */
typedef struct {
    int layer,eid;
    void *slab;
    int64_t bytes;
    uint64_t used;
} Q38Q4Victim;

static Q38Q4Victim *g_q38_q4_victim;
static int g_q38_q4_victim_cap;
static uint64_t g_q38_q4_victim_clock;
static uint64_t g_q38_q4_victim_hits;
static uint64_t g_q38_q4_victim_misses;
static uint64_t g_q38_q4_victim_puts;
static uint64_t g_q38_q4_victim_replaced;
static pthread_mutex_t g_q38_q4_victim_mx=PTHREAD_MUTEX_INITIALIZER;
static int g_q38_q4_victim_report_registered;

static void q38_q4_victim_report(void){
    if(g_q38_q4_victim_cap<=0)return;
    uint64_t lookups=g_q38_q4_victim_hits+g_q38_q4_victim_misses;
    double hitpct=lookups?100.0*(double)g_q38_q4_victim_hits/(double)lookups:0.0;
    fprintf(stderr,
        "[BABA v8 victim] slots=%d hits=%llu misses=%llu hit=%.1f%% puts=%llu replaced=%llu\n",
        g_q38_q4_victim_cap,
        (unsigned long long)g_q38_q4_victim_hits,
        (unsigned long long)g_q38_q4_victim_misses,hitpct,
        (unsigned long long)g_q38_q4_victim_puts,
        (unsigned long long)g_q38_q4_victim_replaced);
}

static int q38_q4_victim_init(void){
    if(g_q38_q4_victim)return g_q38_q4_victim_cap>0;
    const char *e=getenv("Q38_Q4_VICTIM_SLOTS");
    int cap=(e&&*e)?atoi(e):0;
    if(cap<=0){g_q38_q4_victim_cap=0;return 0;}
    if(cap>8192)cap=8192;
    g_q38_q4_victim=(Q38Q4Victim*)calloc((size_t)cap,sizeof(*g_q38_q4_victim));
    if(!g_q38_q4_victim){fprintf(stderr,"OOM Q4 victim cache metadata\n");exit(1);}
    g_q38_q4_victim_cap=cap;
    for(int i=0;i<cap;i++){g_q38_q4_victim[i].layer=-1;g_q38_q4_victim[i].eid=-1;}
    if(!g_q38_q4_victim_report_registered){
        atexit(q38_q4_victim_report);
        g_q38_q4_victim_report_registered=1;
    }
    fprintf(stderr,
        "[BABA v8 victim] ON: %d shared slots, max payload %.2f GiB (lazy)\n",
        cap,(double)cap*(double)g_q38_q4.record_bytes/1073741824.0);
    return 1;
}

static void q38_q4_victim_rebind(Model *m,Slot *s){
    if(!s->q4_slab||s->q4_slab_bytes!=(int64_t)g_q38_q4.record_bytes){
        s->q4_active=0;return;
    }
    q38_weight_free(&s->gate);
    q38_weight_free(&s->up);
    q38_weight_free(&s->down);
    free(s->fp8_slab);s->fp8_slab=NULL;s->fp8_slab_bytes=0;

    uint64_t mb=(uint64_t)m->c.hidden*m->c.inter/2;
    uint64_t sgu=(uint64_t)m->c.inter*(m->c.hidden/64)*4;
    unsigned char *p=(unsigned char*)s->q4_slab;
    s->q4.g4=p;
    s->q4.u4=p+mb;
    s->q4.d4=p+2*mb;
    s->q4.gs=(float*)(p+3*mb);
    s->q4.us=(float*)(p+3*mb+sgu);
    s->q4.ds=(float*)(p+3*mb+2*sgu);
    s->q4_active=1;
}

static void q38_q4_victim_put(Model *m,int layer,Slot *s){
    (void)m;
    if(!q38_q4_victim_init()||!s||s->eid<0||!s->q4_active||
       !s->q4_slab||s->q4_slab_bytes!=(int64_t)g_q38_q4.record_bytes)return;

    pthread_mutex_lock(&g_q38_q4_victim_mx);

    int pick=-1;
    /* Replacing the same key should be practically impossible, but if it
       happens prefer that entry over creating duplicates. */
    for(int i=0;i<g_q38_q4_victim_cap;i++)
        if(g_q38_q4_victim[i].layer==layer&&g_q38_q4_victim[i].eid==s->eid){
            pick=i;break;
        }
    if(pick<0)
        for(int i=0;i<g_q38_q4_victim_cap;i++)
            if(g_q38_q4_victim[i].layer<0){pick=i;break;}
    if(pick<0){
        pick=0;
        for(int i=1;i<g_q38_q4_victim_cap;i++)
            if(g_q38_q4_victim[i].used<g_q38_q4_victim[pick].used)pick=i;
        g_q38_q4_victim_replaced++;
    }

    Q38Q4Victim *v=&g_q38_q4_victim[pick];
    void *tmp=v->slab;int64_t tmpb=v->bytes;
    v->slab=s->q4_slab;v->bytes=s->q4_slab_bytes;
    v->layer=layer;v->eid=s->eid;v->used=++g_q38_q4_victim_clock;

    s->q4_slab=tmp;s->q4_slab_bytes=tmpb;s->q4_active=0;
    memset(&s->q4,0,sizeof(s->q4));
    g_q38_q4_victim_puts++;

    pthread_mutex_unlock(&g_q38_q4_victim_mx);
}

static int q38_q4_victim_take(Model *m,int layer,int eid,Slot *s){
    if(!q38_q4_victim_init()||!s)return 0;

    pthread_mutex_lock(&g_q38_q4_victim_mx);
    int hit=-1;
    for(int i=0;i<g_q38_q4_victim_cap;i++)
        if(g_q38_q4_victim[i].layer==layer&&g_q38_q4_victim[i].eid==eid){
            hit=i;break;
        }

    if(hit<0){
        g_q38_q4_victim_misses++;
        pthread_mutex_unlock(&g_q38_q4_victim_mx);
        return 0;
    }

    Q38Q4Victim *v=&g_q38_q4_victim[hit];
    void *tmp=s->q4_slab;int64_t tmpb=s->q4_slab_bytes;
    s->q4_slab=v->slab;s->q4_slab_bytes=v->bytes;
    v->slab=tmp;v->bytes=tmpb;
    v->layer=-1;v->eid=-1;v->used=0;
    g_q38_q4_victim_hits++;
    pthread_mutex_unlock(&g_q38_q4_victim_mx);

    q38_q4_victim_rebind(m,s);
    return s->q4_active?1:0;
}
/* ====================================================================== */


#ifdef _WIN32
#define Q38_Q4_WIN_OVERLAPPED_V2 1
static HANDLE g_q38_q4_win_handle=INVALID_HANDLE_VALUE;
/* BABA_CPU_HOTPATH_V51
 * Successful Windows overlapped Q4 batch-size histogram.
 * Fresh process per benchmark run => counters naturally reset. */
static uint64_t g_q38_v51_q4_batches;
static uint64_t g_q38_v51_q4_jobs;
static uint64_t g_q38_v51_q4_hist[Q38_MAX_TOPK+1];
/* BABA_CPU_HOTPATH_V6
 * Q38_Q4_IO_MODE:
 *   base      = existing one-ReadFile-per-missing-expert
 *   sorted    = same reads, submitted in ascending file offset
 *   coalesce0 = consecutive expert records become one larger ReadFile
 *   coalesce1 = also bridge ONE missing expert record between requested records
 *               (reads a little extra data to reduce random I/O request count)
 */
static uint64_t g_q38_v6_groups;
static uint64_t g_q38_v6_requested_records;
static uint64_t g_q38_v6_physical_records;
static uint64_t g_q38_v6_copied_bytes;
static uint64_t g_q38_v6_group_hist[Q38_MAX_TOPK+1];
#ifdef _WIN32
static HANDLE g_q38_v7_event_pool[Q38_MAX_TOPK];
static int q38_v7_event_pool_init(void){
    for(int i=0;i<Q38_MAX_TOPK;i++){
        if(!g_q38_v7_event_pool[i]){
            g_q38_v7_event_pool[i]=CreateEventA(NULL,TRUE,FALSE,NULL);
            if(!g_q38_v7_event_pool[i])return 0;
        }
    }
    return 1;
}
static int q38_v7_event_pool_on(void){
    const char *e=getenv("Q38_MEGA_EVENT_POOL");
    return e&&e[0]=='1'&&!e[1];
}
static uint64_t g_q38_v7_event_reuses;
#endif


static int q38_q4_io_mode_v6(void){
    const char *e=getenv("Q38_Q4_IO_MODE");
    if(!e||!*e||!strcmp(e,"base"))return 0;
    if(!strcmp(e,"sorted"))return 1;
    if(!strcmp(e,"coalesce0"))return 2;
    if(!strcmp(e,"coalesce1"))return 3;
    return 0;
}



static int q38_q4_win_open_v8(void){
    if(g_q38_q4_win_handle!=INVALID_HANDLE_VALUE)return 1;
    const char *path=getenv("Q38_Q4_SIDECAR");
    if(!path||!*path)return 0;
    g_q38_q4_win_handle=CreateFileA(
        path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL|FILE_FLAG_RANDOM_ACCESS|FILE_FLAG_OVERLAPPED,NULL);
    if(g_q38_q4_win_handle==INVALID_HANDLE_VALUE){
        fprintf(stderr,"[Q4 OVERLAPPED] CreateFile failed: %lu\n",
                (unsigned long)GetLastError());
        return 0;
    }
    fprintf(stderr,"[Q4 OVERLAPPED] true Windows overlapped Q4 batch I/O enabled\n");
    return 1;
}
#endif

static uint32_t q38_u32le_v7(const unsigned char *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static uint64_t q38_u64le_v7(const unsigned char *p){uint64_t v=0;for(int i=7;i>=0;i--)v=(v<<8)|p[i];return v;}
static int q38_pread_all_v7(int fd,void *dst,size_t n,off_t off){
    unsigned char *p=(unsigned char*)dst;
    while(n){ssize_t r=pread(fd,p,n,off);if(r<0&&errno==EINTR)continue;if(r<=0)return 0;p+=r;n-=(size_t)r;off+=(off_t)r;}
    return 1;
}
static int q38_q4_init_v7(Model *m){
    if(g_q38_q4.fd!=-2)return g_q38_q4.ready;
    const char *path=getenv("Q38_Q4_SIDECAR");
    if(!path||!*path){g_q38_q4.fd=-1;return 0;}
    int fd=open(path,O_RDONLY); if(fd<0){fprintf(stderr,"[BABA v7] cannot open %s: %s\\n",path,strerror(errno));g_q38_q4.fd=-1;return 0;}
    unsigned char h[64];
    if(!q38_pread_all_v7(fd,h,64,0)||memcmp(h,"Q38Q4G1",8)){fprintf(stderr,"[BABA v7] bad sidecar header\\n");close(fd);g_q38_q4.fd=-1;return 0;}
    uint32_t ver=q38_u32le_v7(h+8),L=q38_u32le_v7(h+12),E=q38_u32le_v7(h+16),H=q38_u32le_v7(h+20),F=q38_u32le_v7(h+24),G=q38_u32le_v7(h+28);
    uint64_t rec=q38_u64le_v7(h+36),mb=(uint64_t)H*F/2,expect=3*mb+(uint64_t)2*F*(H/64)*4+(uint64_t)H*(F/64)*4;
    if(ver!=1||L!=(uint32_t)m->c.layers||E!=(uint32_t)m->c.experts||H!=(uint32_t)m->c.hidden||F!=(uint32_t)m->c.inter||G!=64||rec!=expect){
        fprintf(stderr,"[BABA v7] sidecar geometry mismatch\\n");close(fd);g_q38_q4.fd=-1;return 0;
    }
    g_q38_q4=(Q38Q4Sidecar){fd,L,E,H,F,G,rec,1};
    fprintf(stderr,"[BABA v7] Q4-GS64 ON: %.3f MiB/expert, activation=%s\\n",rec/1048576.0,
            getenv("Q38_Q4_ACT_I8")&&getenv("Q38_Q4_ACT_I8")[0]=='0'?"f32":"int8/VNNI");
    return 1;
}
static int q38_q4_on_v7(Model *m){return g_q38_q4_decode_scope&&q38_q4_init_v7(m);}
static off_t q38_q4_off_v7(int layer,int expert){return (off_t)(64ull+((uint64_t)layer*g_q38_q4.experts+(uint64_t)expert)*g_q38_q4.record_bytes);}
static int q38_q4_load_v7(Model *m,int layer,int expert,Slot *s){
    if(!q38_q4_on_v7(m))return 0;
#ifdef _WIN32
    if(q38_q4_mmap_mode_v9()&&q38_q4_mmap_open_v9()){
        unsigned char *mapped=q38_q4_mmap_record_v9(layer,expert);
        if(!mapped)return 0;

        if(q38_q4_mmap_mode_v9()>=2&&g_q38_prefetch_vm){
            Q38MemoryRangeEntry range;
            range.VirtualAddress=mapped;
            range.NumberOfBytes=(SIZE_T)g_q38_q4.record_bytes;
            if(g_q38_prefetch_vm(GetCurrentProcess(),1,&range,0)){
                g_q38_q4_mmap_prefetches++;
                g_q38_q4_mmap_prefetch_bytes+=g_q38_q4.record_bytes;
            }
        }

        q38_q4_bind_mapped_v9(m,s,mapped);
        m->expert_weight_reads++;
        m->expert_pair_reads++;
        return 1;
    }
#endif

    uint64_t rec=g_q38_q4.record_bytes;
    if((uint64_t)s->q4_slab_bytes!=rec){void *p=realloc(s->q4_slab,(size_t)rec);if(!p){fprintf(stderr,"OOM q4 slab\\n");exit(1);}s->q4_slab=p;s->q4_slab_bytes=(int64_t)rec;}
    double started=now_s();
    if(!q38_pread_all_v7(g_q38_q4.fd,s->q4_slab,(size_t)rec,q38_q4_off_v7(layer,expert))){fprintf(stderr,"Q4 short read L%d E%d\\n",layer,expert);exit(1);}
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);
    q38_weight_free(&s->gate);q38_weight_free(&s->up);q38_weight_free(&s->down);
    free(s->fp8_slab);s->fp8_slab=NULL;s->fp8_slab_bytes=0;
    uint64_t mb=(uint64_t)m->c.hidden*m->c.inter/2,sgu=(uint64_t)m->c.inter*(m->c.hidden/64)*4;
    unsigned char *p=(unsigned char*)s->q4_slab;
    s->q4.g4=p;s->q4.u4=p+mb;s->q4.d4=p+2*mb;
    s->q4.gs=(float*)(p+3*mb);s->q4.us=(float*)(p+3*mb+sgu);s->q4.ds=(float*)(p+3*mb+2*sgu);
    s->q4_active=1;m->expert_weight_reads++;m->expert_pair_reads++;return 1;
}


static void q38_q4_prefetch_v7(Model *m,int layer,const int *experts,int count){
    if(!m->expert_prefetch||!q38_q4_on_v7(m)||!experts)return;
#ifdef _WIN32
    /* compat_fadvise(WILLNEED) is a blocking ReadFile on the normal model fd.
     * Never duplicate the Q4 read on the Windows hot path unless explicitly
     * requested for diagnosis. */
    const char *bp=getenv("Q38_WIN_Q4_BLOCKING_PREFETCH");
    if(!(bp&&bp[0]=='1'&&bp[1]=='\0'))return;
#endif
    LCache *cache=&m->cache[layer];
    for(int i=0;i<count;i++){int e=experts[i];if(e<0||e>=m->c.experts||cache->by_expert[e]>=0)continue;
        posix_fadvise(g_q38_q4.fd,q38_q4_off_v7(layer,e),(off_t)g_q38_q4.record_bytes,POSIX_FADV_WILLNEED);m->expert_prefetch_ranges++;}
}
static void q38_q4_reset_layer_v7(Model *m,int layer){
    if(g_q38_q4_reset[layer])return;LCache *lc=&m->cache[layer];
    for(int i=0;i<lc->n;i++){Slot *s=&lc->slots[i];if(s->eid>=0&&s->eid<m->c.experts)lc->by_expert[s->eid]=-1;
        q38_weight_free(&s->gate);q38_weight_free(&s->up);q38_weight_free(&s->down);free(s->fp8_slab);s->fp8_slab=NULL;s->fp8_slab_bytes=0;
        free(s->q4_slab);s->q4_slab=NULL;s->q4_slab_bytes=0;s->q4_active=0;s->eid=-1;s->used=0;}
    lc->n=0;g_q38_q4_reset[layer]=1;
}
static int q38_q4_act_mode_v7(void){const char *e=getenv("Q38_Q4_ACT_I8");return !(e&&e[0]=='0'&&e[1]=='\0');}
/* ====================================================================== */
static int q38_native_fp8_expert_tensors(Model *m,int layer,int expert,
                                         st_tensor *weight[3]) {
    if(!m->native_fp8)return 0;
    Cfg *c=&m->c;const char *projection[3]={"gate_proj","up_proj","down_proj"};
    int rows[3]={c->inter,c->inter,c->hidden};
    int cols[3]={c->hidden,c->hidden,c->inter};
    char suffix[192],name[320];
    for(int kind=0;kind<3;kind++){
        int length=snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight",
                            expert,projection[kind]);
        if(length<0||(size_t)length>=sizeof suffix)return 0;
        q38_name(m,name,sizeof name,layer,suffix);weight[kind]=st_find(&m->S,name);
        if(!weight[kind]||weight[kind]->dtype!=4||weight[kind]->rank!=2||
           weight[kind]->shape[0]!=rows[kind]||weight[kind]->shape[1]!=cols[kind]||
           weight[kind]->nbytes!=(int64_t)rows[kind]*cols[kind])return 0;
    }
    if(weight[0]->fd!=weight[1]->fd||
       weight[0]->off>INT64_MAX-weight[0]->nbytes||
       weight[1]->off!=weight[0]->off+weight[0]->nbytes)return 0;
    return 1;
}

static void q38_load_native_fp8_ranges(Model *m,int layer,int expert,Slot *slot,
                                       st_tensor *weight[3]) {
    Cfg *c=&m->c;
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    float *scales=cache->values+(int64_t)expert*3*cache->scale_count;
    /* Se i tre intervalli sono mappati (COLI_MAP_EXPERTS=1), lo slot li PUNTA
     * invece di copiarli: niente slab, niente 14 MB per miss. */
    {
        const uint8_t *pg=(const uint8_t*)st_map_shard_range(weight[0]->fd,weight[0]->off,weight[0]->nbytes);
        const uint8_t *pu=(const uint8_t*)st_map_shard_range(weight[1]->fd,weight[1]->off,weight[1]->nbytes);
        const uint8_t *pd=(const uint8_t*)st_map_shard_range(weight[2]->fd,weight[2]->off,weight[2]->nbytes);
        if(pg&&pu&&pd){
            int sc=(int)cache->scale_count;
            q38_bind_borrowed_fp8(&slot->gate,(void*)pg,scales,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->up,(void*)pu,scales+sc,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->down,(void*)pd,scales+2*sc,c->hidden,c->inter);
            if(slot->fp8_slab){free(slot->fp8_slab);slot->fp8_slab=NULL;slot->fp8_slab_bytes=0;}
            return;
        }
    }
    q38_bind_fp8_slot(slot,scales,(int)cache->scale_count,c->hidden,c->inter);
    int64_t pair_bytes=weight[0]->nbytes+weight[1]->nbytes;
    unsigned char *raw=(unsigned char*)slot->fp8_slab;
    st_read_range_raw_cap(&m->S,weight[0]->fd,weight[0]->off,pair_bytes,
                          raw,pair_bytes,1,"pread Qwen3.8 gate/up expert");
    st_read_range_raw_cap(&m->S,weight[2]->fd,weight[2]->off,weight[2]->nbytes,
                          raw+pair_bytes,slot->fp8_slab_bytes-pair_bytes,1,
                          "pread Qwen3.8 down expert");
}

static int q38_try_load_native_fp8_expert(Model *m,int layer,int expert,Slot *slot) {
    st_tensor *weight[3];
    if(!q38_native_fp8_expert_tensors(m,layer,expert,weight)||
       !q38_prepare_expert_scale_bank(m,layer))return 0;
    double started=now_s();
    q38_load_native_fp8_ranges(m,layer,expert,slot,weight);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);
    m->expert_weight_reads+=2;m->expert_pair_reads++;
    return 1;
}

static void q38_prefetch_native_fp8_experts(Model *m,int layer,
                                            const int *experts,int count) {
    if(q38_q4_on_v7(m)){q38_q4_prefetch_v7(m,layer,experts,count);return;}
    if(!m->expert_prefetch||!m->native_fp8||!experts||count<1||
       !q38_prepare_expert_scale_bank(m,layer))return;
    LCache *cache=&m->cache[layer];
    for(int index=0;index<count;index++){
        int expert=experts[index];st_tensor *weight[3];
        if(expert<0||expert>=m->c.experts||cache->by_expert[expert]>=0||
           !q38_native_fp8_expert_tensors(m,layer,expert,weight))continue;
        posix_fadvise(weight[0]->fd,weight[0]->off,
                      weight[0]->nbytes+weight[1]->nbytes,POSIX_FADV_WILLNEED);
        posix_fadvise(weight[2]->fd,weight[2]->off,weight[2]->nbytes,
                      POSIX_FADV_WILLNEED);
        m->expert_prefetch_ranges+=2;
    }
}

static void q38_load_fp8_expert_weight(Model *m,const char *wn,const char *sn,
                                        Q38Weight *out,int O,int I) {
    st_tensor *w=st_find(&m->S,wn),*sc=st_find(&m->S,sn);
    int nb_o=(O+127)/128,nb_i=(I+127)/128;
    if(!w||w->dtype!=4||w->rank!=2||w->shape[0]!=O||w->shape[1]!=I||
       w->nbytes!=(int64_t)O*I||!sc||sc->dtype<0||sc->dtype>2||
       sc->rank!=2||sc->shape[0]!=nb_o||sc->shape[1]!=nb_i||
       sc->numel!=(int64_t)nb_o*nb_i){
        fprintf(stderr,"invalid block-FP8 expert matrix %s / %s\n",wn,sn);exit(1);
    }
    if(m->native_fp8){
        q38_weight_reserve(out,Q38_WEIGHT_FP8,O,I);
        double started=now_s();
        st_read_raw_cap(&m->S,wn,out->data,w->nbytes,1);
        st_read_f32(&m->S,sn,out->scales,1);
        q38_tm_add(m,Q38_TM_EXPERT_READ,started);
        m->expert_weight_reads++;m->expert_scale_reads++;
        return;
    }
    q38_weight_reserve(out,Q38_WEIGHT_F32,O,I);
    uint8_t *raw=(uint8_t*)malloc((size_t)w->nbytes);float *scale=falloc(sc->numel);
    if(!raw){fprintf(stderr,"OOM FP8 expert staging\n");exit(1);}
    double started=now_s();
    st_read_raw_cap(&m->S,wn,raw,w->nbytes,1);st_read_f32(&m->S,sn,scale,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);
    m->expert_weight_reads++;m->expert_scale_reads++;started=now_s();
    float *decoded=(float*)out->data;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++)for(int i=0;i<I;i++)
        decoded[(int64_t)o*I+i]=e4m3_decode(raw[(int64_t)o*I+i])*
                                  scale[(o/128)*nb_i+i/128];
    q38_tm_add(m,Q38_TM_FP8_EXPAND,started);
    free(raw);free(scale);
}

static void q38_load_expert_weight(Model *m,const char *name,Q38Weight *out,
                                    int O,int I) {
    st_tensor *tensor=st_find(&m->S,name);
    if(!tensor||tensor->rank!=2||tensor->shape[0]!=O||tensor->shape[1]!=I||
       tensor->numel!=(int64_t)O*I||tensor->dtype<0||tensor->dtype>2){
        fprintf(stderr,"invalid expert matrix %s\n",name);exit(1);
    }
    Q38WeightKind kind=m->native_bf16&&tensor->dtype==0?Q38_WEIGHT_BF16:Q38_WEIGHT_F32;
    q38_weight_reserve(out,kind,O,I);double started=now_s();
    if(kind==Q38_WEIGHT_BF16)
        st_read_raw_cap(&m->S,name,out->data,tensor->nbytes,1);
    else st_read_f32(&m->S,name,(float*)out->data,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_weight_reads++;
}

static void q38_load_expert_slice(Model *m,const char *name,const st_tensor *tensor,
                                   int64_t element_offset,Q38Weight *out,int O,int I) {
    if(!tensor||tensor->dtype<0||tensor->dtype>2||element_offset<0||
       element_offset>tensor->numel-(int64_t)O*I){
        fprintf(stderr,"invalid expert slice %s\n",name);exit(1);
    }
    Q38WeightKind kind=m->native_bf16&&tensor->dtype==0?Q38_WEIGHT_BF16:Q38_WEIGHT_F32;
    q38_weight_reserve(out,kind,O,I);double started=now_s();
    if(kind==Q38_WEIGHT_BF16){
        int64_t bytes=(int64_t)O*I*(int64_t)sizeof(uint16_t);
        st_read_slice_raw_cap(&m->S,name,element_offset*(int64_t)sizeof(uint16_t),
                              bytes,out->data,bytes,1);
    } else st_read_slice_f32(&m->S,name,element_offset,(int64_t)O*I,(float*)out->data,1);
    q38_tm_add(m,Q38_TM_EXPERT_READ,started);m->expert_weight_reads++;
}

static void q38_load_expert(Model *m,int layer,int eid,Slot *s) {
    if(q38_q4_load_v7(m,layer,eid,s))return;
    Cfg *c=&m->c; int H=c->hidden,I=c->inter; char nm[320],sn[340];
    q38_name(m,nm,sizeof nm,layer,"mlp.experts.gate_up_proj");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=2*I||t->shape[2]!=H){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        int64_t base=(int64_t)eid*2*I*H;
        q38_load_expert_slice(m,nm,t,base,&s->gate,I,H);
        q38_load_expert_slice(m,nm,t,base+(int64_t)I*H,&s->up,I,H);
        q38_name(m,nm,sizeof nm,layer,"mlp.experts.down_proj"); t=st_find(&m->S,nm);
        if(!t||t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=H||t->shape[2]!=I){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        q38_load_expert_slice(m,nm,t,(int64_t)eid*H*I,&s->down,H,I);
        return;
    }
    if(q38_try_load_native_fp8_expert(m,layer,eid,s))return;
    const char *kind[3]={"gate_proj","up_proj","down_proj"};Q38Weight *dst[3]={&s->gate,&s->up,&s->down};
    int os[3]={I,I,H},is[3]={H,H,I};
    for(int k=0;k<3;k++){
        char suf[192]; snprintf(suf,sizeof suf,"mlp.experts.%d.%s.weight",eid,kind[k]);q38_name(m,nm,sizeof nm,layer,suf);
        st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing %s\n",nm);exit(1);}
        if(t->dtype==4){snprintf(sn,sizeof sn,"%s_scale_inv",nm);q38_load_fp8_expert_weight(m,nm,sn,dst[k],os[k],is[k]);}
        else q38_load_expert_weight(m,nm,dst[k],os[k],is[k]);
    }
}

/* One byte per expert: routed in this turn or not. The dashboard's Brain tab
 * reads it as the HITS bitmap after every turn (serve_hits in qwen38.c),
 * cleared there. Marked on both lookup paths, single and batched. */
static pthread_mutex_t g_q38_ehit_mx=PTHREAD_MUTEX_INITIALIZER;
static void q38_ehit_mark(Model *m,int layer,int eid) {
    const Cfg *c=&m->c;
    /* The first touch can come from a parallel region (qwen36: the tier
     * warmstart's omp loop calls expert_get from twelve threads at once):
     * one thread published m->ehit while it was still filling the rows and
     * a sibling dereferenced m->ehit[layer] == NULL -- SIGSEGV in about one
     * run in twelve on a CUDA warmstart. Build the table privately, publish
     * it once under a lock (double-checked), and read it with acquire order. */
    uint8_t **ehit=__atomic_load_n(&m->ehit,__ATOMIC_ACQUIRE);
    if(!ehit){
        pthread_mutex_lock(&g_q38_ehit_mx);
        ehit=m->ehit;
        if(!ehit){
            ehit=(uint8_t**)calloc((size_t)c->layers,sizeof(uint8_t*));
            for(int i=0;i<c->layers;i++)ehit[i]=(uint8_t*)calloc((size_t)c->experts,1);
            __atomic_store_n(&m->ehit,ehit,__ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&g_q38_ehit_mx);
    }
    if(layer>=0&&layer<c->layers&&eid>=0&&eid<c->experts)ehit[layer][eid]=1;
}
static int q38_cache_lfu_on(void){
    const char *e=getenv("Q38_CACHE_POLICY");
    return e&&!strcmp(e,"lfu");
}
static void q38_cache_touch(LCache *lc,int experts,int eid){
    if(!lc||!lc->freq||eid<0||eid>=experts)return;
    if(lc->freq[eid]!=UINT16_MAX)lc->freq[eid]++;
    lc->freq_touches++;
    /* Aging prevents an early prompt from pinning the cache forever. */
    if((lc->freq_touches & 4095u)==0)
        for(int e=0;e<experts;e++)lc->freq[e]=(uint16_t)((lc->freq[e]+1u)>>1);
}
static int q38_cache_victim(const LCache *lc,const unsigned char *protected_slots){
    int victim=-1;
    for(int i=0;i<lc->n;i++){
        if(protected_slots&&protected_slots[i])continue;
        if(victim<0){victim=i;continue;}
        if(q38_cache_lfu_on()){
            int ea=lc->slots[i].eid, eb=lc->slots[victim].eid;
            uint16_t fa=(ea>=0&&lc->freq)?lc->freq[ea]:0;
            uint16_t fb=(eb>=0&&lc->freq)?lc->freq[eb]:0;
            if(fa<fb || (fa==fb&&lc->slots[i].used<lc->slots[victim].used))
                victim=i;
        }else if(lc->slots[i].used<lc->slots[victim].used) victim=i;
    }
    return victim;
}
static Slot *q38_expert_get(Model *m,int layer,int eid) {
    q38_ehit_mark(m,layer,eid);
    LCache *lc=&m->cache[layer]; int si=lc->by_expert[eid];
    if(si>=0){m->hits++;if(q38_q4_on_v7(m))m->q4_cache_hits++;q38_cache_touch(lc,m->c.experts,eid);lc->slots[si].used=++m->clock;return &lc->slots[si];}
    m->miss++;if(q38_q4_on_v7(m))m->q4_cache_misses++;q38_cache_touch(lc,m->c.experts,eid); Slot *s;
    if(lc->n<lc->cap){s=&lc->slots[lc->n++];s->eid=-1;}
    else {
        int victim=q38_cache_victim(lc,NULL);if(victim<0){fprintf(stderr,"Q38 cache has no victim\n");exit(1);}
        s=&lc->slots[victim];if(s->eid>=0)lc->by_expert[s->eid]=-1;
    }
    s->eid=-1;q38_load_expert(m,layer,eid,s);s->eid=eid;s->used=++m->clock;lc->by_expert[eid]=(int)(s-lc->slots);return s;
}

typedef struct {
    int expert;
    Slot *slot;
    st_tensor *weight[3];
    int q4;
} Q38ExpertLoadJob;

/* Reserve an entire routed demand set on the main thread, fill distinct slots
 * in parallel, then publish every cache index on the main thread.  Demand-set
 * residents are protected from victim selection, so no worker can overwrite a
 * slot another selected expert will consume.  Smaller caches and heterogeneous
 * layouts retain the serial LRU path. */
/* Says once per model why the parallel read path is not taken, so a slow
 * prefill on a small cache or a converted container is not a mystery. */
static int q38_expert_batch_fallback(Model *m,Q38ExpertBatchFallback reason,
                                     int layer,int first,int second) {
    if(m->expert_batch_fallback!=Q38_EXPERT_BATCH_FALLBACK_NONE)return 0;
    m->expert_batch_fallback=reason;
    fprintf(stderr,"[qwen38 expert I/O] parallel reads unavailable: ");
    switch(reason){
    case Q38_EXPERT_BATCH_FALLBACK_DISABLED:
        fprintf(stderr,"Q38_EXPERT_PARALLEL_READS=0");break;
    case Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY:
        fprintf(stderr,"cache holds %d experts/layer but the route needs %d; "
                       "lower --ctx/Q38_MAXT or raise --ram",first,second);break;
    case Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK:
        fprintf(stderr,"layer %d has no compatible resident FP8 scale bank",layer);break;
    case Q38_EXPERT_BATCH_FALLBACK_DUPLICATE:
        fprintf(stderr,"layer %d route repeats expert %d",layer,first);break;
    case Q38_EXPERT_BATCH_FALLBACK_LAYOUT:
        fprintf(stderr,"layer %d expert %d is not native block-FP8",layer,first);break;
    default:
        fprintf(stderr,"unknown reason");break;
    }
    fprintf(stderr,"; using serial expert reads\n");
    return 0;
}

#ifdef _WIN32
static void q38_q4_bind_slot_v8(Model *m,Slot *s){
    uint64_t rec=g_q38_q4.record_bytes;
    if((uint64_t)s->q4_slab_bytes!=rec){
        void *mem=realloc(s->q4_slab,(size_t)rec);
        if(!mem){fprintf(stderr,"OOM q4 slab\n");exit(1);}
        s->q4_slab=mem;
        s->q4_slab_bytes=(int64_t)rec;
    }
    q38_weight_free(&s->gate);
    q38_weight_free(&s->up);
    q38_weight_free(&s->down);
    free(s->fp8_slab);
    s->fp8_slab=NULL;
    s->fp8_slab_bytes=0;

    uint64_t mb=(uint64_t)m->c.hidden*m->c.inter/2;
    uint64_t sgu=(uint64_t)m->c.inter*(m->c.hidden/64)*4;
    unsigned char *p=(unsigned char*)s->q4_slab;
    s->q4.g4=p;
    s->q4.u4=p+mb;
    s->q4.d4=p+2*mb;
    s->q4.gs=(float*)(p+3*mb);
    s->q4.us=(float*)(p+3*mb+sgu);
    s->q4.ds=(float*)(p+3*mb+2*sgu);
}

/* Submit every missing Q4 expert in one real Windows overlapped I/O wave.
 * The destination is the final cache slot. */
typedef struct {
    int job_index;
    int expert;
} Q38V6Order;

typedef struct {
    int first_order;
    int order_count;       /* number of requested experts in group */
    int first_expert;
    int last_expert;
    unsigned char *buffer; /* only coalesced modes */
    OVERLAPPED ov;
    HANDLE event;
} Q38V6Group;

static int q38_v6_order_cmp(const void *aa,const void *bb){
    const Q38V6Order *a=(const Q38V6Order*)aa,*b=(const Q38V6Order*)bb;
    return a->expert<b->expert?-1:a->expert>b->expert?1:0;
}

/* Windows Q4 batch I/O v6.
 * BASE preserves the old behavior exactly.
 * SORTED changes only submission order.
 * COALESCE modes submit larger sequential ranges into temporary group buffers,
 * then scatter each requested record into its already-reserved final cache slot. */
static int q38_q4_load_batch_win_v8(Model *m,int layer,
                                     Q38ExpertLoadJob *jobs,int n){
    /* BABA v9 mmap fast path: bind missing experts directly to mapped records. */
    if(q38_q4_mmap_mode_v9()&&q38_q4_mmap_open_v9()){
        if(n<1||n>Q38_MAX_TOPK)return 0;

        if(q38_q4_mmap_mode_v9()>=2&&g_q38_prefetch_vm){
            Q38MemoryRangeEntry ranges[Q38_MAX_TOPK];
            ULONG_PTR range_count=0;
            for(int i=0;i<n&&range_count<Q38_MAX_TOPK;i++){
                unsigned char *rp=q38_q4_mmap_record_v9(layer,jobs[i].expert);
                if(!rp)continue;
                ranges[range_count].VirtualAddress=rp;
                ranges[range_count].NumberOfBytes=(SIZE_T)g_q38_q4.record_bytes;
                range_count++;
            }
            if(range_count&&g_q38_prefetch_vm(
                    GetCurrentProcess(),range_count,ranges,0)){
                g_q38_q4_mmap_prefetches++;
                g_q38_q4_mmap_prefetch_bytes+=
                    (uint64_t)range_count*g_q38_q4.record_bytes;
            }
        }

        for(int i=0;i<n;i++){
            unsigned char *rp=q38_q4_mmap_record_v9(layer,jobs[i].expert);
            if(!rp)return 0;
            q38_q4_bind_mapped_v9(m,jobs[i].slot,rp);
        }
        return 1;
    }

    if(n<1||n>Q38_MAX_TOPK||!q38_q4_win_open_v8())return 0;

    int mode=q38_q4_io_mode_v6();

    /* -------- exact legacy path -------- */
    if(mode==0){
        OVERLAPPED ov[Q38_MAX_TOPK];
        HANDLE events[Q38_MAX_TOPK];
        memset(ov,0,sizeof(ov));memset(events,0,sizeof(events));
        int use_pool=q38_v7_event_pool_on()&&q38_v7_event_pool_init();
        int submitted=0;

        for(int i=0;i<n;i++){
            Slot *slot=jobs[i].slot;
            q38_q4_bind_slot_v8(m,slot);

            uint64_t off=(uint64_t)q38_q4_off_v7(layer,jobs[i].expert);
            ov[i].Offset=(DWORD)(off&0xffffffffULL);
            ov[i].OffsetHigh=(DWORD)(off>>32);
            if(use_pool){
                events[i]=g_q38_v7_event_pool[i];
                ResetEvent(events[i]);
                g_q38_v7_event_reuses++;
            }else{
                events[i]=CreateEventA(NULL,TRUE,FALSE,NULL);
                if(!events[i])goto legacy_fail;
            }
            ov[i].hEvent=events[i];

            DWORD got=0;
            BOOL ok=ReadFile(g_q38_q4_win_handle,slot->q4_slab,
                             (DWORD)g_q38_q4.record_bytes,&got,&ov[i]);
            if(!ok&&GetLastError()!=ERROR_IO_PENDING)goto legacy_fail;
            submitted++;
        }

        if(WaitForMultipleObjects((DWORD)submitted,events,TRUE,INFINITE)==WAIT_FAILED)
            goto legacy_fail;

        for(int i=0;i<submitted;i++){
            DWORD got=0;
            if(!GetOverlappedResult(g_q38_q4_win_handle,&ov[i],&got,FALSE)||
               got!=(DWORD)g_q38_q4.record_bytes)goto legacy_fail;
            jobs[i].slot->q4_active=1;
            if(!use_pool)CloseHandle(events[i]);
            events[i]=NULL;
        }

        g_q38_v51_q4_batches++;
        g_q38_v51_q4_jobs+=(uint64_t)submitted;
        if(submitted<=Q38_MAX_TOPK)g_q38_v51_q4_hist[submitted]++;

        g_q38_v6_groups+=(uint64_t)submitted;
        g_q38_v6_requested_records+=(uint64_t)submitted;
        g_q38_v6_physical_records+=(uint64_t)submitted;
        g_q38_v6_group_hist[1]+=(uint64_t)submitted;
        return 1;

legacy_fail:
        for(int i=0;i<submitted;i++){
            CancelIoEx(g_q38_q4_win_handle,&ov[i]);
            if(events[i]){
                WaitForSingleObject(events[i],INFINITE);
                if(!use_pool)CloseHandle(events[i]);
            }
        }
        return 0;
    }

    /* -------- sorted/coalesced scheduler -------- */
    Q38V6Order order[Q38_MAX_TOPK];
    for(int i=0;i<n;i++){
        q38_q4_bind_slot_v8(m,jobs[i].slot);
        order[i].job_index=i;
        order[i].expert=jobs[i].expert;
    }
    qsort(order,(size_t)n,sizeof(order[0]),q38_v6_order_cmp);

    if(mode==1){
        OVERLAPPED ov[Q38_MAX_TOPK];
        HANDLE events[Q38_MAX_TOPK];
        memset(ov,0,sizeof(ov));memset(events,0,sizeof(events));
        int submitted=0;

        for(int oi=0;oi<n;oi++){
            int j=order[oi].job_index;
            uint64_t off=(uint64_t)q38_q4_off_v7(layer,jobs[j].expert);
            ov[oi].Offset=(DWORD)(off&0xffffffffULL);
            ov[oi].OffsetHigh=(DWORD)(off>>32);
            events[oi]=CreateEventA(NULL,TRUE,FALSE,NULL);
            if(!events[oi])goto sorted_fail;
            ov[oi].hEvent=events[oi];

            DWORD got=0;
            BOOL ok=ReadFile(g_q38_q4_win_handle,jobs[j].slot->q4_slab,
                             (DWORD)g_q38_q4.record_bytes,&got,&ov[oi]);
            if(!ok&&GetLastError()!=ERROR_IO_PENDING)goto sorted_fail;
            submitted++;
        }

        if(WaitForMultipleObjects((DWORD)submitted,events,TRUE,INFINITE)==WAIT_FAILED)
            goto sorted_fail;

        for(int oi=0;oi<submitted;oi++){
            DWORD got=0;
            if(!GetOverlappedResult(g_q38_q4_win_handle,&ov[oi],&got,FALSE)||
               got!=(DWORD)g_q38_q4.record_bytes)goto sorted_fail;
            int j=order[oi].job_index;
            jobs[j].slot->q4_active=1;
            CloseHandle(events[oi]);events[oi]=NULL;
        }

        g_q38_v51_q4_batches++;
        g_q38_v51_q4_jobs+=(uint64_t)submitted;
        if(submitted<=Q38_MAX_TOPK)g_q38_v51_q4_hist[submitted]++;

        g_q38_v6_groups+=(uint64_t)submitted;
        g_q38_v6_requested_records+=(uint64_t)submitted;
        g_q38_v6_physical_records+=(uint64_t)submitted;
        g_q38_v6_group_hist[1]+=(uint64_t)submitted;
        return 1;

sorted_fail:
        for(int i=0;i<submitted;i++){
            CancelIoEx(g_q38_q4_win_handle,&ov[i]);
            if(events[i]){
                WaitForSingleObject(events[i],INFINITE);
                CloseHandle(events[i]);
            }
        }
        return 0;
    }

    /* coalesce0: only adjacent records. coalesce1: allow one-record hole. */
    int max_gap=(mode==3)?1:0;
    Q38V6Group groups[Q38_MAX_TOPK];
    memset(groups,0,sizeof(groups));
    int ng=0;

    for(int oi=0;oi<n;){
        int first=oi,last=oi;
        int first_e=order[oi].expert,last_e=first_e;
        oi++;
        while(oi<n){
            int next_e=order[oi].expert;
            int hole=next_e-last_e-1;
            if(hole<0||hole>max_gap)break;
            last=oi;last_e=next_e;oi++;
        }
        Q38V6Group *g=&groups[ng++];
        g->first_order=first;
        g->order_count=last-first+1;
        g->first_expert=first_e;
        g->last_expert=last_e;
    }

    int submitted=0;
    for(int gi=0;gi<ng;gi++){
        Q38V6Group *g=&groups[gi];
        uint64_t physical_records=(uint64_t)(g->last_expert-g->first_expert+1);
        uint64_t bytes64=physical_records*g_q38_q4.record_bytes;
        if(bytes64>0xFFFFFFFFULL)goto coal_fail;

        g->buffer=(unsigned char*)malloc((size_t)bytes64);
        if(!g->buffer){fprintf(stderr,"OOM Q4 coalesced I/O buffer\n");goto coal_fail;}
        g->event=CreateEventA(NULL,TRUE,FALSE,NULL);
        if(!g->event)goto coal_fail;

        uint64_t off=(uint64_t)q38_q4_off_v7(layer,g->first_expert);
        g->ov.Offset=(DWORD)(off&0xffffffffULL);
        g->ov.OffsetHigh=(DWORD)(off>>32);
        g->ov.hEvent=g->event;

        DWORD got=0;
        BOOL ok=ReadFile(g_q38_q4_win_handle,g->buffer,(DWORD)bytes64,&got,&g->ov);
        if(!ok&&GetLastError()!=ERROR_IO_PENDING)goto coal_fail;
        submitted++;
    }

    {
        HANDLE events[Q38_MAX_TOPK];
        for(int gi=0;gi<submitted;gi++)events[gi]=groups[gi].event;
        if(WaitForMultipleObjects((DWORD)submitted,events,TRUE,INFINITE)==WAIT_FAILED)
            goto coal_fail;
    }

    for(int gi=0;gi<submitted;gi++){
        Q38V6Group *g=&groups[gi];
        uint64_t physical_records=(uint64_t)(g->last_expert-g->first_expert+1);
        uint64_t bytes64=physical_records*g_q38_q4.record_bytes;
        DWORD got=0;
        if(!GetOverlappedResult(g_q38_q4_win_handle,&g->ov,&got,FALSE)||
           got!=(DWORD)bytes64)goto coal_fail;

        for(int oi=g->first_order;oi<g->first_order+g->order_count;oi++){
            int j=order[oi].job_index;
            int rel=jobs[j].expert-g->first_expert;
            memcpy(jobs[j].slot->q4_slab,
                   g->buffer+(uint64_t)rel*g_q38_q4.record_bytes,
                   (size_t)g_q38_q4.record_bytes);
            jobs[j].slot->q4_active=1;
            g_q38_v6_copied_bytes+=g_q38_q4.record_bytes;
        }

        g_q38_v6_groups++;
        g_q38_v6_requested_records+=(uint64_t)g->order_count;
        g_q38_v6_physical_records+=physical_records;
        if(g->order_count<=Q38_MAX_TOPK)g_q38_v6_group_hist[g->order_count]++;

        CloseHandle(g->event);g->event=NULL;
        free(g->buffer);g->buffer=NULL;
    }

    g_q38_v51_q4_batches++;
    g_q38_v51_q4_jobs+=(uint64_t)n;
    if(n<=Q38_MAX_TOPK)g_q38_v51_q4_hist[n]++;
    return 1;

coal_fail:
    for(int gi=0;gi<ng;gi++){
        if(gi<submitted)CancelIoEx(g_q38_q4_win_handle,&groups[gi].ov);
        if(groups[gi].event){
            if(gi<submitted)WaitForSingleObject(groups[gi].event,INFINITE);
            CloseHandle(groups[gi].event);
        }
        free(groups[gi].buffer);
    }
    return 0;
}
#endif

static int q38_expert_get_batch(Model *m,int layer,const int *experts,int count,
                                Slot **selected) {
    if(!experts||!selected||count<2)return 0;
    if(!m->expert_parallel_reads)
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_DISABLED,layer,0,0);
    LCache *cache=&m->cache[layer];
    int q4=q38_q4_on_v7(m);
    if(count>cache->cap)
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY,
                                         layer,cache->cap,count);
    if(!q4&&!q38_prepare_expert_scale_bank(m,layer))
        return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_SCALE_BANK,layer,0,0);
    /* The demand set is no longer bounded by the decode top-k: the MoE prefill
     * hands over the whole chunk union (up to the cache cap) so its loads run
     * one OMP wave instead of serial groups of Q38_MAX_TOPK.  Load grouping
     * never touches FP order: routed outputs are written per assignment and
     * the per-position expert sum follows the router order, so a bigger wave
     * only changes WHICH slots serve the reads, not the arithmetic. */
    Q38ExpertLoadJob *jobs=malloc((size_t)count*sizeof(*jobs));
    if(!jobs)return 0;
    for(int index=0;index<count;index++){
        int expert=experts[index];
        if(expert<0||expert>=m->c.experts){free(jobs);return 0;}
        q38_ehit_mark(m,layer,expert);
        for(int previous=0;previous<index;previous++)
            if(experts[previous]==expert){
                free(jobs);
                return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_DUPLICATE,
                                                 layer,expert,0);
            }
        int slot_index=cache->by_expert[expert];
        if(slot_index>=0){
            if(slot_index>=cache->n||cache->slots[slot_index].eid!=expert){free(jobs);return 0;}
            continue;
        }
        if(!q4){st_tensor *weight[3];
            if(!q38_native_fp8_expert_tensors(m,layer,expert,weight)){
                free(jobs);return q38_expert_batch_fallback(m,Q38_EXPERT_BATCH_FALLBACK_LAYOUT,layer,expert,0);
            }}
    }
    unsigned char *protected_slots=(unsigned char*)calloc((size_t)cache->cap,1);
    if(!protected_slots){fprintf(stderr,"OOM expert batch reservations\n");exit(1);}
    for(int index=0;index<count;index++){
        int slot_index=cache->by_expert[experts[index]];
        if(slot_index>=0)protected_slots[slot_index]=1;
    }
    int job_count=0;
    for(int index=0;index<count;index++){
        int expert=experts[index],slot_index=cache->by_expert[expert];Slot *slot;
        if(slot_index>=0){
            slot=&cache->slots[slot_index];m->hits++;if(q4)m->q4_cache_hits++;q38_cache_touch(cache,m->c.experts,expert);
        }else{
            m->miss++;if(q4)m->q4_cache_misses++;q38_cache_touch(cache,m->c.experts,expert);
            if(cache->n<cache->cap){
                slot=&cache->slots[cache->n++];slot->eid=-1;
            }else{
                int victim=q38_cache_victim(cache,protected_slots);
                if(victim<0){
                    fprintf(stderr,"Qwen3.8 expert demand set has no reservable cache slot\n");
                    exit(1);
                }
                slot=&cache->slots[victim];
                if(slot->eid>=0){
                    cache->by_expert[slot->eid]=-1;
                    if(q4)q38_q4_victim_put(m,layer,slot);
                }
            }
            slot->eid=-1;
            if(q4&&q38_q4_victim_take(m,layer,expert,slot)){
                /* L2 victim hit: publish immediately; no sidecar I/O job. */
                slot->eid=expert;
                cache->by_expert[expert]=(int)(slot-cache->slots);
            }else{
                jobs[job_count].expert=expert;jobs[job_count].slot=slot;jobs[job_count].q4=q4;
                if(!q4&&!q38_native_fp8_expert_tensors(m,layer,expert,jobs[job_count].weight)){
                    fprintf(stderr,"Qwen3.8 expert layout changed during batch reservation\n");exit(1);
                }
                job_count++;
            }
        }
        slot->used=++m->clock;selected[index]=slot;
        protected_slots[slot-cache->slots]=1;
    }
    free(protected_slots);
    if(job_count){
        int workers=job_count;
#ifdef _OPENMP
        int thread_limit=omp_get_max_threads();if(workers>thread_limit)workers=thread_limit;
#else
        (void)workers;   /* only the pragma below reads it, and that is gone without OpenMP */
#endif
        double started=now_s();
#ifdef _WIN32
        if(q4&&q38_q4_load_batch_win_v8(m,layer,jobs,job_count)){
            q38_tm_add(m,Q38_TM_EXPERT_READ,started);
            m->expert_weight_reads+=(uint64_t)job_count;
            m->expert_pair_reads+=(uint64_t)job_count;
        }else
#endif
        {
            #pragma omp parallel for schedule(static) num_threads(workers) if(job_count>1)
            for(int job=0;job<job_count;job++){
                if(jobs[job].q4)q38_q4_load_v7(m,layer,jobs[job].expert,jobs[job].slot);
                else q38_load_native_fp8_ranges(m,layer,jobs[job].expert,jobs[job].slot,jobs[job].weight);
            }
            if(!q4){
                q38_tm_add(m,Q38_TM_EXPERT_READ,started);
                m->expert_weight_reads+=(uint64_t)job_count*2;
                m->expert_pair_reads+=(uint64_t)job_count;
            }
        }
        if(job_count>1)m->expert_parallel_batches++;
        for(int job=0;job<job_count;job++){
            Slot *slot=jobs[job].slot;slot->eid=jobs[job].expert;
            cache->by_expert[jobs[job].expert]=(int)(slot-cache->slots);
        }
    }
    free(jobs);
    return 1;
}

static void *q38_async_expert_load_main(void *arg){
    Q38AsyncExpertLoad *a=(Q38AsyncExpertLoad*)arg;
    double t0=now_s();
    a->result=q38_expert_get_batch(a->m,a->layer,a->experts,a->count,a->selected);
    g_q38_v4_async_wall_s+=now_s()-t0;
    return NULL;
}
static void q38_ple_row(Model *m,int64_t row,float *out) {
    Cfg *c=&m->c; int p=0;
    while(p+1<m->ple_part_count&&row>=m->ple_part_start[p+1])p++;
    if(p>=m->ple_part_count||row<m->ple_part_start[p]){fprintf(stderr,"PLE row out of range: %lld\n",(long long)row);exit(1);}
    int64_t local=row-m->ple_part_start[p]; st_tensor *t=m->ple_parts[p]; const char *nm=m->ple_part_names[p];
    Q38_NEED(c->ngram_head_dim>0&&local>=0&&local<=INT64_MAX/c->ngram_head_dim,
             "PLE row byte offset overflows");
    if(t->dtype==4){
        uint8_t raw[512]; Q38_NEED(c->ngram_head_dim<=(int)sizeof raw,"PLE row too wide");
        st_read_slice_raw_cap(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,raw,sizeof raw,1);
        for(int d=0;d<c->ngram_head_dim;d++)out[d]=e4m3_decode(raw[d])*m->ple_weight_scale;
    } else st_read_slice_f32(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,out,1);
}

static int64_t q38_hash_row(Model *m,int head,int ngram,int64_t cur,int64_t p1,int64_t p2) {
    uint64_t x=(uint64_t)cur*(uint64_t)m->ple_multipliers[0];
    x^=(uint64_t)p1*(uint64_t)m->ple_multipliers[1];
    if(ngram==3)x^=(uint64_t)p2*(uint64_t)m->ple_multipliers[2];
    int64_t sx=(int64_t)x,mod=m->ple_head_vocab[head],r=sx%mod;if(r<0)r+=mod;
    return m->ple_head_offset[head]+r;
}


/* Anticipa le letture PLE all'inizio del forward invece di emetterle inline al
 * layer che le usa.
 *
 * Perche' si puo' fare, e perche' NON si puo' fare con gli esperti: gli indici
 * delle righe PLE sono funzione pura degli id dei token e della finestra di due
 * token che li precede -- q38_hash_row non guarda nessuno stato nascosto. Sono
 * quindi noti PRIMA che parta un solo calcolo. Gli esperti no: quelli dipendono
 * dal router del layer precedente, e per questo la loro finestra di anticipo e'
 * di un layer e non di tutto il forward.
 *
 * Il PLE cade sul layer 2 di 48, quindi fra l'emissione e l'uso ci sono due
 * layer interi di calcolo -- con il loro streaming di esperti, che su questo
 * motore e' la parte lenta. E' abbondantemente il tempo di far arrivare 16
 * righe da 160 byte.
 *
 * Sono anche letture PARALLELE invece che in fila: prima erano 16 pread seriali
 * a queue depth 1 per token, che in prefill diventano lunghezza-del-prompt per
 * 16, una alla volta.
 *
 * Riordino puro: stessi byte, letti prima. I token non cambiano, e il test lo
 * pretende invece di darlo per scontato. */
static void q38_ple_prefetch(Model *m,const int *ids,int S) {
    Cfg *c=&m->c;
    free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;
    if(c->ple_layer<0||S<1||c->ngram_heads<1||c->ngram_head_dim<1) return;
    if(!q38_env_bool("Q38_PLE_PREFETCH",1)) return;

    int64_t per_row=c->ngram_head_dim, per_pos=(int64_t)c->ngram_heads*per_row;
    if(S>INT_MAX/c->ngram_heads) return;
    float *buffer=(float*)malloc((size_t)S*per_pos*sizeof(float));
    int64_t *rows=(int64_t*)malloc((size_t)S*c->ngram_heads*sizeof(int64_t));
    if(!buffer||!rows){free(buffer);free(rows);return;}   /* niente prefetch: si legge inline */

    /* La finestra di due token viene SIMULATA, non mutata: q38_ple la aggiorna
     * per conto suo mentre gira, e toccarla qui la farebbe avanzare due volte. */
    int64_t history[2]={m->ple_history[0],m->ple_history[1]};
    int history_len=m->ple_history_len;
    for(int s=0;s<S;s++){
        int64_t p1=history_len>=1?history[history_len-1]:c->eos_id;
        int64_t p2=history_len>=2?history[history_len-2]:c->eos_id;
        for(int h=0;h<c->ngram_heads;h++){
            int ng=h<c->heads_per_ngram?2:3;
            rows[(int64_t)s*c->ngram_heads+h]=q38_hash_row(m,h,ng,ids[s],p1,p2);
        }
        if(ids[s]==c->eos_id) history_len=0;
        else if(history_len==0){history[0]=ids[s];history_len=1;}
        else if(history_len==1){history[1]=ids[s];history_len=2;}
        else {history[0]=history[1];history[1]=ids[s];}
    }

    int64_t total=(int64_t)S*c->ngram_heads;
    volatile int failed=0;
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<total;i++){
        if(failed) continue;
        q38_ple_row(m,rows[i],buffer+i*per_row);
    }
    free(rows);
    if(failed){free(buffer);return;}
    m->ple_pref=buffer; m->ple_pref_rows=S;
}

static void q38_ple(Model *m,const int *ids,int S,const float *hyper,float *out) {
    double phase_started=now_s();
    Cfg *c=&m->c; Layer *l=&m->L[c->ple_layer]; int H=c->hidden,C=c->hc_count,W=c->hc_width,E=c->ple_dim;
    float *emb=falloc(E),*keys=falloc(W),*value=falloc(H),*kn=falloc(W),*qn=falloc(W),*gated=falloc(W),*norm=falloc(W);
    int state_len=(c->ple_convk-1)*c->ngram_size; float *ring=m->PLE_conv_state;
    for(int s=0;s<S;s++){
        int64_t p1=m->ple_history_len>=1?m->ple_history[m->ple_history_len-1]:c->eos_id;
        int64_t p2=m->ple_history_len>=2?m->ple_history[m->ple_history_len-2]:c->eos_id;
        if(m->ple_pref&&s<m->ple_pref_rows){
            /* gia' in memoria: le ha portate q38_ple_prefetch mentre i primi
             * layer calcolavano */
            memcpy(emb,m->ple_pref+(int64_t)s*c->ngram_heads*c->ngram_head_dim,
                   (size_t)c->ngram_heads*c->ngram_head_dim*sizeof(float));
        } else for(int h=0;h<c->ngram_heads;h++){
            int ng=h<c->heads_per_ngram?2:3; int64_t row=q38_hash_row(m,h,ng,ids[s],p1,p2);
            q38_ple_row(m,row,emb+(int64_t)h*c->ngram_head_dim);
        }
        q38_dense_matmul(m,keys,emb,&l->ple_key,1,E,W);q38_dense_matmul(m,value,emb,&l->ple_value,1,E,H);
        for(int b=0;b<C;b++){
            q38_rms0(kn+(int64_t)b*H,keys+(int64_t)b*H,l->ple_norm_key+(int64_t)b*H,H,c->eps);
            q38_rms0(qn+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,l->ple_norm_query+(int64_t)b*H,H,c->eps);
            float dot=0.f;for(int d=0;d<H;d++)dot+=kn[(int64_t)b*H+d]*qn[(int64_t)b*H+d];dot/=sqrtf((float)H);
            float shaped=copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot),g=q38_sigmoid(shaped);
            for(int d=0;d<H;d++)gated[(int64_t)b*H+d]=g*value[d];
            q38_rms0(norm+(int64_t)b*H,gated+(int64_t)b*H,l->ple_norm_conv+(int64_t)b*H,H,c->eps);
        }
        for(int d=0;d<W;d++){
            float a=l->ple_conv[(int64_t)d*c->ple_convk+c->ple_convk-1]*norm[d];
            for(int k=0;k<c->ple_convk-1;k++)a+=l->ple_conv[(int64_t)d*c->ple_convk+k]*ring[(int64_t)d*state_len+k*c->ngram_size];
            out[(int64_t)s*W+d]=gated[d]+q38_silu(a);
            float *r=ring+(int64_t)d*state_len;for(int k=0;k<state_len-1;k++)r[k]=r[k+1];r[state_len-1]=norm[d];
        }
        if(ids[s]==c->eos_id)m->ple_history_len=0;
        else if(m->ple_history_len==0){m->ple_history[0]=ids[s];m->ple_history_len=1;}
        else if(m->ple_history_len==1){m->ple_history[1]=ids[s];m->ple_history_len=2;}
        else {m->ple_history[0]=m->ple_history[1];m->ple_history[1]=ids[s];}
    }
    free(emb);free(keys);free(value);free(kn);free(qn);free(gated);free(norm);
    q38_tm_add(m,Q38_TM_PLE,phase_started);
}

/* The chunk ceiling and the workspace budget used to be compile-time only.  An
 * isolated prefill measurement (274 tokens, one forward) showed that the
 * ceiling -- not the budget -- is what binds: 32 rows cut the prompt into nine
 * chunks, each chunk touches ~91 distinct experts, so a loaded expert serves
 * ~3.3 rows.  That drags 4.69 MiB of FP8 weights in for three rows of
 * activations, which is decode-grade arithmetic intensity inside a path that is
 * supposed to be batched, and it shows: 1.29 TFLOP in 48.8 s is 26.5 GFLOP/s,
 * a few percent of what the cores can do.  Both values are therefore runtime
 * knobs now.  Widening the chunk cannot change any result -- boundaries alter
 * neither routing nor accumulation order -- so this is a pure A/B. */
static int q38_env_positive_int(const char *name,int default_value,
                                int max_value) {
    const char *value=getenv(name);
    if(!value||!*value)return default_value;
    char *end=NULL;long parsed=strtol(value,&end,10);
    if(end==value||*end||parsed<1||parsed>(long)max_value){
        fprintf(stderr,"%s must be an integer in 1..%d\n",name,max_value);
        exit(1);
    }
    return (int)parsed;
}

static int q38_prefill_batch_rows(void) {
    static int cached=0;
    if(!cached)
        cached=q38_env_positive_int("Q38_PREFILL_BATCH_ROWS",
                                    Q38_PREFILL_BATCH_ROWS,1<<20);
    return cached;
}

/* Expressed in MiB because the byte count is the thing a human gets wrong. */
static uint64_t q38_prefill_workspace_bytes(void) {
    static uint64_t cached=0;
    if(!cached)
        cached=(uint64_t)q38_env_positive_int(
                   "Q38_PREFILL_WORKSPACE_MIB",
                   (int)(Q38_PREFILL_WORKSPACE_BYTES>>20),4096)<<20;
    return cached;
}

/* Choose a context-independent prefill chunk whose private workspace fits the
 * common target.  Callers provide exact fixed and per-row byte counts; even a
 * hostile-but-valid geometry gets one row rather than an unbounded allocation. */
static int q38_bounded_prefill_rows(int requested,uint64_t fixed,
                                    uint64_t per_row) {
    int ceiling=q38_prefill_batch_rows();
    uint64_t budget=q38_prefill_workspace_bytes();
    int rows=requested<ceiling?requested:ceiling;
    if(rows<1)return 1;
    for(;rows>1;rows--)
        if(per_row<=UINT64_MAX/(uint64_t)rows&&
           fixed<=UINT64_MAX-per_row*(uint64_t)rows&&
           fixed+per_row*(uint64_t)rows<=budget)
            return rows;
    return 1;
}

/* Batch the four resident DeltaNet input projections and the output projection
 * in bounded chunks.  The convolution and recurrent update remain strictly
 * token-causal inside each chunk, so chunk boundaries cannot change state or
 * floating-point order. */
static void q38_deltanet(Model *m,Layer *l,int layer,const float *x,int S,
                         float *out) {
    double phase_started=now_s();
    Cfg *c=&m->c;
    int H=c->hidden,VH=c->dn_vheads,KH=c->dn_kheads;
    int KD=c->dn_kdim,VD=c->dn_vdim,CD=c->dn_conv_dim,CK=c->dn_convk;
    int V=VH*VD,K=KH*KD,rep=VH/KH;
    uint64_t row_floats=(uint64_t)CD+2u*(uint64_t)V+2u*(uint64_t)VH;
    uint64_t fixed_floats=(uint64_t)CD+2u*(uint64_t)VH*(uint64_t)KD+
                          (uint64_t)V;
    uint64_t row_bytes=row_floats>UINT64_MAX/sizeof(float)?
                       UINT64_MAX:row_floats*sizeof(float);
    uint64_t fixed_bytes=fixed_floats>UINT64_MAX/sizeof(float)?
                         UINT64_MAX:fixed_floats*sizeof(float);
    int rows_capacity=m->prefill_batch?
                      q38_bounded_prefill_rows(S,fixed_bytes,row_bytes):1;

    float *qkv=falloc((int64_t)rows_capacity*CD);
    float *z=falloc((int64_t)rows_capacity*V);
    float *bb=falloc((int64_t)rows_capacity*VH);
    float *aa=falloc((int64_t)rows_capacity*VH);
    float *norm=falloc((int64_t)rows_capacity*V);
    float *conv=falloc(CD);
    float *q=falloc((int64_t)VH*KD),*k=falloc((int64_t)VH*KD);
    float *core=falloc(V);
    float *rec=m->DN_rec[layer],*ring=m->DN_conv[layer];

    for(int base=0;base<S;) {
        int rows=S-base<rows_capacity?S-base:rows_capacity;
        const float *chunk=x+(int64_t)base*H;
        if(rows==1){
            Q38Weight *pw[4]={&l->dn_qkv,&l->dn_z,&l->dn_b,&l->dn_a};
            float *py[4]={qkv,z,bb,aa}; int po[4]={CD,V,VH,VH};
            if(!q38_dense_multi_same_x(m,chunk,pw,py,po,4,H)){
                q38_dense_matmul(m,qkv,chunk,&l->dn_qkv,rows,H,CD);
                q38_dense_matmul(m,z,chunk,&l->dn_z,rows,H,V);
                q38_dense_matmul(m,bb,chunk,&l->dn_b,rows,H,VH);
                q38_dense_matmul(m,aa,chunk,&l->dn_a,rows,H,VH);
            }
        } else {
            q38_dense_matmul(m,qkv,chunk,&l->dn_qkv,rows,H,CD);
            q38_dense_matmul(m,z,chunk,&l->dn_z,rows,H,V);
            q38_dense_matmul(m,bb,chunk,&l->dn_b,rows,H,VH);
            q38_dense_matmul(m,aa,chunk,&l->dn_a,rows,H,VH);
        }

        for(int s=0;s<rows;s++) {
            float *qkv_row=qkv+(int64_t)s*CD;
            float *z_row=z+(int64_t)s*V;
            float *b_row=bb+(int64_t)s*VH;
            float *a_row=aa+(int64_t)s*VH;
            for(int d=0;d<CD;d++) {
                float value=l->dn_conv[(int64_t)d*CK+CK-1]*qkv_row[d];
                float *history=ring+(int64_t)d*(CK-1);
                for(int tap=0;tap<CK-1;tap++)
                    value+=l->dn_conv[(int64_t)d*CK+tap]*history[tap];
                conv[d]=q38_silu(value);
                for(int tap=0;tap<CK-2;tap++)history[tap]=history[tap+1];
                history[CK-2]=qkv_row[d];
            }
            const float *qi=conv,*ki=conv+K,*vi=conv+2*K;
            for(int h=0;h<VH;h++) {
                float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD;
                memcpy(qh,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
                memcpy(kh,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
                double qsum=1e-6,ksum=1e-6;
                for(int d=0;d<KD;d++) {
                    qsum+=(double)qh[d]*qh[d];
                    ksum+=(double)kh[d]*kh[d];
                }
                float qscale=1.f/sqrtf((float)qsum)/sqrtf((float)KD);
                float kscale=1.f/sqrtf((float)ksum);
                for(int d=0;d<KD;d++){qh[d]*=qscale;kh[d]*=kscale;}
            }
            #pragma omp parallel for schedule(static)
            for(int h=0;h<VH;h++) {
                float *state=rec+(int64_t)h*KD*VD;
                const float *qh=q+(int64_t)h*KD;
                const float *kh=k+(int64_t)h*KD;
                const float *vh=vi+(int64_t)h*VD;
                float alpha=expf(-expf(l->dn_alog[h])*
                                 q38_softplus(a_row[h]+l->dn_dtbias[h]));
                float beta=q38_sigmoid(b_row[h]);
                float delta[512];
#if defined(__AVX2__)
                if(q38_v4_on("Q38_DN_AVX2")){
                    q38_dn_head_avx2(state,qh,kh,vh,delta,core+(int64_t)h*VD,KD,VD,alpha,beta);
                }else
#endif
                {
                    int64_t state_cells=(int64_t)KD*VD;
                    for(int64_t cell=0;cell<state_cells;cell++)state[cell]*=alpha;
                    for(int value=0;value<VD;value++) {
                        float previous=0.f;
                        for(int d=0;d<KD;d++)
                            previous+=kh[d]*state[(int64_t)d*VD+value];
                        delta[value]=(vh[value]-previous)*beta;
                    }
                    for(int d=0;d<KD;d++)for(int value=0;value<VD;value++)
                        state[(int64_t)d*VD+value]+=kh[d]*delta[value];
                    for(int value=0;value<VD;value++) {
                        float current=0.f;
                        for(int d=0;d<KD;d++)
                            current+=qh[d]*state[(int64_t)d*VD+value];
                        core[(int64_t)h*VD+value]=current;
                    }
                }
            }
            float *norm_row=norm+(int64_t)s*V;
            for(int h=0;h<VH;h++)
                q38_rmsg(norm_row+(int64_t)h*VD,core+(int64_t)h*VD,
                         z_row+(int64_t)h*VD,l->dn_norm,VD,c->eps,1);
        }
        q38_dense_matmul(m,out+(int64_t)base*H,norm,&l->dn_out,
                         rows,V,H);
        base+=rows;
    }
    free(qkv);free(z);free(bb);free(aa);free(norm);free(conv);
    free(q);free(k);free(core);
    q38_tm_add(m,Q38_TM_DELTANET,phase_started);
}

typedef struct { float score; int block; } Q38Block;
static int q38_block_desc(const void *aa,const void *bb){
    const Q38Block *a=(const Q38Block*)aa,*b=(const Q38Block*)bb;
    if(a->score>b->score)return -1;if(a->score<b->score)return 1;return a->block-b->block;
}

static void q38_attention(Model *m,Layer *l,int layer,const float *x,int S,int pos_base,float *out) {
    Cfg *c=&m->c;int H=c->hidden,QH=c->q_heads,KVH=c->kv_heads,D=c->head_dim;
    int IQ=c->idx_qheads,ID=c->idx_dim,R=c->idx_ratio,maxsel=c->idx_budget+R-1;
    float *qp=falloc((int64_t)S*QH*2*D),*kp=falloc((int64_t)S*KVH*D),*vp=falloc((int64_t)S*KVH*D);
    float *ip=falloc((int64_t)S*(IQ+c->idx_kheads)*ID);
    if(S==1){
        Q38Weight *pw[3]={&l->q,&l->k,&l->v};
        float *py[3]={qp,kp,vp}; int po[3]={QH*2*D,KVH*D,KVH*D};
        if(!q38_dense_multi_same_x(m,x,pw,py,po,3,H)){
            q38_dense_matmul(m,qp,x,&l->q,S,H,QH*2*D);
            q38_dense_matmul(m,kp,x,&l->k,S,H,KVH*D);
            q38_dense_matmul(m,vp,x,&l->v,S,H,KVH*D);
        }
    } else {
        q38_dense_matmul(m,qp,x,&l->q,S,H,QH*2*D);
        q38_dense_matmul(m,kp,x,&l->k,S,H,KVH*D);
        q38_dense_matmul(m,vp,x,&l->v,S,H,KVH*D);
    }
    q38_dense_matmul(m,ip,x,&l->idx_qk,S,H,(IQ+c->idx_kheads)*ID);
    /* Cause before parallelism: the K/V/IK writes are disjoint per position
     * (each s writes only its own row) and must be complete before the
     * ranking, which reads the whole IK[0..pos] prefix.  Guarded on S>1 so
     * decode keeps the serial path it has today. */
    #pragma omp parallel for schedule(static) if(S>1)
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<KVH;h++){
            float *kh=kp+(int64_t)s*KVH*D+(int64_t)h*D;q38_rms0(kh,kh,l->kn,D,c->eps);q38_rope(kh,D,c->rotary_dim,pos,c->theta);
            memcpy(m->K[layer]+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));
            memcpy(m->V[layer]+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));
        }
        memcpy(m->IK[layer]+(int64_t)pos*ID,ip+(int64_t)s*(IQ+1)*ID+(int64_t)IQ*ID,(size_t)ID*sizeof(float));
    }
    float *heads=falloc((int64_t)S*QH*D);
    /* Ranking and attention are independent per position: no shared writes
     * (heads is row-disjoint, the scratch is per-thread) and no FP order
     * changes inside a position, so the result is bit-identical to the
     * serial path. The scheduling is dynamic because the ranking cost grows
     * with the position (the IK prefix to read is O(pos)). */
    double index_dt=0,attn_dt=0;
    /* Wall, not aggregate CPU: the reduction below sums per-thread seconds, so
     * at prefill these two phases would report ~20x what the clock saw while
     * every other phase reports wall -- on a 3006-token prompt the phase sum
     * came to 510 s against a 268 s TTFT, the parts outweighing the whole.
     * Take the clock across the whole team and split it by CPU share.  At
     * decode S==1 the loop is serial, cpu_total equals the wall, and the
     * rescale below is an exact no-op. */
    double qsa_wall_started=now_s();
    #pragma omp parallel for schedule(dynamic,8) reduction(+:index_dt,attn_dt) if(S>1)
    for(int s=0;s<S;s++){
        int pos=pos_base+s,visible=pos+1,blocks=visible/R,tail=blocks*R;
        double phase_started=now_s();
        float *qidx=falloc((int64_t)IQ*ID),*pool=falloc(ID);
        int *selected=(int*)malloc((size_t)maxsel*sizeof(int));
        if(!selected){fprintf(stderr,"OOM QSA selection\n");exit(1);}
        for(int h=0;h<IQ;h++){float *qh=qidx+(int64_t)h*ID;memcpy(qh,ip+(int64_t)s*(IQ+1)*ID+(int64_t)h*ID,(size_t)ID*sizeof(float));q38_rms0(qh,qh,l->idx_qn,ID,c->eps);q38_rope(qh,ID,c->rotary_dim,pos,c->theta);}
        int take=blocks<c->idx_budget/R?blocks:c->idx_budget/R,nsel=0;
        Q38Block *rank=blocks?(Q38Block*)malloc((size_t)blocks*sizeof(Q38Block)):NULL;
        if(blocks&&!rank){fprintf(stderr,"OOM QSA block ranking\n");exit(1);}
        for(int b=0;b<blocks;b++){
            memset(pool,0,(size_t)ID*sizeof(float));for(int r=0;r<R;r++){const float *raw=m->IK[layer]+(int64_t)(b*R+r)*ID;for(int d=0;d<ID;d++)pool[d]+=raw[d]/R;}
            q38_rms0(pool,pool,l->idx_kn,ID,c->eps);q38_rope(pool,ID,c->rotary_dim,b*R,c->theta);
            float score=0.f;for(int h=0;h<IQ;h++){float a=0.f;for(int d=0;d<ID;d++)a+=qidx[(int64_t)h*ID+d]*pool[d];if(a>0.f)score+=a;}rank[b]=(Q38Block){score/sqrtf((float)ID),b};
        }
        if(blocks)qsort(rank,(size_t)blocks,sizeof(Q38Block),q38_block_desc);
        for(int z=0;z<take;z++)for(int r=0;r<R;r++)selected[nsel++]=rank[z].block*R+r;
        for(int t=tail;t<visible;t++)selected[nsel++]=t;free(rank);
        index_dt+=now_s()-phase_started; phase_started=now_s();
        for(int h=0;h<QH;h++){
            float *qraw=qp+(int64_t)s*QH*2*D+(int64_t)h*2*D;
            float *qh=q38_cpu_v3()?q38_tls_f32(&g_q38_qsa_qh_ws,(size_t)D):falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,c->theta);float *score=q38_cpu_v3()?q38_tls_f32(&g_q38_qsa_score_ws,(size_t)nsel):falloc(nsel);float mx=-INFINITY;
            int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->K[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}
            float den=0.f;for(int j=0;j<nsel;j++){score[j]=expf(score[j]-mx);den+=score[j];}
            float *oh=heads+(int64_t)s*QH*D+(int64_t)h*D;memset(oh,0,(size_t)D*sizeof(float));
            for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->V[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}
            for(int d=0;d<D;d++)oh[d]*=q38_sigmoid(qraw[D+d]);if(!q38_cpu_v3()){free(qh);free(score);}
        }
        attn_dt+=now_s()-phase_started;
        free(qidx);free(pool);free(selected);
    }
    double qsa_wall=now_s()-qsa_wall_started,cpu_total=index_dt+attn_dt;
    if(cpu_total>0.0){
        index_dt=qsa_wall*(index_dt/cpu_total);
        attn_dt =qsa_wall*(attn_dt /cpu_total);
    }
    m->timers.seconds[Q38_TM_QSA_INDEX]+=index_dt;
    m->timers.seconds[Q38_TM_QSA_ATTENTION]+=attn_dt;
    q38_dense_matmul(m,out,heads,&l->o,S,QH*D,H);
    free(qp);free(kp);free(vp);free(ip);free(heads);
}

/* The single-row path is intentionally kept separate from prefill.  Decode is
 * the latency-sensitive steady state and should not pay for route tables or a
 * prompt-sized workspace. */
/* ---- CUDA VRAM expert tier (qwen36_tier.c, fp8 streaming mode) ------------
 * The tier keeps its own copies of hot experts in VRAM; the RAM LRU below is
 * untouched by it. qt_issue() takes the resident experts of a token's route
 * off the CPU, qt_take() adds their outputs back, and every expert the CPU
 * did compute is reported with qt_note() so the tier can promote it. The
 * tier copies what it needs during qt_note; the slot may be recycled by the
 * next token. Only native FP8 slots are reported: gate.data at the slab start
 * is what q38_bind_fp8_slot() produces, an expanded slot points elsewhere. */
static void q38_tier_note(int layer,int eid,const Slot *ex) {
    if(!qt_ready()||!ex->fp8_slab||ex->gate.data!=ex->fp8_slab)return;
    qt_note(layer,eid,(const uint8_t*)ex->gate.data,(const uint8_t*)ex->up.data,
            (const uint8_t*)ex->down.data,ex->gate.scales,ex->up.scales,ex->down.scales);
}

/* ---- dense trunk on the GPU (R7b, stage 1) ---------------------------------
 * The BF16 trunk is the largest fixed cost of a decode token here (about a
 * third), and it is bandwidth-bound. Every matrix of at least 1 MiB is
 * offered to the tier's placer by name and layer before qt_init; whatever
 * the placer accepts is quantized to int8 per row at start (scale = max|w| /
 * 127, the qwen36 dnproj/lmhead format) and uploaded once; the BF16 copy
 * stays for prefill and as fallback. Q38_TRUNK_GPU=0 keeps the trunk on the
 * CPU (parity runs against the BF16 reference). */
typedef struct { Q38Weight *w; char name[16]; int layer; } Q38TrunkItem;
static Q38TrunkItem *g_trunk; static int g_trunk_n, g_trunk_cap, g_trunk_offer_gpu;
static long g_trunk_min_kb; static const char *g_trunk_skip;   /* read once per load in q38_trunk_offer_all */
static void q38_trunk_add(Q38Weight *w,const char *name,int layer) {
    if(!w||!w->data||(w->kind!=Q38_WEIGHT_BF16&&w->kind!=Q38_WEIGHT_F32))return;
    size_t bytes=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float);
    /* Q38_TRUNK_MIN_KB (default 1024): a round trip costs more than a tiny
     * GEMV saves, and a tiny matrix in BF16 costs nothing on the CPU either;
     * Q38_TRUNK_SKIP=name,name: leave those components in BF16 on the CPU
     * (bisecting a numeric difference, or a component that does not pay) */
    long min_kb=g_trunk_min_kb; const char *skip=g_trunk_skip;
    if(bytes<(size_t)min_kb*1024)return;
    /* BABA_Q38_TRUNK_SKIP_CPU_INT8_FIX:
     * Q38_TRUNK_SKIP is a GPU-placement control, not a CPU-quantization control.
     * Keep every >=min_kb matrix in g_trunk so q38_trunk_cpu_int8() still
     * creates the normal CPU int8 fallback. Only suppress qt_trunk_offer(). */
    int skip_gpu=0;
    if(skip&&*skip){
        size_t n=strlen(name); const char *s=skip;
        while(*s){ const char *c=strchr(s,','); size_t l=c?(size_t)(c-s):strlen(s);
                   if(l==n&&!strncmp(s,name,n)){skip_gpu=1;break;}
                   if(!c)break; s=c+1; }
    }
    if(g_trunk_n==g_trunk_cap){
        g_trunk_cap=g_trunk_cap?2*g_trunk_cap:256;
        g_trunk=(Q38TrunkItem*)realloc(g_trunk,(size_t)g_trunk_cap*sizeof(*g_trunk));
        if(!g_trunk){fprintf(stderr,"OOM trunk table\n");exit(1);}
    }
    Q38TrunkItem *it=&g_trunk[g_trunk_n++]; it->w=w; it->layer=layer;
    snprintf(it->name,sizeof it->name,"%s",name);
    if(g_trunk_offer_gpu&&!skip_gpu)qt_trunk_offer(it->name,layer,bytes);
}
static int q38_trunk_enabled(void) {
    const char *e=getenv("Q38_TRUNK_GPU"); return !(e&&e[0]=='0'&&!e[1]);
}
/* the trunk table, built once: lm_head first (the placer takes it first),
 * then the layers in order so a partial placement is a prefix of the layers.
 * The same table feeds the CPU's int8 rows; the placer is told about the
 * matrices only when the GPU trunk is enabled (Q38_TRUNK_GPU). */
static void q38_trunk_offer_all(Model *m) {
    g_trunk_n=0; m->trunk_table_built=1;          /* rebuilt per load: a test opens several models in one process */
    g_trunk_offer_gpu=q38_trunk_enabled();
    { const char *e=getenv("Q38_TRUNK_MIN_KB"); g_trunk_min_kb=e?atol(e):1024; g_trunk_skip=getenv("Q38_TRUNK_SKIP"); }
    Cfg *c=&m->c;
    q38_trunk_add(&m->lm_head,"lmhead",0);
    for(int l=0;l<c->layers;l++){
        Layer *L=&m->L[l];
        if(c->is_attn[l]){
            q38_trunk_add(&L->q,"attnq",l); q38_trunk_add(&L->k,"attnk",l);
            q38_trunk_add(&L->v,"attnv",l); q38_trunk_add(&L->o,"attno",l);
            q38_trunk_add(&L->idx_qk,"qsaidx",l);
        } else {
            q38_trunk_add(&L->dn_qkv,"dnqkv",l); q38_trunk_add(&L->dn_z,"dnz",l);
            q38_trunk_add(&L->dn_out,"dnout",l);
        }
        q38_trunk_add(&L->attn_gr.down,"hcad",l); q38_trunk_add(&L->attn_gr.up,"hcau",l);
        q38_trunk_add(&L->attn_gr.inject,"hcai",l);
        q38_trunk_add(&L->mlp_gr.down,"hcmd",l); q38_trunk_add(&L->mlp_gr.up,"hcmu",l);
        q38_trunk_add(&L->mlp_gr.inject,"hcmi",l);
        q38_trunk_add(&L->sh_g,"shg",l); q38_trunk_add(&L->sh_u,"shu",l); q38_trunk_add(&L->sh_d,"shd",l);
        q38_trunk_add(&L->router,"router",l);
    }
}
/* int8 per row, scale = max|w| / 127 (the qwen36 dnproj/lmhead format) */
static void q38_trunk_quantize(const Q38Weight *w,int8_t **qp,float **scp) {
    int O=w->rows,I=w->cols;
    int8_t *q=(int8_t*)malloc((size_t)O*I); float *sc=(float*)malloc((size_t)O*sizeof(float));
    if(!q||!sc){fprintf(stderr,"OOM trunk quantization\n");exit(1);}
    #pragma omp parallel for schedule(static)
    for(int r=0;r<O;r++){
        float row[8192]; float *src=row; float *heap=NULL;
        if(I>8192){heap=(float*)malloc((size_t)I*sizeof(float)); src=heap;}
        q38_weight_row(w,r,src);
        float mx=0.f; for(int k=0;k<I;k++){float a=fabsf(src[k]); if(a>mx)mx=a;}
        float s=mx>0.f?mx/127.f:1.f, inv=1.f/s; sc[r]=s;
        int8_t *dst=q+(size_t)r*I;
        for(int k=0;k<I;k++){int v=(int)lrintf(src[k]*inv); if(v>127)v=127; if(v<-127)v=-127; dst[k]=(int8_t)v;}
        free(heap);
    }
    *qp=q; *scp=sc;
}
/* The trunk on the CPU: int8 rows with one scale per row, and the BF16 copy
 * released. The trunk is read whole on every token (3.6 G weights on the
 * released checkpoint, more than the ten routed experts), so its bytes are
 * the decode's floor: int8 halves them and the integer kernel keeps up with
 * the memory. Default on; Q38_TRUNK_CPU_INT8=0 keeps the BF16 rows and the
 * f32 kernel, the numeric reference. A matrix the tier already quantized for
 * the GPU keeps those same rows here (prefill rows run on the CPU). */
static int q38_trunk_cpu_int8_wanted(void) {
    const char *e=getenv("Q38_TRUNK_CPU_INT8"); return !(e&&e[0]=='0'&&!e[1]);
}
static void q38_trunk_cpu_int8(Model *m) {
    if(!q38_trunk_cpu_int8_wanted())return;
    if(!m->trunk_table_built)q38_trunk_offer_all(m);   /* the tier may have built it already */
    double t0=now_s(); size_t bytes=0,released=0; int n=0;
    for(int i=0;i<g_trunk_n;i++){
        Q38Weight *w=g_trunk[i].w;
        if(!w->q8)q38_trunk_quantize(w,&w->q8,&w->q8sc);
        bytes+=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float); n++;
        if(w->owns_data&&w->data){
            /* every path that reads this matrix now goes through q8 */
            uint64_t was=q38_weight_bytes(w);
            free(w->data); w->data=NULL; w->owns_data=0; released+=was;
            m->resident_weight_bytes-=was; m->resident_weight_bytes+=(size_t)w->rows*w->cols+(size_t)w->rows*sizeof(float);
        }
    }
    fprintf(stderr,"[qwen38] trunk: %d matrices int8 on the CPU (%.2f GiB, %.2f GiB of BF16 released) in %.1fs; Q38_TRUNK_CPU_INT8=0 keeps BF16\n",
            n,bytes/1073741824.0,released/1073741824.0,now_s()-t0);
}
/* after qt_init: quantize and upload what the placer accepted */
static void q38_trunk_place_all(Model *m) {
    (void)m;
    int placed=0; size_t placed_bytes=0; double t0=now_s();
    for(int i=0;i<g_trunk_n;i++){
        Q38TrunkItem *it=&g_trunk[i]; Q38Weight *w=it->w;
        int dev=qt_place_of(it->name,it->layer);
        if(dev==QT_PLACE_CPU)continue;
        int O=w->rows,I=w->cols;
        if(!w->q8)q38_trunk_quantize(w,&w->q8,&w->q8sc);   /* kept: the CPU answers prefill rows from the same bytes */
        const int8_t *q=w->q8; const float *sc=w->q8sc;
        int h=qt_dense_init(q,sc,I,O,dev);
        if(h>=0&&getenv("Q38_TRUNK_SELFTEST")){
            /* DIAG: GPU int8 GEMV against the same int8 matrix on the CPU */
            float *x=(float*)malloc((size_t)I*sizeof(float)),*yg=(float*)malloc((size_t)O*sizeof(float)),*yc=(float*)malloc((size_t)O*sizeof(float));
            for(int k=0;k<I;k++)x[k]=sinf(0.37f*k)+0.1f*(k%7);
            for(int r=0;r<O;r++){double a=0; for(int k=0;k<I;k++)a+=(double)q[(size_t)r*I+k]*x[k]; yc[r]=(float)(a*sc[r]);}
            int ok=qt_dense_matmul(h,yg,x,I,O); double num=0,den=0; int worst=0;
            for(int r=0;r<O;r++){double d=yg[r]-yc[r]; num+=d*d; den+=(double)yc[r]*yc[r]; if(fabs(d)>fabs(yg[worst]-yc[worst]))worst=r;}
            fprintf(stderr,"[selftest] %-7s L%-2d [O=%d I=%d] ok=%d rel.err %.2e worst row %d gpu %.5g cpu %.5g\n",it->name,it->layer,O,I,ok,den>0?sqrt(num/den):-1.0,worst,yg[worst],yc[worst]);
            free(x);free(yg);free(yc);
        }
        if(h>=0){ w->gpu=h+1; placed++; placed_bytes+=(size_t)O*I; }
    }
    if(g_trunk_n)
        fprintf(stderr,"[qtier] qwen38 trunk: %d of %d offered matrices resident as int8 (%.2f GiB) in %.1fs; the rest answers from the CPU\n",
                placed,g_trunk_n,placed_bytes/1073741824.0,now_s()-t0);
}

/* Start the tier after the model is loaded. COLI_CUDA=1 turns it on (the
 * tier reads that itself); it needs every layer's experts in native FP8 with
 * the block-scale bank resident, because the GPU kernels consume exactly the
 * checkpoint layout (e4m3 bytes + [ceil(O/128), ceil(I/128)] scales). */

static void q38_static_sages_warmstart(Model *m){
    const char *e=getenv("QT_STATIC_SAGES");
    if(!e||!*e||*e=='0'||!qt_ready())return;

    int max=m->c.layers*m->c.experts;
    int *layers=(int*)malloc((size_t)max*sizeof(int));
    int *eids=(int*)malloc((size_t)max*sizeof(int));
    if(!layers||!eids){fprintf(stderr,"[sages] OOM plan\n");exit(1);}

    int n=qt_plan_fill(layers,eids,max);
    fprintf(stderr,"[sages] warming %d fixed experts into VRAM\n",n);
    double started=now_s();

    for(int i=0;i<n;i++){
        Slot x;
        memset(&x,0,sizeof x);
        x.eid=-1;

        q38_load_expert(m,layers[i],eids[i],&x);

        if(x.fp8_slab && x.gate.data==x.fp8_slab){
            qt_note_planned(layers[i],eids[i],
                (const uint8_t*)x.gate.data,
                (const uint8_t*)x.up.data,
                (const uint8_t*)x.down.data,
                x.gate.scales,x.up.scales,x.down.scales);
        }else{
            qt_note_planned(layers[i],eids[i],NULL,NULL,NULL,NULL,NULL,NULL);
        }

        q38_weight_free(&x.gate);
        q38_weight_free(&x.up);
        q38_weight_free(&x.down);
        free(x.fp8_slab);
    }

    qt_fill_wait();
    fprintf(stderr,"[sages] monastery ready in %.1fs\n",now_s()-started);
    free(layers);
    free(eids);
}

static void q38_tier_start(Model *m,int cap) {
    const char *on=getenv("COLI_CUDA");
    if(!on||on[0]!='1'||on[1])return;
    Cfg *c=&m->c;
    if(!m->native_fp8){
        fprintf(stderr,"[qtier] qwen38: expert tier needs native FP8 experts (Q38_NATIVE_FP8=1); staying on the CPU\n");
        return;
    }
    for(int layer=0;layer<c->layers;layer++)
        if(!q38_prepare_expert_scale_bank(m,layer)){
            fprintf(stderr,"[qtier] qwen38: layer %d experts are not native FP8; staying on the CPU\n",layer);
            return;
        }
    q38_trunk_offer_all(m);                        /* sizes only; the placer decides in qt_init */
    if(qt_init_fp8(c->layers,c->experts,c->hidden,c->inter,cap,c->topk,E4M3_LUT)){
        atexit(qt_shutdown);
        fprintf(stderr,"[qtier] qwen38: fp8 expert tier on (RAM LRU %d/layer stays)\n",cap);
        q38_static_sages_warmstart(m);
        q38_trunk_place_all(m);
    }
}

static void q38_moe_decode(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    g_q38_q4_decode_scope=1;if(q38_q4_init_v7(m))q38_q4_reset_layer_v7(m,layer);
    Cfg *c=&m->c;int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    float *logits,*sg,*su,*sh,*shared,*eg,*eu,*eh,*eo,*moe_owned=NULL; size_t moe_n=(size_t)E+3u*(size_t)SI+2u*(size_t)H+3u*(size_t)I; float *mw; if(q38_cpu_v3()&&S==1)mw=q38_tls_f32(&g_q38_moe_ws,moe_n); else {moe_owned=falloc((int64_t)moe_n);mw=moe_owned;} logits=mw;sg=logits+E;su=sg+SI;sh=su+SI;shared=sh+SI;eg=shared+H;eu=eg+I;eh=eu+I;eo=eh+I;
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*H;float *ys=out+(int64_t)s*H;memset(ys,0,(size_t)H*sizeof(float));
        q38_dense_matmul(m,logits,xs,&l->router,1,H,E);float mx=logits[0];for(int e=1;e<E;e++)if(logits[e]>mx)mx=logits[e];
        double all=0;for(int e=0;e<E;e++){logits[e]=expf(logits[e]-mx);all+=logits[e];}
        int idx[Q38_MAX_TOPK];float val[Q38_MAX_TOPK];
        for(int z=0;z<K;z++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int j=0;j<z;j++)if(idx[j]==e)used=1;if(!used&&logits[e]>bv){bv=logits[e];best=e;}}idx[z]=rt_router_pick(best,z,E,layer);val[z]=logits[idx[z]];}
        double top=0;for(int z=0;z<K;z++)top+=val[z];double den=c->norm_topk?top:all;
        float route_gates[Q38_MAX_TOPK];
        for(int z=0;z<K;z++) route_gates[z]=(float)(val[z]/den);
        rt_route(layer,s,idx,route_gates,K); /* shared counts + post-normalization trace */
        /* Resident experts run on the GPU from here on (asynchronously); the
         * CPU only loads and computes the rest, in route order. */
        uint32_t qmask=qt_issue(layer,idx,K,xs);
        int cpu_idx[Q38_MAX_TOPK],cpu_rank[Q38_MAX_TOPK],cpu_n=0;
        for(int z=0;z<K;z++)if(!((qmask>>z)&1u)){cpu_idx[cpu_n]=idx[z];cpu_rank[cpu_n]=z;cpu_n++;}
        q38_prefetch_native_fp8_experts(m,layer,cpu_idx,cpu_n);
        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=0;
        pthread_t io_thread; int io_started=0;
        Q38AsyncExpertLoad io_ctx={m,layer,cpu_idx,cpu_n,selected,0};
        int q4_missing=0;
        if(q38_v4_on("Q38_Q4_ASYNC_SHARED")&&q38_q4_on_v7(m)&&cpu_n>=2){
            LCache *lc=&m->cache[layer];
            for(int ii=0;ii<cpu_n;ii++)if(lc->by_expert[cpu_idx[ii]]<0)q4_missing++;
            if(q4_missing>0&&pthread_create(&io_thread,NULL,q38_async_expert_load_main,&io_ctx)==0){
                io_started=1;g_q38_v4_async_batches++;g_q38_v4_async_jobs+=(uint64_t)q4_missing;
            }
        }
        double phase_started=now_s();
        double shared_started=phase_started;
        if(!q38_shared_mlp_cuda(shared,xs,&l->sh_g,&l->sh_u,&l->sh_d)){
            q38_weight_matmul(sg,xs,&l->sh_g,1,H,SI);
            q38_weight_matmul(su,xs,&l->sh_u,1,H,SI);
            for(int j=0;j<SI;j++)sh[j]=q38_silu(sg[j])*su[j];
            q38_weight_matmul(shared,sh,&l->sh_d,1,SI,H);
        }
        float gate=0.f;for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];gate=q38_sigmoid(gate);
        double shared_elapsed=now_s()-shared_started;
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);
        if(io_started){
            double join_started=now_s();
            pthread_join(io_thread,NULL);
            double join_elapsed=now_s()-join_started;
            loaded_batch=io_ctx.result;
            double hidden=shared_elapsed-join_elapsed;
            if(hidden>0)g_q38_v4_async_hidden_s+=hidden;
        }else loaded_batch=q38_expert_get_batch(m,layer,cpu_idx,cpu_n,selected);
        int q4_whole=(qmask==0&&cpu_n==K&&q38_q4_on_v7(m));
        if(q4_whole){
            const XfExpert *xp[Q38_MAX_TOPK];int xid[Q38_MAX_TOPK];float xval[Q38_MAX_TOPK];
            for(int i=0;i<cpu_n;i++){Slot *ex=loaded_batch?selected[i]:q38_expert_get(m,layer,cpu_idx[i]);
                if(!ex->q4_active){q4_whole=0;break;}xp[i]=&ex->q4;xid[i]=cpu_idx[i];xval[i]=route_gates[cpu_rank[i]];}
            if(q4_whole){size_t sb=xf_moe_scratch_bytes(1,cpu_n,H,I);
                void *scratch=q38_v4_on("Q38_Q4_TLS_SCRATCH")?q38_xf_scratch_get(sb):malloc(sb);
                if(!scratch){fprintf(stderr,"OOM q4 expert_ffn scratch\n");exit(1);}phase_started=now_s();
                xf_moe_run(ys,xs,1,cpu_n,H,I,xid,xval,xp,q38_q4_act_mode_v7(),scratch);
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
                if(!q38_v4_on("Q38_Q4_TLS_SCRATCH"))free(scratch);}
        }
        if(!q4_whole)for(int i=0;i<cpu_n;i++){
            int z=cpu_rank[i];Slot *ex=loaded_batch?selected[i]:q38_expert_get(m,layer,cpu_idx[i]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);q38_tier_note(layer,cpu_idx[i],ex);
        }
        /* GPU experts land after the CPU ones: same values, one more group in
         * the float sum (that is the only ordering difference to a CPU run). */
        if(!qt_take(qmask,route_gates,K,ys)){
            fprintf(stderr,"qwen38: CUDA expert collection failed at layer %d; stopping inference\n",layer);
            exit(1);
        }
        for(int d=0;d<H;d++)ys[d]+=gate*shared[d];
    }
    rt_trace_end();g_q38_q4_decode_scope=0;
    if(moe_owned)free(moe_owned);
}

typedef struct {
    int expert;
    float gate;
} Q38RouteAssignment;

/* Pick a prefill size from a fixed workspace budget.  The number of rows is
 * deliberately bounded independently of the context length: a long prompt
 * therefore reuses the same route, expert and matmul buffers one chunk at a
 * time.  A single assignment needs input, gate/up activations and output;
 * these are the only buffers that scale with the number of routed rows. */
static int q38_moe_prefill_rows(const Cfg *c,int requested) {
    int64_t H=c->hidden,I=c->inter,E=c->experts,K=c->topk,SI=c->shared_inter;
    uint64_t per_assignment=(uint64_t)(2LL*((int64_t)H+I))*sizeof(float)+
                             sizeof(Q38RouteAssignment)+2*sizeof(int);
    uint64_t per_row=(uint64_t)E*sizeof(float)+
                     (uint64_t)(3LL*(int64_t)SI+H+1)*sizeof(float);
    uint64_t fixed=(uint64_t)E*(4*sizeof(int)+2*sizeof(Slot*));
    if(fixed<=UINT64_MAX-sizeof(int))fixed+=sizeof(int); /* group_offsets[E] */
    else fixed=UINT64_MAX;
    uint64_t assignment_row=(uint64_t)K>UINT64_MAX/per_assignment?
                            UINT64_MAX:(uint64_t)K*per_assignment;
    uint64_t total_row=per_row>UINT64_MAX-assignment_row?
                       UINT64_MAX:per_row+assignment_row;
    return q38_bounded_prefill_rows(requested,fixed,total_row);
}


#ifdef _WIN32
#define Q38_Q4_STREAM_PREFILL_V11 1

static void q38_q4_stream_bind_v11(Model *m,void *slab,XfExpert *e){
    uint64_t mb=(uint64_t)m->c.hidden*m->c.inter/2;
    uint64_t sgu=(uint64_t)m->c.inter*(m->c.hidden/64)*4;
    unsigned char *p=(unsigned char*)slab;
    e->g4=p;
    e->u4=p+mb;
    e->d4=p+2*mb;
    e->gs=(float*)(p+3*mb);
    e->us=(float*)(p+3*mb+sgu);
    e->ds=(float*)(p+3*mb+2*sgu);
}

static int q38_moe_prefill_q4_stream_v11(Model *m,Layer *l,int layer,
                                         const float *x,int S,float *out){
    if(S<=1||!q38_q4_init_v7(m)||!q38_q4_win_open_v8())return 0;

    Cfg *c=&m->c;
    int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    int rows_capacity=q38_moe_prefill_rows(c,S);
    static int banner_done=0;

    float *logits=falloc((int64_t)rows_capacity*E);
    float *shared_gate=falloc(rows_capacity);
    float *shared_g=falloc((int64_t)rows_capacity*SI);
    float *shared_u=falloc((int64_t)rows_capacity*SI);
    float *shared_hidden=falloc((int64_t)rows_capacity*SI);
    float *shared_out=falloc((int64_t)rows_capacity*H);
    float *routed=falloc((int64_t)rows_capacity*H);

    int max_assign=rows_capacity*K;
    int *route_idx=(int*)malloc((size_t)max_assign*sizeof(int));
    float *route_val=(float*)malloc((size_t)max_assign*sizeof(float));
    const XfExpert **route_xf=(const XfExpert**)malloc((size_t)max_assign*sizeof(*route_xf));
    int *expert_to_unique=(int*)malloc((size_t)E*sizeof(int));
    int *unique=(int*)malloc((size_t)max_assign*sizeof(int));
    if(!route_idx||!route_val||!route_xf||!expert_to_unique||!unique){
        fprintf(stderr,"OOM Q4 streaming prefill metadata\n");exit(1);
    }

    for(int base=0;base<S;){
        int rows=S-base<rows_capacity?S-base:rows_capacity;
        int assignments=rows*K;
        for(int e=0;e<E;e++)expert_to_unique[e]=-1;
        int unique_count=0;

        q38_dense_matmul(m,logits,x+(int64_t)base*H,&l->router,rows,H,E);

        for(int s=0;s<rows;s++){
            float *prob=logits+(int64_t)s*E;
            float mx=prob[0];
            for(int e=1;e<E;e++)if(prob[e]>mx)mx=prob[e];
            double all=0.0;
            for(int e=0;e<E;e++){prob[e]=expf(prob[e]-mx);all+=prob[e];}

            int selected[Q38_MAX_TOPK];
            float selected_p[Q38_MAX_TOPK];
            for(int rank=0;rank<K;rank++){
                int best=-1;float bv=-1.f;
                for(int e=0;e<E;e++){
                    int used=0;
                    for(int p=0;p<rank;p++)if(selected[p]==e){used=1;break;}
                    if(!used&&prob[e]>bv){bv=prob[e];best=e;}
                }
                selected[rank]=rt_router_pick(best,rank,E,layer);
                selected_p[rank]=prob[selected[rank]];
            }

            double top=0.0;
            for(int rank=0;rank<K;rank++)top+=selected_p[rank];
            double den=c->norm_topk?top:all;
            float gates[Q38_MAX_TOPK];

            for(int rank=0;rank<K;rank++){
                int expert=selected[rank];
                float gate=(float)(selected_p[rank]/den);
                gates[rank]=gate;
                int a=s*K+rank;
                route_idx[a]=expert;
                route_val[a]=gate;
                if(expert_to_unique[expert]<0){
                    expert_to_unique[expert]=unique_count;
                    unique[unique_count++]=expert;
                }
            }
            rt_route(layer,base+s,selected,gates,K);
        }

        void **slab=(void**)calloc((size_t)unique_count,sizeof(void*));
        XfExpert *xf=(XfExpert*)calloc((size_t)unique_count,sizeof(*xf));
        OVERLAPPED *ov=(OVERLAPPED*)calloc((size_t)unique_count,sizeof(*ov));
        HANDLE *event=(HANDLE*)calloc((size_t)unique_count,sizeof(*event));
        if(!slab||!xf||!ov||!event){
            fprintf(stderr,"OOM Q4 streaming prefill I/O table\n");exit(1);
        }

        double read_started=now_s();
        for(int u=0;u<unique_count;u++){
            slab[u]=malloc((size_t)g_q38_q4.record_bytes);
            if(!slab[u]){fprintf(stderr,"OOM Q4 streaming expert slab\n");exit(1);}
            uint64_t off=(uint64_t)q38_q4_off_v7(layer,unique[u]);
            ov[u].Offset=(DWORD)(off&0xffffffffULL);
            ov[u].OffsetHigh=(DWORD)(off>>32);
            event[u]=CreateEventA(NULL,TRUE,FALSE,NULL);
            if(!event[u]){fprintf(stderr,"Q4 streaming CreateEvent failed\n");exit(1);}
            ov[u].hEvent=event[u];

            DWORD got=0;
            BOOL ok=ReadFile(g_q38_q4_win_handle,slab[u],
                             (DWORD)g_q38_q4.record_bytes,&got,&ov[u]);
            if(!ok&&GetLastError()!=ERROR_IO_PENDING){
                fprintf(stderr,"Q4 streaming ReadFile submit failed L%d E%d err=%lu\n",
                        layer,unique[u],(unsigned long)GetLastError());
                exit(1);
            }
        }

        for(int u=0;u<unique_count;u++){
            DWORD got=0;
            if(!GetOverlappedResult(g_q38_q4_win_handle,&ov[u],&got,TRUE)||
               got!=(DWORD)g_q38_q4.record_bytes){
                fprintf(stderr,"Q4 streaming completion failed L%d E%d got=%lu err=%lu\n",
                        layer,unique[u],(unsigned long)got,
                        (unsigned long)GetLastError());
                exit(1);
            }
            CloseHandle(event[u]);event[u]=NULL;
            q38_q4_stream_bind_v11(m,slab[u],&xf[u]);
        }
        q38_tm_add(m,Q38_TM_EXPERT_READ,read_started);
        m->expert_weight_reads+=(uint64_t)unique_count;
        m->expert_pair_reads+=(uint64_t)unique_count;
        if(unique_count>1)m->expert_parallel_batches++;

        for(int a=0;a<assignments;a++)
            route_xf[a]=&xf[expert_to_unique[route_idx[a]]];

        double phase_started=now_s();
        q38_weight_matmul(shared_g,x+(int64_t)base*H,&l->sh_g,rows,H,SI);
        q38_weight_matmul(shared_u,x+(int64_t)base*H,&l->sh_u,rows,H,SI);
        for(int s=0;s<rows;s++)
            for(int j=0;j<SI;j++)
                shared_hidden[(int64_t)s*SI+j]=
                    q38_silu(shared_g[(int64_t)s*SI+j])*
                    shared_u[(int64_t)s*SI+j];
        q38_weight_matmul(shared_out,shared_hidden,&l->sh_d,rows,SI,H);
        for(int s=0;s<rows;s++){
            const float *xs=x+(int64_t)(base+s)*H;
            float gate=0.f;
            for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];
            shared_gate[s]=q38_sigmoid(gate);
        }
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);

        size_t scratch_bytes=xf_moe_scratch_bytes(rows,K,H,I);
        void *scratch=malloc(scratch_bytes);
        if(!scratch){fprintf(stderr,"OOM Q4 streaming prefill expert_ffn scratch\n");exit(1);}

        phase_started=now_s();
        xf_moe_run(routed,x+(int64_t)base*H,rows,K,H,I,
                   route_idx,route_val,route_xf,q38_q4_act_mode_v7(),scratch);
        q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
        free(scratch);

        for(int s=0;s<rows;s++){
            float *ys=out+(int64_t)(base+s)*H;
            const float *rr=routed+(int64_t)s*H;
            const float *sh=shared_out+(int64_t)s*H;
            float sg=shared_gate[s];
            for(int d=0;d<H;d++)ys[d]=rr[d]+sg*sh[d];
        }

        for(int u=0;u<unique_count;u++){
            if(event[u])CloseHandle(event[u]);
            free(slab[u]);
        }
        free(slab);free(xf);free(ov);free(event);

        if(!banner_done){
            fprintf(stderr,
                    "[Q4 PREFILL STREAM] one-wave temporary experts, no LRU: "
                    "rows=%d unique=%d q4-bytes=%.1f MiB\n",
                    rows,unique_count,
                    (double)unique_count*g_q38_q4.record_bytes/1048576.0);
            banner_done=1;
        }
        base+=rows;
    }

    rt_trace_end();
    free(logits);free(shared_gate);free(shared_g);free(shared_u);
    free(shared_hidden);free(shared_out);free(routed);
    free(route_idx);free(route_val);free(route_xf);
    free(expert_to_unique);free(unique);
    return 1;
}
#endif

/* Prefill MoE: route a bounded row chunk first, then execute each distinct
 * expert's assignments as one batched SwiGLU.  Grouping is an I/O optimization
 * only.  Expert outputs are placed back in assignment order and the final
 * weighted reduction still visits rank 0..top-k-1 for every row, preserving the
 * decode path's floating-point accumulation order. */
static void q38_moe_prefill(Model *m,Layer *l,int layer,const float *x,
                            int S,float *out) {
#ifdef _WIN32
    if(q38_moe_prefill_q4_stream_v11(m,l,layer,x,S,out))return;
#endif
    Cfg *c=&m->c;
    int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    int rows_capacity=q38_moe_prefill_rows(c,S);
    int64_t max_assign=(int64_t)rows_capacity*K;

    Q38RouteAssignment *routes=(Q38RouteAssignment*)malloc(
        (size_t)max_assign*sizeof(*routes));
    int *assignments=(int*)malloc((size_t)max_assign*sizeof(*assignments));
    int *assignment_positions=(int*)malloc((size_t)max_assign*sizeof(*assignment_positions));
    int *group_counts=(int*)calloc((size_t)E,sizeof(*group_counts));
    int *group_offsets=(int*)malloc((size_t)(E+1)*sizeof(*group_offsets));
    int *group_cursor=(int*)malloc((size_t)E*sizeof(*group_cursor));
    int *unique=(int*)malloc((size_t)E*sizeof(*unique));
    Slot **batch_slots=(Slot**)calloc((size_t)E,sizeof(*batch_slots));
    if(!routes||!assignments||!assignment_positions||!group_counts||
       !group_offsets||!group_cursor||!unique||!batch_slots){
        fprintf(stderr,"OOM Qwen3.8 MoE prefill metadata\n");exit(1);
    }

    float *logits=falloc((int64_t)rows_capacity*E);
    float *shared_gate=falloc(rows_capacity);
    float *shared_g=falloc((int64_t)rows_capacity*SI);
    float *shared_u=falloc((int64_t)rows_capacity*SI);
    float *shared_hidden=falloc((int64_t)rows_capacity*SI);
    float *shared_out=falloc((int64_t)rows_capacity*H);
    float *expert_input=falloc(max_assign*H);
    float *expert_gate=falloc(max_assign*I);
    float *expert_up=falloc(max_assign*I);
    float *routed_out=falloc(max_assign*H);

    for(int base=0;base<S;) {
        int rows=S-base<rows_capacity?S-base:rows_capacity;
        int assignment_count=rows*K;
        memset(group_counts,0,(size_t)E*sizeof(*group_counts));

        /* Route the complete chunk with one resident router matmul.  Selection
         * intentionally mirrors q38_moe_decode, including rt_router_pick's
         * deterministic fallback for invalid logits. */
        q38_dense_matmul(m,logits,x+(int64_t)base*H,&l->router,rows,H,E);
        for(int s=0;s<rows;s++) {
            float *probabilities=logits+(int64_t)s*E;
            float maximum=probabilities[0];
            for(int e=1;e<E;e++)if(probabilities[e]>maximum)maximum=probabilities[e];
            double total=0.0;
            for(int e=0;e<E;e++) {
                probabilities[e]=expf(probabilities[e]-maximum);
                total+=probabilities[e];
            }
            int selected[Q38_MAX_TOPK];
            float selected_probability[Q38_MAX_TOPK];
            for(int rank=0;rank<K;rank++) {
                int best=-1;float best_value=-1.f;
                for(int e=0;e<E;e++) {
                    int used=0;
                    for(int previous=0;previous<rank;previous++)
                        if(selected[previous]==e){used=1;break;}
                    if(!used&&probabilities[e]>best_value){
                        best_value=probabilities[e];best=e;
                    }
                }
                selected[rank]=rt_router_pick(best,rank,E,layer);
                selected_probability[rank]=probabilities[selected[rank]];
            }
            double top=0.0;
            for(int rank=0;rank<K;rank++)top+=selected_probability[rank];
            double denominator=c->norm_topk?top:total;
            float gates[Q38_MAX_TOPK];
            for(int rank=0;rank<K;rank++) {
                gates[rank]=(float)(selected_probability[rank]/denominator);
                int assignment=s*K+rank;
                routes[assignment]=(Q38RouteAssignment){selected[rank],gates[rank]};
                group_counts[selected[rank]]++;
            }
            rt_route(layer,base+s,selected,gates,K);
        }

        /* Prefixing by expert makes every group contiguous while the inverse
         * map lets the final reduction recover the original row/rank order. */
        group_offsets[0]=0;
        int unique_count=0;
        for(int e=0;e<E;e++) {
            group_offsets[e+1]=group_offsets[e]+group_counts[e];
            if(group_counts[e])unique[unique_count++]=e;
        }
        memset(group_cursor,0,(size_t)E*sizeof(*group_cursor));
        for(int assignment=0;assignment<assignment_count;assignment++) {
            int expert=routes[assignment].expert;
            int position=group_offsets[expert]+group_cursor[expert]++;
            assignments[position]=assignment;
            assignment_positions[assignment]=position;
        }

        /* One advice range per distinct expert is enough for this chunk. */
        q38_prefetch_native_fp8_experts(m,layer,unique,unique_count);

        /* Shared expert work is independent across rows and remains resident;
         * batching it here also keeps its cost out of the routed groups. */
        double phase_started=now_s();
        q38_weight_matmul(shared_g,x+(int64_t)base*H,&l->sh_g,rows,H,SI);
        q38_weight_matmul(shared_u,x+(int64_t)base*H,&l->sh_u,rows,H,SI);
        for(int s=0;s<rows;s++)
            for(int j=0;j<SI;j++)
                shared_hidden[(int64_t)s*SI+j]=
                    q38_silu(shared_g[(int64_t)s*SI+j])*
                    shared_u[(int64_t)s*SI+j];
        q38_weight_matmul(shared_out,shared_hidden,&l->sh_d,rows,SI,H);
        for(int s=0;s<rows;s++) {
            const float *xs=x+(int64_t)(base+s)*H;
            float gate=0.f;
            for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];
            shared_gate[s]=q38_sigmoid(gate);
        }
        q38_tm_add(m,Q38_TM_SHARED_EXPERT,phase_started);

        /* A demand-set batch is particularly effective for native FP8: reserve
         * all slots before workers read the two coalesced ranges.  Other native
         * layouts use the same grouped matmul loop with the ordinary bounded
         * LRU loader, so no format loses correctness or caching. */
        memset(out+(int64_t)base*H,0,(size_t)rows*H*sizeof(float));
        /* A prompt chunk can route to more distinct experts than the retained
         * cache can hold.  Load and consume cache-sized groups instead of
         * declining the complete parallel demand set: every expert is still
         * loaded once for this chunk, and a later group may safely reuse its
         * slots because the preceding outputs already live in routed_out. */
        int load_limit=m->cache[layer].cap;
        if(load_limit<1)load_limit=1;
        for(int unique_base=0;unique_base<unique_count;) {
            int load_count=unique_count-unique_base;
            if(load_count>load_limit)load_count=load_limit;
            int loaded_batch=load_count>=2&&q38_expert_get_batch(
                m,layer,unique+unique_base,load_count,batch_slots);
            for(int offset=0;offset<load_count;offset++) {
                int e=unique[unique_base+offset];
                int count=group_counts[e];
                Slot *expert=loaded_batch?batch_slots[offset]:
                                          q38_expert_get(m,layer,e);
                int first=group_offsets[e];
                for(int a=0;a<count;a++) {
                    int assignment=assignments[first+a];
                    int row=assignment/K;
                    memcpy(expert_input+(int64_t)a*H,
                           x+(int64_t)(base+row)*H,(size_t)H*sizeof(float));
                }
                phase_started=now_s();
                q38_weight_matmul(expert_gate,expert_input,&expert->gate,
                                  count,H,I);
                q38_weight_matmul(expert_up,expert_input,&expert->up,count,H,I);
                for(int a=0;a<count;a++)for(int j=0;j<I;j++)
                    expert_gate[(int64_t)a*I+j]=
                        q38_silu(expert_gate[(int64_t)a*I+j])*
                        expert_up[(int64_t)a*I+j];
                q38_weight_matmul(routed_out+(int64_t)first*H,expert_gate,
                                  &expert->down,count,I,H);
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            }
            unique_base+=load_count;
        }

        /* The grouped execution above is intentionally not the reduction
         * order.  Replaying the original top-k sequence gives the same result
         * as the single-row implementation for F32, BF16 and block-FP8. */
        for(int s=0;s<rows;s++) {
            float *ys=out+(int64_t)(base+s)*H;
            for(int rank=0;rank<K;rank++) {
                int assignment=s*K+rank;
                const float *expert_output=routed_out+
                    (int64_t)assignment_positions[assignment]*H;
                float gate=routes[assignment].gate;
                for(int d=0;d<H;d++)ys[d]+=gate*expert_output[d];
            }
            const float *shared=shared_out+(int64_t)s*H;
            for(int d=0;d<H;d++)ys[d]+=shared_gate[s]*shared[d];
        }
        base+=rows;
    }
    rt_trace_end();
    free(logits);free(shared_gate);free(shared_g);free(shared_u);
    free(shared_hidden);free(shared_out);free(expert_input);free(expert_gate);
    free(expert_up);free(routed_out);free(routes);free(assignments);
    free(assignment_positions);free(group_counts);free(group_offsets);
    free(group_cursor);free(unique);free(batch_slots);
}

static void q38_moe(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    if(S<=1||!m->prefill_batch)q38_moe_decode(m,l,layer,x,S,out);
    else q38_moe_prefill(m,l,layer,x,S,out);
}

static void reset_recurrent(Model *m) {
    kv_prefix_clear(&m->kvp);
    Cfg *c=&m->c;
    for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
        memset(m->DN_rec[i],0,(size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim*sizeof(float));
        memset(m->DN_conv[i],0,(size_t)c->dn_conv_dim*(c->dn_convk-1)*sizeof(float));
    }
    memset(m->PLE_conv_state,0,(size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size*sizeof(float));m->ple_history_len=0;
}

static void ensure_kv(Model *m) {
    Cfg *c=&m->c;if(m->max_t<=m->kv_cap&&m->K)return;
    if(m->K){for(int i=0;i<c->layers;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);}free(m->K);free(m->V);free(m->IK);}
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));m->V=(float**)calloc((size_t)c->layers,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
    for(int i=0;i<c->layers;i++)if(c->is_attn[i]){
        m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);
    }
    m->kv_cap=m->max_t;
    kv_prefix_alloc(&m->kvp,m->kv_cap); /* ensure_kv discards the old rows */
}

/* Run only the requested native layer interval over hyper-residual activations.
 * Segment callers supply the four-stream boundary state directly; unlike step,
 * this path deliberately does not gather embeddings or apply the final mixer. */
static void q38_layers_forward_range(Model *m,float *hyper,const int *ids,
                                     int S,int pos_base,int layer_begin,
                                     int layer_end) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){
            float *ple=falloc((int64_t)S*W); q38_ple(m,ids,S,hyper,ple);
            for(int64_t z=0;z<(int64_t)S*W;z++) hyper[z]+=ple[z];
            free(ple);
        }
        q38_gr_read(m,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i]) q38_attention(m,l,i,mixed,S,pos_base,block);
        else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(m,&l->mlp_gr,hyper,S,mixed,inject);
        q38_moe(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
    }
    free(mixed); free(inject); free(block);
}

/* Canale logprobs (modalita jev). Qui la fotografia dello stato NON si
 * aggiunge: questo motore ce l'ha gia, si chiama Q38PrefixCache e salva stato
 * ricorrente, id e logit finali dopo ogni prompt. Manca solo la lettura, cioe
 * un passaggio di lm_head per posizione invece che solo sull'ultima, e il
 * predittore del primo token fresco, che sono i logit gia salvati. */
static int    g_echo_k = 0;
static const char *g_echo_id = NULL;
static const float *g_echo_pin_logit = NULL;   /* logit dell'ultima posizione riusata */

static void q38_echo(const char *id, int pos, int token, const float *lo, int V, int k){
    char tail[1024]; coli_logprob_tail(tail, sizeof tail, lo, V, token, k);
    unsigned char *piece=NULL; int n=0;
    if(decode_id_alloc(token,&piece,&n)) { piece=NULL; n=0; }
    if(n<0) n=0;
    printf("ECHO %s %d %d%s\n", id, n, pos, tail);
    if(n>0) fwrite(piece,1,(size_t)n,stdout);
    fputc('\n', stdout); fflush(stdout);
    free(piece);
}

static float *step(Model *m,const int *ids,int S,int pos_base) {
    Cfg *c=&m->c;int H=c->hidden,W=c->hc_width,C=c->hc_count;
    m->timers.forwards++;
    float *hyper=falloc((int64_t)S*W);
    /* Le righe PLE partono ADESSO, non al layer 2 dove servono: sono note dagli
     * id dei token soltanto, e i due layer che le precedono danno il tempo di
     * farle arrivare dal disco. */
    q38_ple_prefetch(m,ids,S);
    for(int s=0;s<S;s++){
        if(ids[s]<0||ids[s]>=c->vocab){fprintf(stderr,"token id %d outside vocabulary\n",ids[s]);exit(1);}
        float *e=hyper+(int64_t)s*W;
        int abs_pos=pos_base+s, vis_row=-1;
        if(m->vis_map&&abs_pos>=0&&abs_pos<m->vis_map_len) vis_row=m->vis_map[abs_pos];
        if(vis_row>=0&&vis_row<m->vis_rows_n)
            memcpy(e,m->vis_rows+(int64_t)vis_row*H,(size_t)H*sizeof(float));
        else q38_weight_row(&m->embed,ids[s],e);
        for(int b=1;b<C;b++)memcpy(e+(int64_t)b*H,e,(size_t)H*sizeof(float));
    }
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=0;i<c->layers;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){float *ple=falloc((int64_t)S*W);q38_ple(m,ids,S,hyper,ple);for(int64_t z=0;z<(int64_t)S*W;z++)hyper[z]+=ple[z];free(ple);
            /* consumate: il chunk successivo ha altri token, e riusare queste
             * righe darebbe gli embedding del chunk precedente combaciando in
             * silenzio invece di dare errore. */
            free(m->ple_pref); m->ple_pref=NULL; m->ple_pref_rows=0;}
        q38_gr_read(m,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i])q38_attention(m,l,i,mixed,S,pos_base,block);else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(m,&l->mlp_gr,hyper,S,mixed,inject);q38_moe(m,l,i,mixed,S,block);q38_gr_apply(c,hyper,block,inject,S);
    }
    q38_gr_read(m,&m->final_gr,hyper,S,mixed,NULL);m->kv_len=pos_base+S;
    /* Rewinding and writing a shorter branch invalidates its old tail. */
    if(m->kvp.len>pos_base)m->kvp.len=pos_base;
    kv_prefix_record(&m->kvp,ids,pos_base,S);
    if(m->vis_map && m->vis_rows_n>0)kv_prefix_taint(&m->kvp);
    float *logit=falloc(c->vocab);double phase_started=now_s();
    /* Lettura del prefill: la posizione p predice il token p+1. Il primo token
     * fresco lo predice la fotografia del prefisso, quando c'e. Pagata solo da
     * chi ha chiesto il canale. */
    if(g_echo_k>0&&g_echo_id&&S>0){
        if(g_echo_pin_logit)
            q38_echo(g_echo_id,pos_base,ids[0],g_echo_pin_logit,c->vocab,g_echo_k);
        for(int p=0;p+1<S;p++){
            q38_weight_matmul(logit,mixed+(int64_t)p*H,&m->lm_head,1,H,c->vocab);
            q38_echo(g_echo_id,pos_base+p+1,ids[p+1],logit,c->vocab,g_echo_k);
        }
    }
    q38_weight_matmul(logit,mixed+(int64_t)(S-1)*H,&m->lm_head,1,H,c->vocab);
    q38_tm_add(m,Q38_TM_LM_HEAD,phase_started);
    free(hyper);free(mixed);free(inject);free(block);return logit;
}

static int q38_tm_enabled(void) {
    const char *enabled=getenv("COLI_TIMERS");
    return enabled&&enabled[0]=='1'&&enabled[1]=='\0';
}

static void q38_tm_report_bank(const Q38Timers *timers,const char *scope) {
    if(!q38_tm_enabled())return;
    static const char *names[Q38_TM_COUNT]={
        "expert-read","fp8-expand","routed-expert","shared-expert",
        "resident-mm","deltanet","qsa-index","qsa-attn","ple","lm-head"
    };
    double per=timers->forwards?1000.0/timers->forwards:0.0;
    fprintf(stderr,"[qwen38 timers] %s: %llu forwards\n",scope,
            (unsigned long long)timers->forwards);
    for(int i=0;i<Q38_TM_COUNT;i++)
        fprintf(stderr,"[qwen38 timers]   %-14s %9.3f s  %9.3f ms/forward\n",
                names[i],timers->seconds[i],timers->seconds[i]*per);
    fprintf(stderr,"[qwen38 timers] architecture phases overlap resident-mm; "
                   "expert-read is disk service and fp8-expand is synchronous miss work\n");
}

/* Snapshot of the timer bank taken when the prompt's forward has finished
 * (generate() sets it), so a "decode" bank can be reported apart from the
 * prefill. Without it the per-forward figures of a long prompt are prefill
 * work divided by decode forwards, which reads like a decode budget and is
 * not one. */
static Q38Timers g_tm_prefill_snapshot; static int g_tm_have_snapshot;
static void q38_tm_snapshot_prefill(const Model *m) {
    g_tm_prefill_snapshot=m->timers; g_tm_have_snapshot=1;
}
static void q38_cache_v5_report(const Model *m){
    if(!getenv("Q38_CACHE_TELEMETRY"))return;
    uint64_t q=m->q4_cache_hits+m->q4_cache_misses;
    double hit=q?100.0*(double)m->q4_cache_hits/(double)q:0.0;
    double mib=(g_q38_q4.ready?(double)m->q4_cache_misses*(double)g_q38_q4.record_bytes/1048576.0:0.0);
    const char *policy=q38_cache_lfu_on()?"lfu":"lru";
    fprintf(stderr,"[q38 cache v5] policy=%s q4-hit=%llu q4-miss=%llu hit-rate=%.2f%% q4-read=%.1f MiB\n",
            policy,(unsigned long long)m->q4_cache_hits,
            (unsigned long long)m->q4_cache_misses,hit,mib);
    if(g_q38_v51_q4_batches){
        fprintf(stderr,"[q38 cache v5.1] batches=%llu jobs=%llu avg=%.2f hist=",
                (unsigned long long)g_q38_v51_q4_batches,
                (unsigned long long)g_q38_v51_q4_jobs,
                (double)g_q38_v51_q4_jobs/(double)g_q38_v51_q4_batches);
        int maxbin=m->c.topk<Q38_MAX_TOPK?m->c.topk:Q38_MAX_TOPK;
        for(int i=1;i<=maxbin;i++)
            if(g_q38_v51_q4_hist[i])
                fprintf(stderr,"%s%d:%llu",i==1?"":" ",i,
                        (unsigned long long)g_q38_v51_q4_hist[i]);
        fprintf(stderr,"\n");
    if(g_q38_v6_groups){
        double req_per_group=(double)g_q38_v6_requested_records/(double)g_q38_v6_groups;
        double phys_per_group=(double)g_q38_v6_physical_records/(double)g_q38_v6_groups;
        double overhead=g_q38_v6_requested_records?
            100.0*((double)g_q38_v6_physical_records/(double)g_q38_v6_requested_records-1.0):0.0;
        const char *mode=getenv("Q38_Q4_IO_MODE");
        if(!mode||!*mode)mode="base";
        fprintf(stderr,
                "[q38 io v6] mode=%s groups=%llu requested=%llu physical=%llu req/group=%.2f phys/group=%.2f extra-read=%.1f%% copied=%.1f MiB hist=",
                mode,
                (unsigned long long)g_q38_v6_groups,
                (unsigned long long)g_q38_v6_requested_records,
                (unsigned long long)g_q38_v6_physical_records,
                req_per_group,phys_per_group,overhead,
                g_q38_v6_copied_bytes/1048576.0);
        for(int i=1;i<=m->c.topk&&i<=Q38_MAX_TOPK;i++)
            if(g_q38_v6_group_hist[i])
                fprintf(stderr,"%s%d:%llu",i==1?"":" ",i,
                        (unsigned long long)g_q38_v6_group_hist[i]);
        fprintf(stderr,"\n");
#ifdef _WIN32
    if(getenv("Q38_MEGA_TELEMETRY"))
        fprintf(stderr,"[q38 mega v7] cpu-multi=%d event-pool=%d event-reuses=%llu\n",
                q38_mega_cpu_multi_on(),q38_v7_event_pool_on(),
                (unsigned long long)g_q38_v7_event_reuses);
#endif
    }
    }
}
static void tm_report(const Model *m) {
    q38_tm_report_bank(&m->timers,"total");
    if(g_tm_have_snapshot&&q38_tm_enabled()){
        Q38Timers decode=m->timers;
        for(int i=0;i<Q38_TM_COUNT;i++)decode.seconds[i]-=g_tm_prefill_snapshot.seconds[i];
        decode.forwards-=g_tm_prefill_snapshot.forwards;
        q38_tm_report_bank(&g_tm_prefill_snapshot,"prefill");
        q38_tm_report_bank(&decode,"decode");
    }
    if(q38_tm_enabled())
        fprintf(stderr,"[qwen38 expert I/O] weight-ranges=%llu scale-ranges=%llu "
                       "coalesced-gate-up=%llu prefetched=%llu parallel-batches=%llu "
                       "resident-scales=%.2f MiB\n",
                (unsigned long long)m->expert_weight_reads,
                (unsigned long long)m->expert_scale_reads,
                (unsigned long long)m->expert_pair_reads,
                (unsigned long long)m->expert_prefetch_ranges,
                (unsigned long long)m->expert_parallel_batches,
                m->expert_scale_bytes/1048576.0);
    q38_cache_v5_report(m);
    q38_v4_report();
}

static void q38_v4_report(void){
    if(!q38_v4_on("Q38_Q4_PROFILE")||!g_q38_v4_async_batches)return;
    fprintf(stderr,"[q38 v4] async-q4 batches=%llu missing-jobs=%llu io-thread-wall=%.3fs hidden-under-shared=%.3fs\n",
            (unsigned long long)g_q38_v4_async_batches,
            (unsigned long long)g_q38_v4_async_jobs,
            g_q38_v4_async_wall_s,g_q38_v4_async_hidden_s);
}
static void q38_layer_free(Layer *l) {
    if(!l) return;
    free(l->attn_gr.norm);q38_weight_free(&l->attn_gr.down);q38_weight_free(&l->attn_gr.up);q38_weight_free(&l->attn_gr.inject);
    free(l->mlp_gr.norm);q38_weight_free(&l->mlp_gr.down);q38_weight_free(&l->mlp_gr.up);q38_weight_free(&l->mlp_gr.inject);
    q38_weight_free(&l->router);q38_weight_free(&l->sh_g);q38_weight_free(&l->sh_u);q38_weight_free(&l->sh_d);free(l->sh_gate);
    q38_weight_free(&l->q);q38_weight_free(&l->k);q38_weight_free(&l->v);q38_weight_free(&l->o);free(l->qn);free(l->kn);
    q38_weight_free(&l->idx_qk);free(l->idx_qn);free(l->idx_kn);
    q38_weight_free(&l->dn_qkv);q38_weight_free(&l->dn_z);q38_weight_free(&l->dn_b);q38_weight_free(&l->dn_a);free(l->dn_conv);
    free(l->dn_dtbias);free(l->dn_alog);free(l->dn_norm);q38_weight_free(&l->dn_out);
    q38_weight_free(&l->ple_key);q38_weight_free(&l->ple_value);free(l->ple_norm_key);free(l->ple_norm_query);
    free(l->ple_norm_conv); free(l->ple_conv);
    memset(l,0,sizeof(*l));
}

static void q38_model_free(Model *m) {
    if(!m) return;
    kv_prefix_free(&m->kvp);
    for(int i=0;i<m->c.layers;i++) {
        if(m->L)q38_layer_free(&m->L[i]);
        if(m->cache) {
            if(m->cache[i].slots) {
                for(int s=0;s<m->cache[i].n;s++) {
                    q38_weight_free(&m->cache[i].slots[s].gate);
                    q38_weight_free(&m->cache[i].slots[s].up);
                    q38_weight_free(&m->cache[i].slots[s].down);
                    free(m->cache[i].slots[s].fp8_slab);
                    free(m->cache[i].slots[s].q4_slab);
                }
            }
            free(m->cache[i].slots); free(m->cache[i].by_expert); free(m->cache[i].freq);
        }
        if(m->expert_scales)free(m->expert_scales[i].values);
        free(m->DN_rec ? m->DN_rec[i] : NULL); free(m->DN_conv ? m->DN_conv[i] : NULL);
        free(m->K ? m->K[i] : NULL); free(m->V ? m->V[i] : NULL); free(m->IK ? m->IK[i] : NULL);
    }
    free(m->L); free(m->cache); free(m->expert_scales); free(m->DN_rec); free(m->DN_conv); free(m->K); free(m->V); free(m->IK);
    q38_weight_free(&m->embed);q38_weight_free(&m->lm_head);
    free(m->final_gr.norm);q38_weight_free(&m->final_gr.down);q38_weight_free(&m->final_gr.up);q38_weight_free(&m->final_gr.inject);
    free(m->ple_history); free(m->PLE_conv_state); free(m->c.is_attn); st_destroy(&m->S);
    memset(m,0,sizeof(*m));
}

#endif /* COLI_QWEN38_CORE_H */










