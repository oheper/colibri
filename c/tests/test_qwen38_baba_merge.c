/* Packed BABA Q4 beside main's MTP/Vulkan interfaces: no downloaded model.
 * Distinct expert records pin small-cache eviction, serial/batched parity,
 * and verify rows using the same quantized weights as decode. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static void put32(unsigned char *p, uint32_t v) {
    for (int i=0;i<4;i++) p[i]=(unsigned char)(v>>(8*i));
}
static void put64(unsigned char *p, uint64_t v) {
    for (int i=0;i<8;i++) p[i]=(unsigned char)(v>>(8*i));
}

static int packed_fixture(const char *path) {
    unsigned char header[64]={0};
    memcpy(header,"Q38Q4G1",8);
    put32(header+8,1);put32(header+12,1);put32(header+16,4);
    put32(header+20,64);put32(header+24,64);put32(header+28,64);
    const size_t codes=64*64/2, scales=64;
    const size_t bytes=3*codes+3*scales*sizeof(float);
    put64(header+36,bytes);
    FILE *f=fopen(path,"wb");CHECK(f);
    CHECK(fwrite(header,1,sizeof header,f)==sizeof header);
    unsigned char *record=malloc(bytes);CHECK(record);
    for(int e=0;e<4;e++){
        for(size_t j=0;j<3*codes;j++)
            record[j]=(unsigned char)(0x99+e*0x11);
        float *sc=(float*)(record+3*codes);
        for(size_t j=0;j<3*scales;j++)sc[j]=0.001f*(float)(1+e);
        CHECK(fwrite(record,1,bytes,f)==bytes);
    }
    free(record);CHECK(fclose(f)==0);return 0;
}

static void reset_cache(Model *m,int cap) {
    LCache *lc=m->cache;
    for(int i=0;i<lc->n;i++){
        q38_weight_free(&lc->slots[i].gate);
        q38_weight_free(&lc->slots[i].up);
        q38_weight_free(&lc->slots[i].down);
        free(lc->slots[i].q4_slab);
        free(lc->slots[i].fp8_slab);
        free(lc->slots[i].int4_slab);
    }
    memset(lc->slots,0,4*sizeof(Slot));lc->n=0;lc->cap=cap;
    for(int i=0;i<4;i++)lc->by_expert[i]=-1;
    g_q38_q4_reset[0]=0;
}

int main(void) {
    char dir[]="test_qwen38_baba_XXXXXX";CHECK(mkdtemp(dir));
    char path[256];snprintf(path,sizeof path,"%s/packed.bin",dir);
    CHECK(packed_fixture(path)==0);
    setenv("Q38_Q4_SIDECAR",path,1);
    setenv("Q38_CPU_V3","1",1);
    setenv("Q38_XF_SIMD_ACT","1",1);
    setenv("Q38_XF_FUSED_DOWN","1",1);
    setenv("Q38_Q4_TLS_SCRATCH","1",1);
    setenv("Q38_Q4_MMAP","0",1);
    setenv("Q38_Q4_IO_MODE","base",1);
    Model m={0};m.c.layers=1;m.c.hidden=64;m.c.inter=64;
    m.c.shared_inter=64;m.c.experts=4;m.c.topk=2;m.c.norm_topk=1;
    m.native_fp8=1;m.prefill_batch=1;m.expert_parallel_reads=1;
    m.cache=calloc(2,sizeof(LCache));CHECK(m.cache);
    LCache *lc=m.cache;lc->slots=calloc(4,sizeof(Slot));
    lc->by_expert=malloc(4*sizeof(int));lc->freq=calloc(4,sizeof(uint16_t));
    CHECK(lc->slots&&lc->by_expert&&lc->freq);
    Layer l={0};float router[4*64]={0},weights[64*64]={0},shgate[64]={0};
    l.router=(Q38Weight){.data=router,.rows=4,.cols=64,.kind=Q38_WEIGHT_F32};
    l.sh_g=l.sh_u=l.sh_d=(Q38Weight){.data=weights,.rows=64,.cols=64,.kind=Q38_WEIGHT_F32};
    l.sh_gate=shgate;
    float x[3*64],baseline[3*64],out[3*64];
    for(int i=0;i<3*64;i++)x[i]=0.1f+sinf((float)i*0.3f);
    reset_cache(&m,4);q38_moe_decode(&m,&l,0,x,3,baseline);
    CHECK(g_q38_q4_decode_scope==0);
    CHECK(lc->slots[0].q4_active);
    CHECK(fabsf(baseline[0])>1e-8f);
    for(int cap=1;cap<=4;cap++)for(int parallel=0;parallel<=1;parallel++){
        reset_cache(&m,cap);m.expert_parallel_reads=parallel;
        q38_moe_decode(&m,&l,0,x,3,out);
        CHECK(memcmp(out,baseline,sizeof out)==0);
        /* Speculative verify: prefill enabled but the packed format must stay
         * on the decode kernel. No use of the original FP8 tensors. */
        g_q38_rowwise=1;
        q38_moe_ex(&m,&l,0,x,3,out,NULL,0);
        g_q38_rowwise=0;
        CHECK(memcmp(out,baseline,sizeof out)==0);
        CHECK(g_q38_q4_decode_scope==0);
        /* Windows uses one-wave streamed prefill here; other platforms use
         * packed decode rows. Both must compute these same expert records. */
        q38_moe_ex(&m,&l,0,x,3,out,NULL,0);
        CHECK(memcmp(out,baseline,sizeof out)==0);
    }
    /* The MTP head's layer has no record in the packed sidecar. */
    CHECK(q38_layer_int4(&m,1)==0);
    Slot head={0};g_q38_q4_decode_scope=1;
    CHECK(q38_q4_load_v7(&m,1,0,&head)==0);
    g_q38_q4_decode_scope=0;
    reset_cache(&m,4);free(lc->slots);free(lc->by_expert);free(lc->freq);
    if(m.ehit){free(m.ehit[0]);free(m.ehit);}
    free(m.cache);close(g_q38_q4.fd);g_q38_q4.ready=0;g_q38_q4.fd=-1;
#ifdef _WIN32
    if(g_q38_q4_win_handle!=INVALID_HANDLE_VALUE){
        CloseHandle(g_q38_q4_win_handle);g_q38_q4_win_handle=INVALID_HANDLE_VALUE;
    }
#endif
    CHECK(remove(path)==0);CHECK(rmdir(dir)==0);
    puts("BABA packed Q4: cache caps 1-4, serial/parallel, verify/decode parity: ok");
    return 0;
}
