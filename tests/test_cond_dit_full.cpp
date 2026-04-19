// Test: Full 22-block DiT with non-zero conditioning, batch=2 CFG.
// Gold: python3 tests/gen_cond_dit_gold.py → /tmp/cv_cond_dit_gold/
// Tests each block output progressively to find where C++ diverges from Python.

#include <catch2/catch_test_macros.hpp>
#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const int D=1024, HEADS=16, HD=D/HEADS, DEPTH=22, MEL=80, SEQ=20, BATCH=2, HALF=128;
static const char* GOLD="/tmp/cv_cond_dit_gold";

struct gold_t { std::vector<int64_t> shape; std::vector<float> data; int64_t n() const {return (int64_t)data.size();} };

static bool load_gold(const char* name, gold_t& out) {
    std::string path = std::string(GOLD)+"/"+name+".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t nd; f.read((char*)&nd,4);
    out.shape.resize(nd);
    for (uint32_t i=0;i<nd;i++){int64_t d;f.read((char*)&d,8);out.shape[i]=d;}
    int64_t n=1; for(auto d:out.shape) n*=d;
    out.data.resize(n);
    f.read((char*)out.data.data(),n*4);
    return f.good();
}

struct gguf_file {
    gguf_context* gc=nullptr; ggml_context* mc=nullptr;
    gguf_file(const char* p){gc=gguf_init_from_file(p,{.no_alloc=false,.ctx=&mc});}
    ~gguf_file(){if(gc)gguf_free(gc);if(mc)ggml_free(mc);}
    operator bool()const{return gc!=nullptr;}
    ggml_tensor* t(const char* n)const{return ggml_get_tensor(mc,n);}
};

static ggml_tensor* mk(ggml_context* c, int64_t n0, int64_t n1=1, int64_t n2=1) {
    auto*t=ggml_new_tensor_3d(c,GGML_TYPE_F32,n0,n1,n2); ggml_set_input(t); return t;
}
static void fill_gguf(ggml_tensor* dst, ggml_tensor* src) {
    int64_t n=ggml_nelements(src);
    if(src->type==GGML_TYPE_F32){ggml_backend_tensor_set(dst,src->data,0,n*4);}
    else{std::vector<float>b(n);ggml_get_type_traits(src->type)->to_float(src->data,b.data(),n);ggml_backend_tensor_set(dst,b.data(),0,n*4);}
}
static ggml_tensor* linop(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    auto*y=ggml_mul_mat(c,w,x);
    if(b){auto*bv=ggml_view_4d(c,b,b->ne[0],1,1,1,b->nb[0],b->nb[0],b->nb[0],0);y=ggml_add(c,y,bv);}
    return y;
}
static ggml_tensor* unsq1(ggml_context* c, ggml_tensor* x){
    return ggml_view_4d(c,x,x->ne[0],1,x->ne[1],x->ne[2],x->nb[1],x->nb[1],x->nb[2],0);
}
static double compare(const float* a, const float* b, int64_t n, const char* label) {
    double mx=0, sm=0;
    for(int64_t i=0;i<n;i++){double d=std::abs((double)a[i]-(double)b[i]);mx=std::max(mx,d);sm+=d;}
    printf("  %s: max=%.4f mad=%.4f\n",label,mx,sm/n);
    return mx;
}

static const int CPE_KERNEL=31, CPE_GROUPS=16, CPE_CH_PER_GROUP=D/CPE_GROUPS;

static ggml_tensor* grouped_conv1d(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int g) {
    auto*xs=(ggml_tensor**)alloca(sizeof(ggml_tensor*)*g);
    auto*ws=(ggml_tensor**)alloca(sizeof(ggml_tensor*)*g);
    int64_t cpg=x->ne[1]/g;
    for(int i=0;i<g;i++){
        xs[i]=ggml_view_3d(c,x,x->ne[0],cpg,x->ne[2],x->nb[1],x->nb[2],i*cpg*x->nb[1]);
        ws[i]=ggml_view_3d(c,w,w->ne[0],w->ne[1],w->ne[2]/g,w->nb[1],w->nb[2],i*(w->ne[2]/g)*w->nb[2]);
    }
    for(int i=0;i<g;i++){
        auto*im=ggml_im2col(c,ws[i],ggml_cont(c,xs[i]),1,0,0,0,1,0,false,GGML_TYPE_F32);
        xs[i]=ggml_view_4d(c,im,im->ne[0],im->ne[1],1,im->ne[2],im->nb[1],im->nb[1],im->nb[2],0);
    }
    auto*cat=xs[0]; for(int i=1;i<g;i++) cat=ggml_concat(c,cat,xs[i],2);
    ggml_set_output(cat);
    auto*im=ggml_cont(c,cat);
    auto*wr=ggml_reshape_3d(c,w,w->ne[0]*w->ne[1],w->ne[2]/g,g);
    if(x->ne[2]!=1) wr=ggml_repeat_4d(c,wr,wr->ne[0],wr->ne[1],g,x->ne[2]);
    auto*r=ggml_mul_mat(c,im,wr);
    r=ggml_reshape_3d(c,r,im->ne[1],w->ne[2],x->ne[2]);
    if(b){auto*bv=ggml_reshape_3d(c,b,1,b->ne[0],1);r=ggml_add(c,r,bv);}
    return r;
}

static ggml_tensor* ggml_mish(ggml_context* c, ggml_tensor* x){
    return ggml_mul(c,x,ggml_tanh(c,ggml_softplus(c,x)));
}

struct BW { ggml_tensor *aw,*ab,*qw,*qb,*kw,*kb,*vw,*vb,*ow,*ob,*fw,*fb,*dw,*db; };

static ggml_tensor* dit_block(ggml_context* c, ggml_tensor* x, ggml_tensor* emb, ggml_tensor* pos, const BW& w, int seq, int batch, bool use_flash_attn=false){
    auto*es=ggml_silu(c,emb);
    auto*ep=linop(c,es,w.aw,w.ab);
    int64_t ch=ep->ne[0]/6;
    auto v6=[&](int i)->ggml_tensor*{return ggml_cont(c,ggml_view_1d(c,ep,ch,i*ch*4));};
    auto*sm=v6(0),*cm=ggml_scale_bias(c,v6(1),1.f,1.f),*gm=v6(2);
    auto*sf=v6(3),*cf=ggml_scale_bias(c,v6(4),1.f,1.f),*gf=v6(5);
    auto*xn=ggml_norm(c,x,1e-6f);
    xn=ggml_mul(c,xn,unsq1(c,cm)); xn=ggml_add(c,xn,unsq1(c,sm));
    auto*q=linop(c,xn,w.qw,w.qb),*k=linop(c,xn,w.kw,w.kb),*v=linop(c,xn,w.vw,w.vb);
    q=ggml_reshape_3d(c,q,HD,HEADS,seq*batch);
    k=ggml_reshape_3d(c,k,HD,HEADS,seq*batch);
    auto*p1=ggml_reshape_1d(c,pos,seq*batch);
    q=ggml_rope(c,q,p1,HD,GGML_ROPE_TYPE_NORMAL);
    k=ggml_rope(c,k,p1,HD,GGML_ROPE_TYPE_NORMAL);
    q=ggml_reshape_4d(c,q,HD,HEADS,seq,batch);
    k=ggml_reshape_4d(c,k,HD,HEADS,seq,batch);
    v=ggml_reshape_4d(c,v,HD,HEADS,seq,batch);
    q=ggml_cont(c,ggml_permute(c,q,0,2,1,3));
    k=ggml_cont(c,ggml_permute(c,k,0,2,1,3));
    v=ggml_permute(c,v,0,2,1,3);
    ggml_tensor* ao;
    if (use_flash_attn) {
        ao=ggml_flash_attn_ext(c,q,k,v,nullptr,1.f/sqrtf((float)HD),0.f,0.f);
    } else {
        auto*sc=ggml_scale(c,ggml_mul_mat(c,k,q),1.f/sqrtf((float)HD));
        auto*aw_=ggml_soft_max(c,sc);
        auto*vp=ggml_cont(c,ggml_permute(c,v,1,0,2,3));
        ao=ggml_cont(c,ggml_permute(c,ggml_mul_mat(c,vp,aw_),0,2,1,3));
    }
    ao=ggml_reshape_3d(c,ao,D,seq,batch);
    ao=linop(c,ao,w.ow,w.ob);
    ao=ggml_mul(c,ao,unsq1(c,gm));
    x=ggml_add(c,x,ao);
    auto*fn=ggml_norm(c,x,1e-6f);
    fn=ggml_mul(c,fn,unsq1(c,cf)); fn=ggml_add(c,fn,unsq1(c,sf));
    auto*ff=ggml_gelu_erf(c,linop(c,fn,w.fw,w.fb));
    ff=linop(c,ff,w.dw,w.db);
    ff=ggml_mul(c,ff,unsq1(c,gf));
    return ggml_add(c,x,ff);
}

SCENARIO("Full DiT with non-zero conditioning matches PyTorch gold",
         "[e2e][cond_dit]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }

    gold_t gx,gc_,gm,gs,gte,gie,gie_full;
    gold_t g_conv1_out, g_conv1_mish, g_conv2_out, g_conv2_mish;
    bool ok = load_gold("x",gx)&&load_gold("cond",gc_)&&load_gold("mu",gm)&&
              load_gold("spks",gs)&&load_gold("time_emb",gte)&&load_gold("ie_proj",gie)&&
              load_gold("ie_full",gie_full);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}
    // Sub-op gold (optional)
    bool has_subop = load_gold("cpe_conv1_out",g_conv1_out)&&load_gold("cpe_conv1_mish",g_conv1_mish)&&
                     load_gold("cpe_conv2_out",g_conv2_out)&&load_gold("cpe_conv2_mish",g_conv2_mish);

    gguf_file model(mp);
    REQUIRE(bool(model));

    const size_t csz=ggml_tensor_overhead()*2000+1024LL*1024*200;
    auto*ctx=ggml_init({.mem_size=csz,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx);

    // Create input tensors
    auto*x=mk(ctx,MEL,SEQ,BATCH), *cond=mk(ctx,MEL,SEQ,BATCH), *mu_t=mk(ctx,MEL,SEQ,BATCH), *spks_t=mk(ctx,MEL,SEQ,BATCH);
    auto*te=mk(ctx,D);

    // Load proj weight
    auto*pw=mk(ctx,320,D), *pb=mk(ctx,D);

    // Load block weights
    BW bw[DEPTH];
    auto*raw_pw=model.t("decoder.estimator.input_embed.proj.weight");
    auto*raw_pb=model.t("decoder.estimator.input_embed.proj.bias");

    for(int i=0;i<DEPTH;i++){
        char pfx[128];
        snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto tn=[&](const char*s)->ggml_tensor*{char n[256];snprintf(n,256,"%s.%s",pfx,s);auto*t=model.t(n);REQUIRE(t);return t;};
        auto mkw=[&](ggml_tensor*src)->ggml_tensor*{
            auto*t=ggml_new_tensor(ctx,GGML_TYPE_F32,GGML_MAX_DIMS,src->ne);ggml_set_input(t);return t;};
        bw[i]={mkw(tn("attn_norm.linear.weight")),mkw(tn("attn_norm.linear.bias")),
               mkw(tn("attn.to_q.weight")),mkw(tn("attn.to_q.bias")),
               mkw(tn("attn.to_k.weight")),mkw(tn("attn.to_k.bias")),
               mkw(tn("attn.to_v.weight")),mkw(tn("attn.to_v.bias")),
               mkw(tn("attn.to_out.0.weight")),mkw(tn("attn.to_out.0.bias")),
               mkw(tn("ff.ff.0.0.weight")),mkw(tn("ff.ff.0.0.bias")),
               mkw(tn("ff.ff.2.weight")),mkw(tn("ff.ff.2.bias"))};
    }

    // conv_pos_embed weights
    auto*raw_c1w=model.t("decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight");
    auto*raw_c1b=model.t("decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias");
    auto*raw_c2w=model.t("decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight");
    auto*raw_c2b=model.t("decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias");
    REQUIRE(raw_c1w); REQUIRE(raw_c2w);

    auto*c1w=mk(ctx,CPE_KERNEL,CPE_CH_PER_GROUP,D); auto*c1b=mk(ctx,D);
    auto*c2w=mk(ctx,CPE_KERNEL,CPE_CH_PER_GROUP,D); auto*c2b=mk(ctx,D);

    // Build graph: InputEmbedding proj + conv_pos_embed + 22 DiT blocks
    auto*cat_in=ggml_concat(ctx,x,cond,0);
    cat_in=ggml_concat(ctx,cat_in,mu_t,0);
    cat_in=ggml_concat(ctx,cat_in,spks_t,0);
    auto*proj_out=linop(ctx,cat_in,pw,pb);
    ggml_set_output(proj_out);

    // conv_pos_embed: [D, SEQ, 2] → permute → [SEQ, D, 2] → causal conv → permute back → add residual
    auto*cpe=ggml_permute(ctx,proj_out,1,0,2,3);  // [D,SEQ,2] → [SEQ,D,2]
    cpe=ggml_cont(ctx,cpe);
    cpe=ggml_pad_ext(ctx,cpe,CPE_KERNEL-1,0,0,0,0,0,0,0);
    ggml_set_output(cpe);
    cpe=grouped_conv1d(ctx,cpe,c1w,c1b,CPE_GROUPS);  // conv1
    auto*cpe_conv1_out=cpe; ggml_set_output(cpe_conv1_out);
    cpe=ggml_mish(ctx,cpe);
    auto*cpe_conv1_mish=cpe; ggml_set_output(cpe_conv1_mish);
    cpe=ggml_pad_ext(ctx,cpe,CPE_KERNEL-1,0,0,0,0,0,0,0);
    cpe=grouped_conv1d(ctx,cpe,c2w,c2b,CPE_GROUPS);  // conv2
    auto*cpe_conv2_out=cpe; ggml_set_output(cpe_conv2_out);
    cpe=ggml_mish(ctx,cpe);
    auto*cpe_conv2_mish=cpe; ggml_set_output(cpe_conv2_mish);
    cpe=ggml_permute(ctx,cpe,1,0,2,3);  // back to [D, SEQ, 2]
    cpe=ggml_cont(ctx,cpe);
    auto*ie_full=ggml_add(ctx,proj_out,cpe);  // residual
    ggml_set_output(ie_full);

    auto*pos=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,SEQ,BATCH);
    ggml_set_input(pos);

    auto*cur=ie_full;
    ggml_tensor* block_outs[DEPTH];
    for(int i=0;i<DEPTH;i++){
        cur=dit_block(ctx,cur,te,pos,bw[i],SEQ,BATCH);
        ggml_set_output(cur);
        block_outs[i]=cur;
    }

    // Allocate
    auto*backend=ggml_backend_cpu_init();
    REQUIRE(backend);
    auto*buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    REQUIRE(buf);

    // Fill inputs from gold
    // Gold x is [20,80] numpy C-contiguous = ggml ne=[80,20] flat layout
    // But our x tensor is [80, 20, 2] where batch[0]=real, batch[1]=zeros
    // We need to fill batch[0] from gold and batch[1] from zeros
    {
        int64_t bf = MEL*SEQ;
        std::vector<float> xd(bf*2,0.f), cd(bf*2,0.f), md(bf*2,0.f), sd(bf*2,0.f);
        // x: SAME for both batches (CFG shares noise/state)
        memcpy(xd.data(), gx.data.data(), bf*4);       // batch[0]
        memcpy(xd.data()+bf, gx.data.data(), bf*4);    // batch[1] = same x
        // cond, mu, spks: batch[0]=real, batch[1]=zeros
        memcpy(cd.data(), gc_.data.data(), bf*4);
        memcpy(md.data(), gm.data.data(), bf*4);
        // spks is [1,80] in gold, expand to [80, SEQ] for batch[0]
        for(int s=0;s<SEQ;s++)
            memcpy(sd.data()+s*MEL, gs.data.data(), MEL*4);
        ggml_backend_tensor_set(x,xd.data(),0,xd.size()*4);
        ggml_backend_tensor_set(cond,cd.data(),0,cd.size()*4);
        ggml_backend_tensor_set(mu_t,md.data(),0,md.size()*4);
        ggml_backend_tensor_set(spks_t,sd.data(),0,sd.size()*4);
    }
    ggml_backend_tensor_set(te,gte.data.data(),0,gte.n()*4);

    // Fill proj weights
    fill_gguf(pw,raw_pw);
    ggml_backend_tensor_set(pb,raw_pb->data,0,raw_pb->ne[0]*4);

    // Fill conv_pos_embed weights
    fill_gguf(c1w,raw_c1w);
    ggml_backend_tensor_set(c1b,raw_c1b->data,0,raw_c1b->ne[0]*4);
    fill_gguf(c2w,raw_c2w);
    ggml_backend_tensor_set(c2b,raw_c2b->data,0,raw_c2b->ne[0]*4);

    // Fill block weights
    for(int i=0;i<DEPTH;i++){
        char pfx[128];
        snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto fill=[&](const char*s,ggml_tensor*dst){
            char n[256];snprintf(n,256,"%s.%s",pfx,s);
            fill_gguf(dst,model.t(n));};
        fill("attn_norm.linear.weight",bw[i].aw);
        fill("attn_norm.linear.bias",bw[i].ab);
        fill("attn.to_q.weight",bw[i].qw); fill("attn.to_q.bias",bw[i].qb);
        fill("attn.to_k.weight",bw[i].kw); fill("attn.to_k.bias",bw[i].kb);
        fill("attn.to_v.weight",bw[i].vw); fill("attn.to_v.bias",bw[i].vb);
        fill("attn.to_out.0.weight",bw[i].ow); fill("attn.to_out.0.bias",bw[i].ob);
        fill("ff.ff.0.0.weight",bw[i].fw); fill("ff.ff.0.0.bias",bw[i].fb);
        fill("ff.ff.2.weight",bw[i].dw); fill("ff.ff.2.bias",bw[i].db);
    }

    // Fill position ids
    std::vector<int32_t> pids(SEQ*BATCH);
    for(int i=0;i<SEQ*BATCH;i++) pids[i]=i%SEQ;
    ggml_backend_tensor_set(pos,pids.data(),0,pids.size()*4);

    // Build and execute
    auto*graph=ggml_new_graph_custom(ctx,8192,false);
    ggml_build_forward_expand(graph,cur);

    auto status=ggml_backend_graph_compute(backend,graph);
    REQUIRE(status==GGML_STATUS_SUCCESS);

    // Compare proj output
    {
        std::vector<float> r(ggml_nelements(proj_out));
        ggml_backend_tensor_get(proj_out,r.data(),0,r.size()*4);
        double md=compare(r.data(),gie.data.data(),r.size(),"ie_proj");
        CHECK(md<0.1);
    }

    // Compare conv_pos_embed sub-ops (progressive divergence tracking)
    if (has_subop) {
        auto cmp_tensor=[&](ggml_tensor* t, gold_t& g, const char* name) {
            std::vector<float> r(ggml_nelements(t));
            ggml_backend_tensor_get(t,r.data(),0,r.size()*4);
            return compare(r.data(),g.data.data(),r.size(),name);
        };
        double d1=cmp_tensor(cpe_conv1_out, g_conv1_out, "cpe_conv1");
        double d2=cmp_tensor(cpe_conv1_mish, g_conv1_mish, "cpe_conv1_mish");
        double d3=cmp_tensor(cpe_conv2_out, g_conv2_out, "cpe_conv2");
        double d4=cmp_tensor(cpe_conv2_mish, g_conv2_mish, "cpe_conv2_mish");
        printf("  conv_pos_embed progressive: conv1=%.4f → mish=%.4f → conv2=%.4f → mish=%.4f\n",d1,d2,d3,d4);
    }

    // Compare ie_full (proj + conv_pos_embed)
    {
        std::vector<float> r(ggml_nelements(ie_full));
        ggml_backend_tensor_get(ie_full,r.data(),0,r.size()*4);
        double md=compare(r.data(),gie_full.data.data(),r.size(),"ie_full");
        // This is the key test: if conv_pos_embed introduces bias, this will show it
    }

    // Compare each block
    double max_block_diff = 0;
    int first_bad_block = -1;
    for(int i=0;i<DEPTH;i++){
        char name[32]; snprintf(name,32,"block%d",i);
        gold_t gb;
        if(!load_gold(name,gb)) continue;
        std::vector<float> r(ggml_nelements(block_outs[i]));
        ggml_backend_tensor_get(block_outs[i],r.data(),0,r.size()*4);
        char label[64]; snprintf(label,64,"block%d",i);
        double md=compare(r.data(),gb.data.data(),r.size(),label);
        if(md>max_block_diff) max_block_diff=md;
        if(md>1.0 && first_bad_block<0) first_bad_block=i;
    }

    printf("\nmax_block_diff=%.4f, first_bad_block=%d\n",max_block_diff,first_bad_block);
    CHECK(max_block_diff < 5.0);  // generous for Q8_0

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// Test: ggml_pad CFG batching vs pre-filled 3D tensor
// This tests whether prepare_context's ggml_pad produces the same result
// as manually creating a batch=2 tensor.
SCENARIO("ggml_pad CFG batching produces same ie_proj as pre-filled tensor",
         "[e2e][pad_batch]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }

    gold_t gx,gc_,gm,gs,gie;
    bool ok = load_gold("x",gx)&&load_gold("cond",gc_)&&load_gold("mu",gm)&&
              load_gold("spks",gs)&&load_gold("ie_proj",gie);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    gguf_file model(mp);
    REQUIRE(bool(model));

    const size_t csz=ggml_tensor_overhead()*500+1024LL*1024*50;
    auto*ctx=ggml_init({.mem_size=csz,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx);

    // Inputs as 2D (no batch) — like encoder outputs
    auto*x_2d=mk(ctx,MEL,SEQ);     // [80, 20]
    auto*cond_2d=mk(ctx,MEL,SEQ);
    auto*mu_2d=mk(ctx,MEL,SEQ);
    auto*spks_2d=mk(ctx,MEL,1);    // [80, 1]

    auto*pw=mk(ctx,320,D), *pb=mk(ctx,D);

    // Build graph using ggml_pad (like prepare_context)
    auto*cond_b2=ggml_pad(ctx,cond_2d,0,0,1,0);  // [80,20,1]→[80,20,2]
    auto*mu_b2=ggml_pad(ctx,mu_2d,0,0,1,0);
    auto*spks_b2=ggml_pad(ctx,spks_2d,0,0,1,0);  // [80,1,1]→[80,1,2]
    auto*x_b2=ggml_repeat_4d(ctx,x_2d,x_2d->ne[0],x_2d->ne[1],2,1); // repeat x for both batches

    // Repeat spks along seq dim to match x
    auto*spks_rep=ggml_repeat(ctx,spks_b2,x_b2);  // [80,1,2]→[80,20,2]

    auto*cat=ggml_concat(ctx,x_b2,cond_b2,0);
    cat=ggml_concat(ctx,cat,mu_b2,0);
    cat=ggml_concat(ctx,cat,spks_rep,0);
    auto*proj_out=linop(ctx,cat,pw,pb);
    ggml_set_output(proj_out);

    auto*backend=ggml_backend_cpu_init();
    REQUIRE(backend);
    auto*buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    REQUIRE(buf);

    // Fill 2D inputs from gold (first batch only)
    int64_t bf=MEL*SEQ;
    ggml_backend_tensor_set(x_2d,gx.data.data(),0,bf*4);
    ggml_backend_tensor_set(cond_2d,gc_.data.data(),0,bf*4);
    ggml_backend_tensor_set(mu_2d,gm.data.data(),0,bf*4);
    ggml_backend_tensor_set(spks_2d,gs.data.data(),0,MEL*4);

    auto*raw_pw=model.t("decoder.estimator.input_embed.proj.weight");
    auto*raw_pb=model.t("decoder.estimator.input_embed.proj.bias");
    fill_gguf(pw,raw_pw);
    ggml_backend_tensor_set(pb,raw_pb->data,0,raw_pb->ne[0]*4);

    auto*graph=ggml_new_graph_custom(ctx,2048,false);
    ggml_build_forward_expand(graph,proj_out);
    auto status=ggml_backend_graph_compute(backend,graph);
    REQUIRE(status==GGML_STATUS_SUCCESS);

    // Compare against gold (same gold as pre-filled test)
    std::vector<float> r(ggml_nelements(proj_out));
    ggml_backend_tensor_get(proj_out,r.data(),0,r.size()*4);
    double md=compare(r.data(),gie.data.data(),r.size(),"pad_batch_ie_proj");

    printf("  proj_out ne=[%lld,%lld,%lld]\n",proj_out->ne[0],proj_out->ne[1],proj_out->ne[2]);

    // Should match gold exactly — same weights, same data, just different batching method
    CHECK(md<0.001);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// Test: simulate production pipeline layout (two contexts + sched)
SCENARIO("Two-context ggml_pad + sched produces same result as single context",
         "[e2e][sched_pad]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }

    gold_t gx,gc_,gm,gs,gie;
    bool ok = load_gold("x",gx)&&load_gold("cond",gc_)&&load_gold("mu",gm)&&
              load_gold("spks",gs)&&load_gold("ie_proj",gie);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    gguf_file model(mp);
    REQUIRE(bool(model));

    const size_t csz0=ggml_tensor_overhead()*500+1024LL*1024*50;
    const size_t csz1=ggml_tensor_overhead()*10+1024*1024;
    auto*ctx0=ggml_init({.mem_size=csz0,.mem_buffer=nullptr,.no_alloc=true});
    auto*ctx1=ggml_init({.mem_size=csz1,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx0); REQUIRE(ctx1);

    // "Encoder outputs" in ctx0
    auto*enc_mu=mk(ctx0,MEL,SEQ);
    auto*enc_cond=mk(ctx0,MEL,SEQ);
    auto*enc_spks=mk(ctx0,MEL,1);
    ggml_set_output(enc_mu);
    ggml_set_output(enc_cond);
    ggml_set_output(enc_spks);

    // prepare_context: ggml_pad in ctx1 (cross-context)
    auto*mu_in=ggml_pad(ctx1,enc_mu,0,0,1,0);
    auto*cond_in=ggml_pad(ctx1,enc_cond,0,0,1,0);
    auto*spks_in=ggml_pad(ctx1,enc_spks,0,0,1,0);
    auto*x_in=ggml_new_tensor_2d(ctx1,GGML_TYPE_F32,MEL,SEQ);
    ggml_set_input(x_in);

    auto*pw=mk(ctx0,320,D), *pb=mk(ctx0,D);

    // Build graph in ctx0
    auto*x_b2=ggml_repeat_4d(ctx0,x_in,x_in->ne[0],x_in->ne[1],2,1);
    auto*spks_rep=ggml_repeat(ctx0,spks_in,x_b2);
    auto*cat=ggml_concat(ctx0,x_b2,cond_in,0);
    cat=ggml_concat(ctx0,cat,mu_in,0);
    cat=ggml_concat(ctx0,cat,spks_rep,0);
    auto*proj_out=linop(ctx0,cat,pw,pb);
    ggml_set_output(proj_out);

    // sched
    auto*backend=ggml_backend_cpu_init();
    REQUIRE(backend);
    auto*sched=ggml_backend_sched_new(&backend,nullptr,1,csz0,false,false);
    REQUIRE(sched);

    // Pre-allocate ctx1 in separate buffer (like token2wav_buffer)
    auto*buft=ggml_backend_get_default_buffer_type(backend);
    auto*ctx1_buf=ggml_backend_alloc_ctx_tensors_from_buft(ctx1,buft);
    REQUIRE(ctx1_buf);

    auto*gf=ggml_new_graph_custom(ctx0,2048,false);
    ggml_build_forward_expand(gf,proj_out);
    ggml_backend_sched_alloc_graph(sched,gf);

    // Set inputs AFTER sched allocation
    int64_t bf=MEL*SEQ;
    ggml_backend_tensor_set(enc_mu,gm.data.data(),0,bf*4);
    ggml_backend_tensor_set(enc_cond,gc_.data.data(),0,bf*4);
    ggml_backend_tensor_set(enc_spks,gs.data.data(),0,MEL*4);
    ggml_backend_tensor_set(x_in,gx.data.data(),0,bf*4);

    auto*raw_pw=model.t("decoder.estimator.input_embed.proj.weight");
    auto*raw_pb=model.t("decoder.estimator.input_embed.proj.bias");
    fill_gguf(pw,raw_pw);
    ggml_backend_tensor_set(pb,raw_pb->data,0,raw_pb->ne[0]*4);

    auto status=ggml_backend_sched_graph_compute(sched,gf);
    REQUIRE(status==GGML_STATUS_SUCCESS);

    std::vector<float> r(ggml_nelements(proj_out));
    ggml_backend_tensor_get(proj_out,r.data(),0,r.size()*4);
    double md=compare(r.data(),gie.data.data(),r.size(),"sched_pad_ie_proj");

    printf("  proj_out ne=[%lld,%lld,%lld]\n",proj_out->ne[0],proj_out->ne[1],proj_out->ne[2]);

    CHECK(md<0.001);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(ctx1_buf);
    ggml_backend_free(backend);
    ggml_free(ctx1);
    ggml_free(ctx0);
}


// Test: non-contiguous x (permuted) + contiguous cond/mu in concat.
// Reproduces the exact tensor layout of the production pipeline:
//   x starts as [SEQ, MEL, 2], gets permuted to [MEL, SEQ, 2] (non-contiguous)
//   cond/mu are [MEL, SEQ, 2] contiguous (from ggml_pad)
//   concat(x_noncontig, cond_contig, mu_contig, spks, dim=0) → proj
//
// If this produces different results from the contiguous test,
// we've found the root cause.
SCENARIO("Non-contiguous x from permute produces same proj output as contiguous x",
         "[e2e][noncontig]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }

    gold_t gx,gc_,gm,gs,gie;
    bool ok = load_gold("x",gx)&&load_gold("cond",gc_)&&load_gold("mu",gm)&&
              load_gold("spks",gs)&&load_gold("ie_proj",gie);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    gguf_file model(mp);
    REQUIRE(bool(model));

    const size_t csz=ggml_tensor_overhead()*500+1024LL*1024*50;
    auto*ctx=ggml_init({.mem_size=csz,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx);

    // x as [SEQ, MEL, 2] — will be permuted to [MEL, SEQ, 2] non-contiguous
    // This matches the production pipeline: x starts as ne=[seq, 80] then repeat_4d → [seq, 80, 2]
    auto*x_raw=mk(ctx,SEQ,MEL,BATCH);  // [SEQ=20, MEL=80, 2] — NOTE: transposed!

    // cond/mu as [MEL, SEQ, 2] contiguous — from ggml_pad
    auto*cond_c=mk(ctx,MEL,SEQ,BATCH);
    auto*mu_c=mk(ctx,MEL,SEQ,BATCH);
    auto*spks_c=mk(ctx,MEL,SEQ,BATCH);

    auto*pw=mk(ctx,320,D), *pb=mk(ctx,D);

    // Permute x: [SEQ, MEL, 2] → [MEL, SEQ, 2] (non-contiguous!)
    auto*x_perm=ggml_permute(ctx,x_raw,1,0,2,3);
    // x_perm ne=[MEL, SEQ, 2] but nb=[MEL*4, 4, SEQ*MEL*4] — non-contiguous!
    // (nb[0]=MEL*4 > nb[1]=4, so stride of dim0 > stride of dim1)

    // Concat non-contiguous x with contiguous cond/mu
    auto*cat=ggml_concat(ctx,x_perm,cond_c,0);   // [160, SEQ, 2]
    cat=ggml_concat(ctx,cat,mu_c,0);               // [240, SEQ, 2]
    cat=ggml_concat(ctx,cat,spks_c,0);             // [320, SEQ, 2]
    auto*proj_out=linop(ctx,cat,pw,pb);             // [1024, SEQ, 2]
    ggml_set_output(proj_out);

    auto*backend=ggml_backend_cpu_init();
    REQUIRE(backend);
    auto*buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    REQUIRE(buf);

    // Fill x_raw as [SEQ, MEL, 2] — same data as gold but in transposed layout
    // Gold x is [SEQ=20, MEL=80] numpy C-contiguous: flat[s*80 + d]
    // x_raw is ggml ne=[SEQ=20, MEL=80, 2]: flat[s + d*20 + b*20*80]
    // These are DIFFERENT layouts! Need to transpose the gold data.
    {
        int64_t bf = MEL*SEQ;
        // For x_raw [SEQ, MEL, 2]: need data as flat[s + d*SEQ]
        // Gold is flat[s*MEL + d] = flat[d + s*MEL] — this is what ggml [MEL, SEQ] expects
        // For ggml [SEQ, MEL]: need flat[s + d*SEQ]
        // So we need to transpose: dst[s + d*SEQ] = src[d + s*MEL]
        std::vector<float> x_transposed(bf);
        for (int s=0; s<SEQ; s++)
            for (int d=0; d<MEL; d++)
                x_transposed[s + d*SEQ] = gx.data[s*MEL + d];
        // batch[0] = transposed gold, batch[1] = same (x is repeated in CFG)
        std::vector<float> x_full(bf*2);
        memcpy(x_full.data(), x_transposed.data(), bf*4);
        memcpy(x_full.data()+bf, x_transposed.data(), bf*4);
        ggml_backend_tensor_set(x_raw, x_full.data(), 0, x_full.size()*4);
    }

    // cond/mu/spks filled same as before (contiguous [MEL, SEQ, 2])
    {
        int64_t bf = MEL*SEQ;
        std::vector<float> cd(bf*2,0.f), md(bf*2,0.f), sd(bf*2,0.f);
        memcpy(cd.data(), gc_.data.data(), bf*4);
        memcpy(md.data(), gm.data.data(), bf*4);
        for(int s=0;s<SEQ;s++)
            memcpy(sd.data()+s*MEL, gs.data.data(), MEL*4);
        ggml_backend_tensor_set(cond_c, cd.data(), 0, cd.size()*4);
        ggml_backend_tensor_set(mu_c, md.data(), 0, md.size()*4);
        ggml_backend_tensor_set(spks_c, sd.data(), 0, sd.size()*4);
    }

    auto*raw_pw=model.t("decoder.estimator.input_embed.proj.weight");
    auto*raw_pb=model.t("decoder.estimator.input_embed.proj.bias");
    fill_gguf(pw,raw_pw);
    ggml_backend_tensor_set(pb,raw_pb->data,0,raw_pb->ne[0]*4);

    auto*graph=ggml_new_graph_custom(ctx,2048,false);
    ggml_build_forward_expand(graph,proj_out);
    auto status=ggml_backend_graph_compute(backend,graph);
    REQUIRE(status==GGML_STATUS_SUCCESS);

    std::vector<float> r(ggml_nelements(proj_out));
    ggml_backend_tensor_get(proj_out,r.data(),0,r.size()*4);

    printf("  x_raw ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
           x_raw->ne[0],x_raw->ne[1],x_raw->ne[2],x_raw->nb[0],x_raw->nb[1],x_raw->nb[2]);
    printf("  x_perm ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
           x_perm->ne[0],x_perm->ne[1],x_perm->ne[2],x_perm->nb[0],x_perm->nb[1],x_perm->nb[2]);
    printf("  cond ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
           cond_c->ne[0],cond_c->ne[1],cond_c->ne[2],cond_c->nb[0],cond_c->nb[1],cond_c->nb[2]);

    double md=compare(r.data(),gie.data.data(),r.size(),"noncontig_ie_proj");

    // Per-batch analysis
    int64_t bs=D*SEQ;
    float sum0=0,sum1=0;
    for(int i=0;i<bs;i++){sum0+=r[i];sum1+=r[bs+i];}
    printf("  batch0 mean=%.6f, batch1 mean=%.6f\n",sum0/bs,sum1/bs);

    if (md > 0.001) {
        printf("*** NON-CONTIGUOUS PERMUTE CAUSES DIVERGENCE ***\n");
        // Find first diverging element
        for(int i=0;i<(int)r.size()&&i<20;i++){
            double d=std::abs((double)r[i]-(double)gie.data[i]);
            if(d>0.001)
                printf("  [%d]: got=%.6f gold=%.6f diff=%.6f\n",i,r[i],gie.data[i],d);
        }
    }

    CHECK(md<0.001);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// Helper: load gold from a specific directory
static bool load_gold_dir(const char* dir, const char* name, gold_t& out) {
    std::string path = std::string(dir)+"/"+name+".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t nd; f.read((char*)&nd,4);
    out.shape.resize(nd);
    for (uint32_t i=0;i<nd;i++){int64_t d;f.read((char*)&d,8);out.shape[i]=d;}
    int64_t n=1; for(auto d:out.shape) n*=d;
    out.data.resize(n);
    f.read((char*)out.data.data(),n*4);
    return f.good();
}

// Test: full ODE step 1 with production layout — permuted x, ggml_pad, split, CFG, Euler.
// SEQ is read from gold x.shape[0], so the same test code works for SEQ=20 and SEQ=184.
// Set COSYVOICE_TEST_GOLD_DIR to use a different gold directory (default: /tmp/cv_cond_dit_gold).
SCENARIO("Full ODE step 1 with production tensor layout matches Python gold",
         "[e2e][full_step]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }
    const char* gold_dir = std::getenv("COSYVOICE_TEST_GOLD_DIR");
    if (!gold_dir) gold_dir = GOLD;

    gold_t gx,gc_,gm,gs,gte,gvc,gvu,gvcfg,gxnew;
    auto lg = [&](const char* n, gold_t& o) { return load_gold_dir(gold_dir, n, o); };
    bool ok = lg("x",gx)&&lg("cond",gc_)&&lg("mu",gm)&&
              lg("spks",gs)&&lg("time_emb",gte)&&
              lg("v_cond",gvc)&&lg("v_uncond",gvu)&&
              lg("v_cfg",gvcfg)&&lg("x_after_step1",gxnew);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    // Read SEQ from gold x shape (shape[0] = SEQ for [SEQ, MEL])
    const int tSEQ = static_cast<int>(gx.shape[0]);
    printf("  [full_step] SEQ=%d (from gold, dir=%s)\n", tSEQ, gold_dir);

    gguf_file mdl(mp);
    REQUIRE(bool(mdl));

    const size_t csz=ggml_tensor_overhead()*2500+1024LL*1024*1024;
    auto*ctx=ggml_init({.mem_size=csz,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx);

    auto*x_2d=mk(ctx,tSEQ,MEL);
    auto*enc_cond=mk(ctx,MEL,tSEQ); auto*enc_mu=mk(ctx,MEL,tSEQ); auto*enc_spks=mk(ctx,MEL,1);
    ggml_set_output(enc_cond); ggml_set_output(enc_mu); ggml_set_output(enc_spks);
    auto*pw=mk(ctx,320,D),*pb=mk(ctx,D);
    auto*c1w=mk(ctx,CPE_KERNEL,CPE_CH_PER_GROUP,D),*c1b=mk(ctx,D);
    auto*c2w=mk(ctx,CPE_KERNEL,CPE_CH_PER_GROUP,D),*c2b=mk(ctx,D);
    auto*te_t=mk(ctx,D);
    BW bw[DEPTH];
    for(int i=0;i<DEPTH;i++){
        char pfx[128]; snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto tn=[&](const char*s)->ggml_tensor*{char n[256];snprintf(n,256,"%s.%s",pfx,s);return mdl.t(n);};
        auto mkw=[&](ggml_tensor*src)->ggml_tensor*{
            auto*t=ggml_new_tensor(ctx,GGML_TYPE_F32,GGML_MAX_DIMS,src->ne);ggml_set_input(t);return t;};
        bw[i]={mkw(tn("attn_norm.linear.weight")),mkw(tn("attn_norm.linear.bias")),
               mkw(tn("attn.to_q.weight")),mkw(tn("attn.to_q.bias")),
               mkw(tn("attn.to_k.weight")),mkw(tn("attn.to_k.bias")),
               mkw(tn("attn.to_v.weight")),mkw(tn("attn.to_v.bias")),
               mkw(tn("attn.to_out.0.weight")),mkw(tn("attn.to_out.0.bias")),
               mkw(tn("ff.ff.0.0.weight")),mkw(tn("ff.ff.0.0.bias")),
               mkw(tn("ff.ff.2.weight")),mkw(tn("ff.ff.2.bias"))};
    }
    auto*now=mk(ctx,D,D*2),*nob=mk(ctx,D*2);
    auto*pjow=mk(ctx,D,MEL),*pjob=mk(ctx,MEL);

    // --- Graph: production layout ---
    auto*cond_b2=ggml_pad(ctx,enc_cond,0,0,1,0);
    auto*mu_b2=ggml_pad(ctx,enc_mu,0,0,1,0);
    auto*spks_b2=ggml_pad(ctx,enc_spks,0,0,1,0);
    auto*x_b2=ggml_repeat_4d(ctx,x_2d,tSEQ,MEL,2,1);
    auto*x_perm=ggml_permute(ctx,x_b2,1,0,2,3);

    auto*spks_rep=ggml_repeat(ctx,spks_b2,x_perm);
    auto*ccat=ggml_concat(ctx,x_perm,cond_b2,0);
    ccat=ggml_concat(ctx,ccat,mu_b2,0);
    ccat=ggml_concat(ctx,ccat,spks_rep,0);
    auto*proj=linop(ctx,ccat,pw,pb);
    auto*cvp=ggml_cont(ctx,ggml_permute(ctx,proj,1,0,2,3));
    cvp=ggml_pad_ext(ctx,cvp,CPE_KERNEL-1,0,0,0,0,0,0,0);
    cvp=grouped_conv1d(ctx,cvp,c1w,c1b,CPE_GROUPS);
    cvp=ggml_mish(ctx,cvp);
    cvp=ggml_pad_ext(ctx,cvp,CPE_KERNEL-1,0,0,0,0,0,0,0);
    cvp=grouped_conv1d(ctx,cvp,c2w,c2b,CPE_GROUPS);
    cvp=ggml_mish(ctx,cvp);
    cvp=ggml_cont(ctx,ggml_permute(ctx,cvp,1,0,2,3));
    auto*h=ggml_add(ctx,proj,cvp);

    // Use flash attention if COSYVOICE_TEST_FLASH_ATTN is set
    bool use_fattn = std::getenv("COSYVOICE_TEST_FLASH_ATTN") != nullptr;
    if (use_fattn) printf("  [full_step] flash_attn=ON\n");

    auto*pos=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,tSEQ,BATCH); ggml_set_input(pos);
    for(int i=0;i<DEPTH;i++) h=dit_block(ctx,h,te_t,pos,bw[i],tSEQ,BATCH,use_fattn);

    auto*es=linop(ctx,ggml_silu(ctx,te_t),now,nob);
    auto*sf=ggml_scale_bias(ctx,ggml_cont(ctx,ggml_view_1d(ctx,es,D,0)),1.f,1.f);
    auto*hf=ggml_cont(ctx,ggml_view_1d(ctx,es,D,D*4));
    h=ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,h,1e-6f),unsq1(ctx,sf)),unsq1(ctx,hf));
    auto*dout=linop(ctx,h,pjow,pjob);
    ggml_set_output(dout);
    auto*vc=ggml_cont(ctx,ggml_view_3d(ctx,dout,dout->ne[0],dout->ne[1],1,dout->nb[1],dout->nb[2],0));
    auto*vu=ggml_cont(ctx,ggml_view_3d(ctx,dout,dout->ne[0],dout->ne[1],1,dout->nb[1],dout->nb[2],dout->nb[2]));
    ggml_set_output(vc); ggml_set_output(vu);
    auto*vcfg=ggml_sub(ctx,ggml_scale(ctx,vc,1.7f),ggml_scale(ctx,vu,0.7f));
    ggml_set_output(vcfg);

    // --- Fill ---
    auto*backend=ggml_backend_cpu_init(); REQUIRE(backend);
    auto*bbuf=ggml_backend_alloc_ctx_tensors(ctx,backend); REQUIRE(bbuf);

    { std::vector<float> xt(tSEQ*MEL);
      for(int s=0;s<tSEQ;s++) for(int d=0;d<MEL;d++) xt[s+d*tSEQ]=gx.data[s*MEL+d];
      ggml_backend_tensor_set(x_2d,xt.data(),0,xt.size()*4); }
    ggml_backend_tensor_set(enc_cond,gc_.data.data(),0,MEL*tSEQ*4);
    ggml_backend_tensor_set(enc_mu,gm.data.data(),0,MEL*tSEQ*4);
    ggml_backend_tensor_set(enc_spks,gs.data.data(),0,MEL*4);
    ggml_backend_tensor_set(te_t,gte.data.data(),0,D*4);

    auto fl=[&](const char*n,ggml_tensor*d){fill_gguf(d,mdl.t(n));};
    fl("decoder.estimator.input_embed.proj.weight",pw);
    ggml_backend_tensor_set(pb,mdl.t("decoder.estimator.input_embed.proj.bias")->data,0,D*4);
    fl("decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight",c1w);
    ggml_backend_tensor_set(c1b,mdl.t("decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias")->data,0,D*4);
    fl("decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight",c2w);
    ggml_backend_tensor_set(c2b,mdl.t("decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias")->data,0,D*4);
    fl("decoder.estimator.norm_out.linear.weight",now);
    ggml_backend_tensor_set(nob,mdl.t("decoder.estimator.norm_out.linear.bias")->data,0,D*2*4);
    fl("decoder.estimator.proj_out.weight",pjow);
    ggml_backend_tensor_set(pjob,mdl.t("decoder.estimator.proj_out.bias")->data,0,MEL*4);
    for(int i=0;i<DEPTH;i++){
        char pfx[128]; snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto fb=[&](const char*s,ggml_tensor*d){char n[256];snprintf(n,256,"%s.%s",pfx,s);fill_gguf(d,mdl.t(n));};
        fb("attn_norm.linear.weight",bw[i].aw); fb("attn_norm.linear.bias",bw[i].ab);
        fb("attn.to_q.weight",bw[i].qw); fb("attn.to_q.bias",bw[i].qb);
        fb("attn.to_k.weight",bw[i].kw); fb("attn.to_k.bias",bw[i].kb);
        fb("attn.to_v.weight",bw[i].vw); fb("attn.to_v.bias",bw[i].vb);
        fb("attn.to_out.0.weight",bw[i].ow); fb("attn.to_out.0.bias",bw[i].ob);
        fb("ff.ff.0.0.weight",bw[i].fw); fb("ff.ff.0.0.bias",bw[i].fb);
        fb("ff.ff.2.weight",bw[i].dw); fb("ff.ff.2.bias",bw[i].db);
    }
    std::vector<int32_t> pids(tSEQ*BATCH);
    for(int i=0;i<tSEQ*BATCH;i++) pids[i]=i%tSEQ;
    ggml_backend_tensor_set(pos,pids.data(),0,pids.size()*4);

    auto*graph=ggml_new_graph_custom(ctx,8192,false);
    ggml_build_forward_expand(graph,vcfg);
    REQUIRE(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);

    auto rd=[&](ggml_tensor*t)->std::vector<float>{
        std::vector<float>r(ggml_nelements(t));ggml_backend_tensor_get(t,r.data(),0,r.size()*4);return r;};
    auto rv=rd(vc); double md_vc=compare(rv.data(),gvc.data.data(),rv.size(),"v_cond");
    rv=rd(vu); double md_vu=compare(rv.data(),gvu.data.data(),rv.size(),"v_uncond");
    rv=rd(vcfg); double md_cfg=compare(rv.data(),gvcfg.data.data(),rv.size(),"v_cfg");

    {auto r2=rd(vc); float s=0; for(auto v:r2) s+=v;
     printf("  C++ v_cond mean=%.4f (gold=%.4f)\n",s/r2.size(),
       [&]{float s2=0;for(auto v:gvc.data)s2+=v;return s2/gvc.n();}());}
    {auto r2=rd(vu); float s=0; for(auto v:r2) s+=v;
     printf("  C++ v_uncond mean=%.4f (gold=%.4f)\n",s/r2.size(),
       [&]{float s2=0;for(auto v:gvu.data)s2+=v;return s2/gvu.n();}());}

    CHECK(md_vc<2.0);
    CHECK(md_vu<2.0);
    CHECK(md_cfg<5.0);

    ggml_backend_buffer_free(bbuf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// Test: same as [full_step] but uses model tensors directly (no fill_gguf copy).
// This reproduces the pipeline's weight reference pattern.
SCENARIO("Full ODE step 1 with direct model tensor reference",
         "[e2e][full_step_direct]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }
    const char* gold_dir = std::getenv("COSYVOICE_TEST_GOLD_DIR");
    if (!gold_dir) gold_dir = GOLD;

    gold_t gx,gc_,gm,gs,gte,gvc,gvu,gvcfg,gxnew;
    auto lg = [&](const char* n, gold_t& o) { return load_gold_dir(gold_dir, n, o); };
    bool ok = lg("x",gx)&&lg("cond",gc_)&&lg("mu",gm)&&
              lg("spks",gs)&&lg("time_emb",gte)&&
              lg("v_cond",gvc)&&lg("v_uncond",gvu)&&
              lg("v_cfg",gvcfg)&&lg("x_after_step1",gxnew);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    const int tSEQ = static_cast<int>(gx.shape[0]);
    printf("  [full_step_direct] SEQ=%d (from gold, dir=%s)\n", tSEQ, gold_dir);

    // Load model — weights stay as-is in model context (Q8_0 or F32)
    gguf_file mdl(mp);
    REQUIRE(bool(mdl));

    const size_t csz=ggml_tensor_overhead()*2500+1024LL*1024*1024;
    auto*ctx=ggml_init({.mem_size=csz,.mem_buffer=nullptr,.no_alloc=true});
    REQUIRE(ctx);

    // Input tensors (these are F32, allocated fresh)
    auto*x_2d=mk(ctx,tSEQ,MEL);
    auto*enc_cond=mk(ctx,MEL,tSEQ); auto*enc_mu=mk(ctx,MEL,tSEQ); auto*enc_spks=mk(ctx,MEL,1);
    ggml_set_output(enc_cond); ggml_set_output(enc_mu); ggml_set_output(enc_spks);
    auto*te_t=mk(ctx,D);
    auto*pb=mk(ctx,D); // proj bias
    auto*c1b=mk(ctx,D); auto*c2b=mk(ctx,D); // conv biases
    auto*nob=mk(ctx,D*2); auto*pjob=mk(ctx,MEL); // norm/proj biases

    // Allocate weight tensors with ORIGINAL type (Q8_0 or F32) — no dequantize.
    // This mirrors the pipeline's weight loading pattern.
    auto mkw_native=[&](const char* name)->ggml_tensor*{
        auto*src=mdl.t(name);
        auto*t=ggml_new_tensor(ctx,src->type,GGML_MAX_DIMS,src->ne);
        ggml_set_input(t);
        return t;
    };
    auto*pw = mkw_native("decoder.estimator.input_embed.proj.weight");
    auto*c1w = mkw_native("decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight");
    auto*c2w = mkw_native("decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight");
    auto*now = mkw_native("decoder.estimator.norm_out.linear.weight");
    auto*pjow = mkw_native("decoder.estimator.proj_out.weight");

    BW bw[DEPTH];
    for(int i=0;i<DEPTH;i++){
        char pfx[128]; snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto tn=[&](const char*s)->ggml_tensor*{
            char n[256];snprintf(n,256,"%s.%s",pfx,s);return mkw_native(n);};
        auto mkb=[&](const char*s)->ggml_tensor*{
            char n[256];snprintf(n,256,"%s.%s",pfx,s);
            auto*src=mdl.t(n); auto*t=mk(ctx,src->ne[0]); return t;};
        bw[i]={tn("attn_norm.linear.weight"),mkb("attn_norm.linear.bias"),
               tn("attn.to_q.weight"),mkb("attn.to_q.bias"),
               tn("attn.to_k.weight"),mkb("attn.to_k.bias"),
               tn("attn.to_v.weight"),mkb("attn.to_v.bias"),
               tn("attn.to_out.0.weight"),mkb("attn.to_out.0.bias"),
               tn("ff.ff.0.0.weight"),mkb("ff.ff.0.0.bias"),
               tn("ff.ff.2.weight"),mkb("ff.ff.2.bias")};
    }

    // --- Graph: same as [full_step] ---
    auto*cond_b2=ggml_pad(ctx,enc_cond,0,0,1,0);
    auto*mu_b2=ggml_pad(ctx,enc_mu,0,0,1,0);
    auto*spks_b2=ggml_pad(ctx,enc_spks,0,0,1,0);
    auto*x_b2=ggml_repeat_4d(ctx,x_2d,tSEQ,MEL,2,1);
    auto*x_perm=ggml_permute(ctx,x_b2,1,0,2,3);

    auto*spks_rep=ggml_repeat(ctx,spks_b2,x_perm);
    auto*ccat=ggml_concat(ctx,x_perm,cond_b2,0);
    ccat=ggml_concat(ctx,ccat,mu_b2,0);
    ccat=ggml_concat(ctx,ccat,spks_rep,0);
    auto*proj=linop(ctx,ccat,pw,pb);
    auto*cvp=ggml_cont(ctx,ggml_permute(ctx,proj,1,0,2,3));
    cvp=ggml_pad_ext(ctx,cvp,CPE_KERNEL-1,0,0,0,0,0,0,0);
    cvp=grouped_conv1d(ctx,cvp,c1w,c1b,CPE_GROUPS);
    cvp=ggml_mish(ctx,cvp);
    cvp=ggml_pad_ext(ctx,cvp,CPE_KERNEL-1,0,0,0,0,0,0,0);
    cvp=grouped_conv1d(ctx,cvp,c2w,c2b,CPE_GROUPS);
    cvp=ggml_mish(ctx,cvp);
    cvp=ggml_cont(ctx,ggml_permute(ctx,cvp,1,0,2,3));
    auto*h=ggml_add(ctx,proj,cvp);

    bool use_fattn = std::getenv("COSYVOICE_TEST_FLASH_ATTN") != nullptr;
    auto*pos=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,tSEQ,BATCH); ggml_set_input(pos);
    for(int i=0;i<DEPTH;i++) h=dit_block(ctx,h,te_t,pos,bw[i],tSEQ,BATCH,use_fattn);

    auto*es=linop(ctx,ggml_silu(ctx,te_t),now,nob);
    auto*sf=ggml_scale_bias(ctx,ggml_cont(ctx,ggml_view_1d(ctx,es,D,0)),1.f,1.f);
    auto*hf=ggml_cont(ctx,ggml_view_1d(ctx,es,D,D*4));
    h=ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,h,1e-6f),unsq1(ctx,sf)),unsq1(ctx,hf));
    auto*dout=linop(ctx,h,pjow,pjob);
    ggml_set_output(dout);
    auto*vc=ggml_cont(ctx,ggml_view_3d(ctx,dout,dout->ne[0],dout->ne[1],1,dout->nb[1],dout->nb[2],0));
    auto*vu=ggml_cont(ctx,ggml_view_3d(ctx,dout,dout->ne[0],dout->ne[1],1,dout->nb[1],dout->nb[2],dout->nb[2]));
    ggml_set_output(vc); ggml_set_output(vu);
    auto*vcfg=ggml_sub(ctx,ggml_scale(ctx,vc,1.7f),ggml_scale(ctx,vu,0.7f));
    ggml_set_output(vcfg);

    // --- Fill ---
    auto*backend=ggml_backend_cpu_init(); REQUIRE(backend);
    auto*bbuf=ggml_backend_alloc_ctx_tensors(ctx,backend); REQUIRE(bbuf);

    { std::vector<float> xt(tSEQ*MEL);
      for(int s=0;s<tSEQ;s++) for(int d=0;d<MEL;d++) xt[s+d*tSEQ]=gx.data[s*MEL+d];
      ggml_backend_tensor_set(x_2d,xt.data(),0,xt.size()*4); }
    ggml_backend_tensor_set(enc_cond,gc_.data.data(),0,MEL*tSEQ*4);
    ggml_backend_tensor_set(enc_mu,gm.data.data(),0,MEL*tSEQ*4);
    ggml_backend_tensor_set(enc_spks,gs.data.data(),0,MEL*4);
    ggml_backend_tensor_set(te_t,gte.data.data(),0,D*4);

    // Fill weights with NATIVE data (raw bytes, no dequantize)
    auto fill_native=[&](const char*n,ggml_tensor*d){
        auto*src=mdl.t(n);
        ggml_backend_tensor_set(d,src->data,0,ggml_nbytes(src));
    };
    fill_native("decoder.estimator.input_embed.proj.weight",pw);
    ggml_backend_tensor_set(pb,mdl.t("decoder.estimator.input_embed.proj.bias")->data,0,D*4);
    fill_native("decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight",c1w);
    ggml_backend_tensor_set(c1b,mdl.t("decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias")->data,0,D*4);
    fill_native("decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight",c2w);
    ggml_backend_tensor_set(c2b,mdl.t("decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias")->data,0,D*4);
    fill_native("decoder.estimator.norm_out.linear.weight",now);
    ggml_backend_tensor_set(nob,mdl.t("decoder.estimator.norm_out.linear.bias")->data,0,D*2*4);
    fill_native("decoder.estimator.proj_out.weight",pjow);
    ggml_backend_tensor_set(pjob,mdl.t("decoder.estimator.proj_out.bias")->data,0,MEL*4);
    for(int i=0;i<DEPTH;i++){
        char pfx[128]; snprintf(pfx,sizeof(pfx),"decoder.estimator.transformer_blocks.%d",i);
        auto fw=[&](const char*s,ggml_tensor*d){char n[256];snprintf(n,256,"%s.%s",pfx,s);fill_native(n,d);};
        auto fb=[&](const char*s,ggml_tensor*d){char n[256];snprintf(n,256,"%s.%s",pfx,s);
            ggml_backend_tensor_set(d,mdl.t(n)->data,0,ggml_nbytes(d));};
        fw("attn_norm.linear.weight",bw[i].aw); fb("attn_norm.linear.bias",bw[i].ab);
        fw("attn.to_q.weight",bw[i].qw); fb("attn.to_q.bias",bw[i].qb);
        fw("attn.to_k.weight",bw[i].kw); fb("attn.to_k.bias",bw[i].kb);
        fw("attn.to_v.weight",bw[i].vw); fb("attn.to_v.bias",bw[i].vb);
        fw("attn.to_out.0.weight",bw[i].ow); fb("attn.to_out.0.bias",bw[i].ob);
        fw("ff.ff.0.0.weight",bw[i].fw); fb("ff.ff.0.0.bias",bw[i].fb);
        fw("ff.ff.2.weight",bw[i].dw); fb("ff.ff.2.bias",bw[i].db);
    }
    std::vector<int32_t> pids(tSEQ*BATCH);
    for(int i=0;i<tSEQ*BATCH;i++) pids[i]=i%tSEQ;
    ggml_backend_tensor_set(pos,pids.data(),0,pids.size()*4);

    auto*graph=ggml_new_graph_custom(ctx,8192,false);
    ggml_build_forward_expand(graph,vcfg);
    REQUIRE(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);

    auto rd=[&](ggml_tensor*t)->std::vector<float>{
        std::vector<float>r(ggml_nelements(t));ggml_backend_tensor_get(t,r.data(),0,r.size()*4);return r;};
    auto rv=rd(vc); double md_vc=compare(rv.data(),gvc.data.data(),rv.size(),"v_cond");
    rv=rd(vu); double md_vu=compare(rv.data(),gvu.data.data(),rv.size(),"v_uncond");

    {auto r2=rd(vc); float s=0; for(auto v:r2) s+=v;
     printf("  C++ v_cond mean=%.4f (gold=%.4f)\n",s/r2.size(),
       [&]{float s2=0;for(auto v:gvc.data)s2+=v;return s2/gvc.n();}());}
    {auto r2=rd(vu); float s=0; for(auto v:r2) s+=v;
     printf("  C++ v_uncond mean=%.4f (gold=%.4f)\n",s/r2.size(),
       [&]{float s2=0;for(auto v:gvu.data)s2+=v;return s2/gvu.n();}());}

    CHECK(md_vc<2.0);
    CHECK(md_vu<2.0);

    ggml_backend_buffer_free(bbuf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

// Pipeline-style split_tensor: ggml_view_4d with dim-based chunking
// Exact copy of cosyvoice-graph.cpp:7-29
static void pipe_split(ggml_context* ctx, ggml_tensor* tensor, int dim, ggml_tensor** out, uint16_t chunks) {
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t off = tensor->nb[dim] * ne[dim];
    for(uint16_t i=0;i!=chunks;++i)
        out[i] = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3],
            tensor->nb[1], tensor->nb[2], tensor->nb[3], i*off);
}

// Pipeline-style unsqueeze: ggml_view_4d dimension insertion
// Exact copy of cosyvoice-graph.cpp:124-133
static ggml_tensor* pipe_unsqueeze(ggml_context* ctx, ggml_tensor* x, int dim) {
    if(dim==0)      return ggml_view_4d(ctx,x,1,x->ne[0],x->ne[1],x->ne[2],x->nb[0],x->nb[1],x->nb[2],0);
    else if(dim==1) return ggml_view_4d(ctx,x,x->ne[0],1,x->ne[1],x->ne[2],x->nb[1],x->nb[1],x->nb[2],0);
    else if(dim==2) return ggml_view_4d(ctx,x,x->ne[0],x->ne[1],1,x->ne[2],x->nb[1],x->nb[2],x->nb[2],0);
    else { GGML_ABORT("unsqueeze: invalid dim"); return nullptr; }
}

// Pipeline-faithful DiT block: mirrors DiTBlock::build_cgraph() exactly,
// using pipe_split (4D views) and pipe_unsqueeze instead of the test's
// ggml_view_1d approach.
static ggml_tensor* pipe_dit_block(ggml_context* c, ggml_tensor* x, ggml_tensor* emb,
    ggml_tensor* pos, const BW& w, int seq, int batch)
{
    // === AdaLayerNormZero::build_cgraph ===
    auto* es = ggml_silu(c, emb);
    auto* ep = linop(c, es, w.aw, w.ab);

    // split_tensor<6> on dim=0: 4D views
    ggml_tensor* chunks[6];
    pipe_split(c, ep, 0, chunks, 6);
    auto* shift_msa = chunks[0];
    auto* scale_msa = chunks[1];
    auto* gate_msa  = chunks[2];
    auto* shift_mlp = chunks[3];
    auto* scale_mlp = chunks[4];
    auto* gate_mlp  = chunks[5];

    // Pipeline: scale_msa gets scale_bias + unsqueeze BEFORE norm
    scale_msa = ggml_scale_bias(c, ggml_cont(c, scale_msa), 1.f, 1.f);
    scale_msa = pipe_unsqueeze(c, scale_msa, 1);
    shift_msa = ggml_cont(c, shift_msa);
    shift_msa = pipe_unsqueeze(c, shift_msa, 1);

    // norm (no learned weight/bias)
    auto* xn = ggml_norm(c, x, 1e-6f);
    xn = ggml_mul(c, xn, scale_msa);
    xn = ggml_add(c, xn, shift_msa);

    // === Attention::build_cgraph (cut_len=0) ===
    const int64_t full_seq_len = pos->ne[0];
    const int64_t seq_len = full_seq_len;
    const int64_t batch_size = xn->ne[2];
    const int head_dim = HD;

    auto* key   = linop(c, xn, w.kw, w.kb);
    auto* value = linop(c, xn, w.vw, w.vb);
    auto* query = linop(c, xn, w.qw, w.qb);

    query = ggml_reshape_3d(c, query, head_dim, HEADS, seq_len * batch_size);
    key   = ggml_reshape_3d(c, key,   head_dim, HEADS, full_seq_len * batch_size);
    auto* pos_flat = ggml_reshape_1d(c, pos, seq_len * batch_size);
    auto* full_pos_flat = ggml_reshape_1d(c, pos, full_seq_len * batch_size);

    query = ggml_rope(c, query, pos_flat, head_dim, GGML_ROPE_TYPE_NORMAL);
    key   = ggml_rope(c, key, full_pos_flat, head_dim, GGML_ROPE_TYPE_NORMAL);

    query = ggml_reshape_4d(c, query, head_dim, HEADS, seq_len, batch_size);
    key   = ggml_reshape_4d(c, key,   head_dim, HEADS, full_seq_len, batch_size);
    value = ggml_reshape_4d(c, value, head_dim, HEADS, full_seq_len, batch_size);

    query = ggml_permute(c, query, 0, 2, 1, 3);
    key   = ggml_permute(c, key,   0, 2, 1, 3);
    value = ggml_permute(c, value, 0, 2, 1, 3);

    query = ggml_cont(c, query);
    key   = ggml_cont(c, key);

    // Manual attention (no flash_attn)
    auto* attn_scores = ggml_mul_mat(c, key, query);
    attn_scores = ggml_scale(c, attn_scores, 1.f / sqrtf((float)head_dim));
    auto* attn_weights = ggml_soft_max(c, attn_scores);
    auto* vp = ggml_permute(c, value, 1, 0, 2, 3);
    vp = ggml_cont(c, vp);
    auto* attn_output = ggml_mul_mat(c, vp, attn_weights);
    attn_output = ggml_permute(c, attn_output, 0, 2, 1, 3);
    attn_output = ggml_cont(c, attn_output);

    attn_output = ggml_reshape_3d(c, attn_output,
        attn_output->ne[0] * attn_output->ne[1], seq_len, batch_size);
    attn_output = linop(c, attn_output, w.ow, w.ob);

    // === gate + residual ===
    gate_msa = ggml_cont(c, gate_msa);
    gate_msa = pipe_unsqueeze(c, gate_msa, 1);
    attn_output = ggml_mul(c, attn_output, gate_msa);
    x = ggml_add(c, x, attn_output);

    // === FeedForward ===
    auto* ff_norm = ggml_norm(c, x, 1e-6f);
    scale_mlp = ggml_scale_bias(c, ggml_cont(c, scale_mlp), 1.f, 1.f);
    ff_norm = ggml_mul(c, ff_norm, pipe_unsqueeze(c, scale_mlp, 1));
    shift_mlp = ggml_cont(c, shift_mlp);
    ff_norm = ggml_add(c, ff_norm, pipe_unsqueeze(c, shift_mlp, 1));

    auto* ff = ggml_gelu_erf(c, linop(c, ff_norm, w.fw, w.fb));
    ff = linop(c, ff, w.dw, w.db);

    gate_mlp = ggml_cont(c, gate_mlp);
    ff = ggml_mul(c, ff, pipe_unsqueeze(c, gate_mlp, 1));
    return ggml_add(c, x, ff);
}

// Test: Compare dit_block() (hand-written, known correct) vs pipe_dit_block()
// (pipeline-faithful reimplementation) on block 0 with identical inputs.
// This isolates graph construction differences without needing library linking.
SCENARIO("Pipeline-faithful dit_block vs hand-written dit_block for block 0",
         "[dit_cgraph]") {
    auto* mp = std::getenv("COSYVOICE_TEST_MODEL");
    if (!mp) { SKIP("Set COSYVOICE_TEST_MODEL"); return; }

    gold_t gte, gie_full;
    bool ok = load_gold("time_emb",gte)&&load_gold("ie_full",gie_full);
    if(!ok){SKIP("Run: python3 tests/gen_cond_dit_gold.py");return;}

    gguf_file mdl(mp);
    REQUIRE(bool(mdl));

    const size_t csz = ggml_tensor_overhead()*4000 + 1024LL*1024*400;
    auto* ctx = ggml_init({.mem_size=csz, .mem_buffer=nullptr, .no_alloc=true});
    REQUIRE(ctx);

    // Input tensors — separate copies for each path
    auto* ie1 = mk(ctx, D, SEQ, BATCH);
    auto* ie2 = mk(ctx, D, SEQ, BATCH);
    auto* te = mk(ctx, D);
    auto* pos = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, SEQ, BATCH); ggml_set_input(pos);

    // Two independent weight sets for fair comparison
    auto load_bw = [&](const char* blk) -> BW {
        auto mkw=[&](const char*s)->ggml_tensor*{
            char n[256];snprintf(n,256,"%s.%s",blk,s);
            auto*src=mdl.t(n);REQUIRE(src);
            auto*t=ggml_new_tensor(ctx,src->type,GGML_MAX_DIMS,src->ne);ggml_set_input(t);return t;};
        auto mkb=[&](const char*s)->ggml_tensor*{
            char n[256];snprintf(n,256,"%s.%s",blk,s);
            auto*src=mdl.t(n);REQUIRE(src);auto*t=mk(ctx,src->ne[0]);return t;};
        return {mkw("attn_norm.linear.weight"),mkb("attn_norm.linear.bias"),
                mkw("attn.to_q.weight"),mkb("attn.to_q.bias"),
                mkw("attn.to_k.weight"),mkb("attn.to_k.bias"),
                mkw("attn.to_v.weight"),mkb("attn.to_v.bias"),
                mkw("attn.to_out.0.weight"),mkb("attn.to_out.0.bias"),
                mkw("ff.ff.0.0.weight"),mkb("ff.ff.0.0.bias"),
                mkw("ff.ff.2.weight"),mkb("ff.ff.2.bias")};
    };
    auto bw_hw = load_bw("decoder.estimator.transformer_blocks.0");
    auto bw_pp = load_bw("decoder.estimator.transformer_blocks.0");

    // Build both graphs
    auto* out_hw = dit_block(ctx, ie1, te, pos, bw_hw, SEQ, BATCH);
    ggml_set_output(out_hw);
    auto* out_pp = pipe_dit_block(ctx, ie2, te, pos, bw_pp, SEQ, BATCH);
    ggml_set_output(out_pp);

    // Allocate & fill
    auto* backend = ggml_backend_cpu_init(); REQUIRE(backend);
    auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend); REQUIRE(buf);

    // Fill ie_full from gold
    {
        int64_t bf = D * SEQ;
        std::vector<float> ifd(bf*2, 0.f);
        if(gie_full.n() >= bf) memcpy(ifd.data(), gie_full.data.data(), bf*4);
        ggml_backend_tensor_set(ie1, ifd.data(), 0, ifd.size()*4);
        ggml_backend_tensor_set(ie2, ifd.data(), 0, ifd.size()*4);
    }
    ggml_backend_tensor_set(te, gte.data.data(), 0, D*4);
    std::vector<int32_t> pids(SEQ*BATCH);
    for(int i=0;i<SEQ*BATCH;i++) pids[i]=i%SEQ;
    ggml_backend_tensor_set(pos, pids.data(), 0, pids.size()*4);

    // Fill BOTH weight sets with identical data
    auto fill_bw = [&](const char* blk, BW& bw) {
        auto fw=[&](const char*s,ggml_tensor*d){
            char n[256];snprintf(n,256,"%s.%s",blk,s);
            ggml_backend_tensor_set(d,mdl.t(n)->data,0,ggml_nbytes(d));};
        fw("attn_norm.linear.weight",bw.aw); fw("attn_norm.linear.bias",bw.ab);
        fw("attn.to_q.weight",bw.qw); fw("attn.to_q.bias",bw.qb);
        fw("attn.to_k.weight",bw.kw); fw("attn.to_k.bias",bw.kb);
        fw("attn.to_v.weight",bw.vw); fw("attn.to_v.bias",bw.vb);
        fw("attn.to_out.0.weight",bw.ow); fw("attn.to_out.0.bias",bw.ob);
        fw("ff.ff.0.0.weight",bw.fw); fw("ff.ff.0.0.bias",bw.fb);
        fw("ff.ff.2.weight",bw.dw); fw("ff.ff.2.bias",bw.db);
    };
    fill_bw("decoder.estimator.transformer_blocks.0", bw_hw);
    fill_bw("decoder.estimator.transformer_blocks.0", bw_pp);

    // Compute
    auto* graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, out_hw);
    ggml_build_forward_expand(graph, out_pp);
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    // Compare
    auto rd=[&](ggml_tensor*t)->std::vector<float>{
        std::vector<float>r(ggml_nelements(t));ggml_backend_tensor_get(t,r.data(),0,r.size()*4);return r;};

    printf("  [dit_cgraph] block 0: handwritten vs pipeline-faithful\n");

    auto rh = rd(out_hw);
    auto rp = rd(out_pp);
    double md = compare(rh.data(), rp.data(), rh.size(), "block0_final");

    {float sh=0,sp=0;for(auto v:rh)sh+=v;for(auto v:rp)sp+=v;
     printf("  handwritten mean=%.6f  pipeline mean=%.6f\n",sh/rh.size(),sp/rp.size());}

    // If this fails, the difference is in how split_tensor/unsqueeze constructs the graph
    CHECK(md < 0.001);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}
