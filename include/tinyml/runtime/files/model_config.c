#include "model_config.h"
#include <stdint.h>

// set by code-gen
#define ARENA_BYTES 1024

static uint8_t arena_buf[ARENA_BYTES];
static Arena arena;
static ModelCtx ctx;

// validates the model to ensure we can make safe forward passes
// also sets the max_acc and max_act so we can build the correct sized arena
// @return true for a valid model structure
bool validate_model(const Model* m, size_t* act_cap, size_t* acc_cap){
    if(!m || !act_cap || !acc_cap || !m->layers){ return false; }

    size_t max_act = 0;
    size_t max_acc = 0;

    int16_t curr_dim = m->in_dim;
    if(curr_dim == 0) { return false;}

    for(uint8_t i = 0; i < m->num_layers; i++){
        const LayerRef* l = &m->layers[i];
        if(l->type == L_DENSE){
            const DenseLayer* d = (const DenseLayer*)l->layer;
            if(!d) { return false; }

            // validate dense layer consistency
            if(d->in_dim != curr_dim) {return false; }
            curr_dim = d->out_dim;

            if((size_t)d->out_dim > max_act) { max_act = d->out_dim; }
            if((size_t)d->out_dim > max_acc) { max_acc = d->out_dim; }
        } else if(l->type == L_RELU){
            const ReLuLayer* r = (const ReLuLayer*)l->layer;
            if(!r) { return false; }

            if(r->dim != curr_dim) { return false; }

            if((size_t)r->dim > max_act) max_act = r->dim;
        }
        else{
            return false;
        }
    }

    if(curr_dim != m->out_dim) { return false; }

    *act_cap = max_act;
    *acc_cap = max_acc;

    return true;
}

// make sure that the arena had been initialized
// @return false if initialization fails, true otherwise

bool model_init_ctx(const Model* m, ModelCtx* ctx, Arena* arena){
    if(!m || !ctx || !arena) { return false; } // safety check

    size_t act_cap = 0; size_t acc_cap = 0;
    if(!validate_model(m, &act_cap, &acc_cap)) { return false; }

    ctx->arena = arena;
    ctx->act_cap = act_cap;
    ctx->acc_cap = acc_cap;

    // set up activation buffers
    ctx->act0 = (int8_t*)arena_alloc(arena, ctx->act_cap * sizeof(int8_t), 16);
    ctx->act1 = (int8_t*)arena_alloc(arena, ctx->act_cap * sizeof(int8_t), 16);

    // set up accumulation buffer
    ctx->acc = (int32_t*)arena_alloc(arena, ctx->acc_cap * sizeof(int32_t), 16);

    // // safety check
    if(!ctx->act0 || !ctx->act1 || !ctx->acc) {
        return false;
    }

    return true;
}

int8_t clamp_int8(int32_t x) {
    if(x > 127) { x = 127; }
    if(x < -127) { x = -127; }
    return (int8_t)x;
}

static inline int32_t shift_int64(int64_t x, int32_t shift) {
    if(shift <= 0){
        int32_t neg_shift = -shift;
        if(neg_shift > 32) { return (x >= 0) ? INT32_MAX : INT32_MIN; }

        int64_t y = x << neg_shift;

        if(y > INT32_MAX) { return INT32_MAX; }
        if(y < INT32_MIN) { return INT32_MIN; }
        return (int32_t)y;
    }

    if(shift >= 63) return (x >= 0) ? 0 : -1;

    int64_t bias = (int64_t)1 << (shift - 1);
    if(x >= 0) {x += bias; }
    if(x < 0) { x -= bias; }

    return (int32_t)(x >> shift);
}

void model_forward(const Model* m, ModelCtx* ctx, const int8_t* input, int8_t* output){
    if(!m || !ctx || !input || !output){ return; }

    const int8_t* cur = input; // current read pointer
    int8_t* nxt = ctx->act0; // current write pointer

    int16_t cur_dim = m->in_dim;

    size_t n = (size_t)m->num_layers;
    for(size_t i = 0; i < n; i++){
        const LayerRef* l = &m->layers[i];
        const int last = (i == n-1);

        if(last) { nxt = output; }

        if(l->type == L_DENSE){
            const DenseLayer* d = (const DenseLayer*)l->layer;
            if(!d) { return; }

            dense_forward(cur, nxt, d);
        }else if(l->type == L_RELU){
            const ReLuLayer* r = (const ReLuLayer*)l->layer;
            if(!r) { return; }

            relu_forward(cur, nxt, r);
        }

        cur = nxt;
        if(!last) {
            nxt = (nxt == ctx->act0) ? ctx->act1 : ctx->act0;
        }
    }
}

// quantized dense forward pass
void dense_forward(const int8_t* input, int8_t* output, const DenseLayer* dense){
    if(!input || !output || !dense){ return; }

    // populate local variables for effecieny
    const int8_t* weights = dense->weights;
    const int32_t* bias_acc = dense->bias_acc;
    int32_t in_zp = dense->in_zp;
    int32_t out_zp = dense->out_zp;
    const int32_t* mult = dense->mult;
    const int8_t* shift = dense->shift;
    const int32_t* w_row_sum = dense->w_row_sum;

    size_t n_out = dense->out_dim;
    size_t n_in = dense->in_dim;
    for(size_t o = 0; o < n_out; o++){
        int32_t acc = bias_acc[o];
        const int8_t* w_row = &weights[o * n_in];

        int32_t dot_prod = 0;
        for(size_t i = 0; i < n_in; i++){
            dot_prod += (int32_t)input[i] * (int32_t)w_row[i];
        }

        dot_prod -= in_zp * w_row_sum[o];
        acc += dot_prod;

        int64_t prod = (int64_t)acc * (int64_t)mult[o];
        int32_t scaled = shift_int64(prod, shift[o]);

        int32_t q = scaled + out_zp;
        output[o] = clamp_int8(q);
    }
}

// quantized relu forward pass
void relu_forward(const int8_t* input, int8_t* output, const ReLuLayer* relu){
    int32_t z = relu->zp;
    size_t n = relu->dim;
    for(size_t i = 0; i < n; i++){
        int32_t v = input[i];
        if(v < z) v = z;
        output[i] = clamp_int8(v);
    }
}