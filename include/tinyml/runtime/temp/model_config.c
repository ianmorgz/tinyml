#include "model_config.h"
#include <stdbool.h>
#include <stdio.h>
#include <math.h>


int model_init_ctx(const ModelRef* m, ModelCtx* ctx, Arena* arena){
    if(!m || !ctx || !arena) { return -1; } // safety check

    size_t act_cap = 8; size_t acc_cap = 8; // TODO pull from generated code instead of hardcoding these
//    if(!validate_model(m, &act_cap, &acc_cap)) { return -1; }

    ctx->arena = arena;
    ctx->act_cap = act_cap;
    ctx->acc_cap = acc_cap;

    // set up activation buffers
    ctx->act0 = (int8_t*)arena_alloc(arena, ctx->act_cap * sizeof(int8_t), 16); // the first activation buffer
    ctx->act1 = (int8_t*)arena_alloc(arena, ctx->act_cap * sizeof(int8_t), 16); // the second activation buffer

    // set up accumulation buffer
    ctx->acc = (int32_t*)arena_alloc(arena, ctx->acc_cap * sizeof(int32_t), 16);

    // // safety check
    if(!ctx->act0 || !ctx->act1 || !ctx->acc) {
        return -1;
    }

    return 0;
}


int model_forward(const ModelRef* m, const ModelCtx* ctx, const float* in, float* out){
	int8_t* act_in = ctx->act0;
	int8_t* act_out = ctx->act1;

	quantize(in, act_in, m->in_dim, m->in_scale, m->in_zp);

	size_t n_layers = m->num_layers;

	for(size_t i = 0; i < n_layers; i++){
		const LayerRef* l_ref = &m->layers[i];
		if(l_ref->type == L_DENSE) {
			const DenseLayer* d = (const DenseLayer*)l_ref->layer;
			if(dense_forward(act_in, act_out, d, ctx) < 0) { return -1; }
		} else if(l_ref->type == L_RELU) {
			const ReluLayer* r = (const ReluLayer*)l_ref->layer;
			if(relu_forward(act_in, act_out, r) < 0) { return -1; }
		} else{ return -1; }

		int8_t* tmp = act_in;
		act_in = act_out;
		act_out = tmp;
	}

	dequantize(act_in, out, m->out_dim, m->out_scale, m->out_zp);
	return 1;
}

int dense_forward(const int8_t* input, int8_t* output, const DenseLayer* dense, const ModelCtx* ctx){
	if(!input || !output || !dense){ return -1; }

	// populate local variables for efficiency
	const int8_t* weights = dense->weights;
	const int32_t* biases = dense->biases;
	const float* multipliers = dense->multipliers;
	int32_t* acc = ctx->acc;

	int32_t in_zp = dense->in_zp;
	int32_t out_zp = dense->out_zp;

	size_t n_out = dense->out_dim;
	size_t n_in = dense->in_dim;

	for(size_t o = 0; o < n_out; o++){
		acc[o] = biases[o];

		for(size_t i = 0; i < n_in; i++){
			size_t idx = o * n_in + i;
			acc[o] += weights[idx] * (input[i] - in_zp);
		}

		acc[o] = (int32_t)lrintf((float)acc[o] * multipliers[o]) + out_zp;

	    output[o] = clamp_int8(acc[o]);
	}

	return 1;
}


int relu_forward(const int8_t* input, int8_t* output, const ReluLayer* relu){
	if(!input || !output || !relu) { return -1; }
	int32_t z = relu->zp;
	size_t n = relu->dim;
	for(size_t i = 0; i < n; i++){
		int32_t v = input[i];
		if(v < z) v = z;
		output[i] = clamp_int8(v);
	}

	return 1;
}

int8_t clamp_int8(int32_t x) {
    if(x > 127) { x = 127; }
    if(x < -127) { x = -127; }
    return (int8_t)x;
}

void quantize(const float* x, int8_t* y, const size_t n, const float s, const int32_t zp){
	for(size_t i = 0; i < n; i++){
		y[i] = clamp_int8((int32_t)lrintf((float)x[i] / s) + zp);
	}
}

void dequantize(const int8_t* x, float* y, const size_t n, const float s, const int32_t zp){
	for(size_t i = 0; i < n; i++){
		y[i] = (x[i] - zp) * s;
	}
}



