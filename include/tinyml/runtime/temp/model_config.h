#ifndef MODEL_CONFIG
#define MODEL_CONFIG

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

#define MAX_ACT_SIZE 8

typedef struct {
	Arena* arena;

	// activation scratch
	int8_t* act0;
	int8_t* act1;
	size_t act_cap;

	// dense accumulation scratch
	int32_t* acc;
	size_t acc_cap;
} ModelCtx;

typedef enum { L_DENSE, L_RELU } LayerType;

typedef struct {
    LayerType type;
    const void* layer;
} LayerRef;

typedef struct {
	size_t in_dim;
	size_t out_dim;

	int32_t in_zp;
	float in_scale;

	int32_t out_zp;
	float out_scale;

	size_t num_layers;
	const LayerRef* layers;
} ModelRef;

typedef struct {
	size_t in_dim;
	size_t out_dim;

	const int8_t* weights; // in_dim x out_dim
	const int32_t* biases; // out_dim
	const float* multipliers; // out_dim

	int32_t in_zp;
	float in_scale;

	int32_t out_zp;
	float out_scale;
} DenseLayer;

typedef struct  {
	int32_t zp;
	size_t dim;
} ReluLayer;


int model_init_ctx(const ModelRef* m, ModelCtx* ctx, Arena* arena);
// forward functions
int model_forward(const ModelRef* m, const ModelCtx* ctx, const float* in, float* out);

// layer forward functions
int dense_forward(const int8_t* input, int8_t* output, const DenseLayer* dense, const ModelCtx* ctx);
int relu_forward(const int8_t* input, int8_t* output, const ReluLayer* relu);

// helper functions
int8_t clamp_int8(int32_t x);
void quantize(const float* x, int8_t* y, const size_t n, const float s, const int32_t zp);
void dequantize(const int8_t* x, float* y, const size_t n, const float s, const int32_t zp);

#endif
