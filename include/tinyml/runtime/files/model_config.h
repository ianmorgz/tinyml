#pragma once

#include <stdint.h>
#include <stddef.h>

#include "arena.h"

// model arena managment
typedef struct {
    Arena* arena;

    // ping-pong activations
    int8_t* act0;
    int8_t* act1;
    size_t act_cap;

    // dense accumulator
    int32_t* acc;
    size_t acc_cap;
} ModelCtx;

// model
typedef enum { L_DENSE, L_RELU } LayerType;

// the reference point for the layers
typedef struct {
    LayerType type;
    const void* layer;
} LayerRef;

typedef struct {
    uint16_t in_dim;
    uint16_t out_dim;
    uint8_t num_layers;
    const LayerRef* layers;
} Model;

//layers
typedef struct {
    uint16_t in_dim;
    uint16_t out_dim;

    const int8_t* weights; // size in_dim * out_dim
    const int32_t* bias_acc; // size out_dim

    int32_t in_zp;  // input zero point
    int32_t out_zp; // output zero point

    const int32_t* mult; // size out_dim
    const int8_t* shift; //size out_dim

    const int32_t* w_row_sum; // size out_dim
} DenseLayer;

typedef struct {
    int32_t zp;
} ReLuLayer;

// initialization function
bool validate_model(const Model* m, size_t act_cap, size_t acc_cap);
bool model_init_ctx(const Model* m, ModelCtx* ctx, Arena* arena);

void model_init(Model* m);

// forward functions
int model_forward(const Model* m, ModelCtx* ctx, const int8_t* input, int8_t* output);

void dense_forward(const int8_t* input, int8_t* output, const DenseLayer* dense);
void relu_forward(const int8_t* input, int8_t* output, const ReLuLayer* relu);

// helper functions
static inline int8_t clamp_int8(int32_t x);
static inline int32_t shift_int64(int64_t x, int32_t shift);