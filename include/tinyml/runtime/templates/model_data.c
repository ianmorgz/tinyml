// This is where the flash storage data will live for the model
// Things like weights, bias accumulators, multipliers, and params
// name format <layer type><layer number>_<name> eg. dense0_W

#include "model_data.h"

// Layer 0 dense layer
static const int8_t dense0_W[784*128] = {
    0 // ...generated... //
};

static const int8_t dense0_Bacc[128] = {
    0 // ...generated... //
};

static const int32_t dense0_mult[128] = {
    0 // ...generated... //
};

static const int8_t dense0_shift[128] = {
    0 // ...generated... //
};

static const DenseLayer dense0_layer = {
    .in_dim    = 128,
    .out_dim   = 64,
    .weights         = dense0_W,
    .bias_acc     = dense0_Bacc,
    .in_zp     = 0,
    .out_zp    = 0,
    .mult      = dense0_mult,
    .shift     = dense0_shift,
  //   .w_row_sum = dense0_w_row_sum,
  };

const Model model = {
    .in_dim     = 784,
    .out_dim    = 10,
    .num_layers = 0,
    .layers     = 0,
};