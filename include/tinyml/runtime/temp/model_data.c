#include <model_data.h>
#include "model_config.h"

/* BEGIN GENERATED CODE */


//Layer 0 Dense
static const int8_t dense0_weights[16] = {
	 127, 127, -63, -127, -87, -127, -127, 127, 127, -126, 25, 127, 127, -59, 46, 127
};
static const int32_t dense0_biases[8] = {
	 -3642, 0, 0, -21, -20, 20855, -529, -60
};
static const float dense0_multipliers[8] = {
	 0.00339165, 0.00282164, 0.00351653, 0.00325097, 0.00102194, 0.00315884, 0.00343653, 0.00320197
};
static const DenseLayer dense0_layer = {
	.weights = dense0_weights,
	.biases = dense0_biases,
	.multipliers = dense0_multipliers,
	.in_scale = 0.00787402,
	.in_zp = 0,
	.in_dim = 2,
	.out_scale = 0.012571,
	.out_zp = 0,
	.out_dim = 8,
};

//Layer 1 Relu
static const ReluLayer relu1_layer = {
	.zp = 0,
	.dim = 8,
};


//Layer 2 Dense
static const int8_t dense2_weights[16] = {
	 110, 85, -40, -100, -73, -127, -71, 83, -2, 7, 22, 127, 112, -79, 43, 85
};
static const int32_t dense2_biases[2] = {
	 -8633, -13098
};
static const float dense2_multipliers[2] = {
	 0.00642069, 0.00600289
};
static const DenseLayer dense2_layer = {
	.weights = dense2_weights,
	.biases = dense2_biases,
	.multipliers = dense2_multipliers,
	.in_scale = 0.012571,
	.in_zp = 0,
	.in_dim = 8,
	.out_scale = 0.0139871,
	.out_zp = 0,
	.out_dim = 2,
};


static const LayerRef layers[3] = {
	{ .type = L_DENSE, .layer = &dense0_layer },
	{ .type = L_RELU, .layer = &relu1_layer },
	{ .type = L_DENSE, .layer = &dense2_layer },
};

const ModelRef model = {
	.in_dim = 2,
	.out_dim = 2,

	.in_scale = 0.00787402,
	.in_zp = 0,

	.out_scale = 0.0139871,
	.out_zp = 0,

	.num_layers = 3,
	.layers = layers
};


/* END GENERATED CODE */
