#include "tinyml/codegen/dense_gen.hpp"
#include "tinyml/core/config.hpp"
#include <fstream>
#include <math.h>

namespace tinyml::codegen {
void generate_dense(std::ofstream& file, const quant::QDense& qdense, const std::size_t layer_idx) {
    // open the file
    if (!file.is_open()) { TINYML_EXCEPTION("Dense Generation file is not open");}
    file << "\n//Layer " << layer_idx << " Dense\n";

    const std::size_t in_f = qdense.in_features();
    const std::size_t out_f = qdense.out_features();

    // generate the weights
    file << "static const int8_t dense" << layer_idx << "_weights[" << (in_f * out_f) << "] = {\n";
    file << "\t";
    const int8_t* w = qdense.weights().data();

    for (std::size_t i = 0; i < out_f * in_f; i++) {
        file << " " << static_cast<int>(w[i]);
        if (i+1 != out_f * in_f) {
            file << ",";
        }
    }
    file << "\n};\n";

    // generate the biases
    file << "static const int32_t dense" << layer_idx << "_biases[" << (out_f) << "] = {\n";
    file << "\t";
    const int32_t* b = qdense.biases().data();

    for (std::size_t i = 0; i < out_f; i++) {
        file << " " << static_cast<int>(b[i]);
        if (i+1 != out_f) {
            file << ",";
        }
    }
    file << "\n};\n";

    // generate weight sums
    file << "static const float dense" << layer_idx << "_multipliers[" << (out_f) << "] = {\n";
    file << "\t";
    const float* w_s = qdense.weight_scales().data();
    const float in_scale = qdense.in_param()->scale;
    const float out_scale = qdense.out_param()->scale;

    float m = 0.0f;
    for (std::size_t i = 0; i < out_f; i++) {
        m = (w_s[i] * in_scale) / out_scale;
        file << " " << (m);
        if (i+1 != out_f) {
            file << ",";
        }
    }
    file << "\n};\n";

    // generate the layer structure
    file << "static const DenseLayer dense" << layer_idx << "_layer = {\n";

    const int32_t in_zp = qdense.in_param()->zero_point;
    const int32_t out_zp = qdense.out_param()->zero_point;

    file << "\t" << ".weights = dense" << layer_idx << "_weights,\n";
    file << "\t" << ".biases = dense" << layer_idx << "_biases,\n";
    file << "\t" << ".multipliers = dense" << layer_idx << "_multipliers,\n";

    file << "\t" << ".in_scale = " << in_scale << ",\n";
    file << "\t" << ".in_zp = " << in_zp << ",\n";
    file << "\t" << ".in_dim = " << in_f << ",\n";

    file << "\t" << ".out_scale = " << out_scale << ",\n";
    file << "\t" << ".out_zp = " << out_zp << ",\n";
    file << "\t" << ".out_dim = " << out_f << ",\n";
    file << "};\n";
}
}