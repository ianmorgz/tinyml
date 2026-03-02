#include "tinyml/codegen/dense_gen.hpp"
#include "tinyml/core/config.hpp"
#include <fstream>

namespace tinyml::codegen {
void generate_dense(std::ofstream& file, const quant::QDense& qdense, const std::size_t layer_idx) {
    // open the file
    if (!file.is_open()) { TINYML_EXCEPTION("Dense Generation file is not open");}
    file << "\n//Layer " << layer_idx << " Dense\n";

    std::size_t in_f = qdense.in_features();
    std::size_t out_f = qdense.out_features();

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
    // generate the weights
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
}
}