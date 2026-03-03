#include "tinyml/codegen/relu_gen.hpp"
#include "tinyml/quant/qrelu.hpp"
#include <fstream>

void tinyml::codegen::generate_relu(std::ofstream &file, const quant::QRelu &qrelu, std::size_t layer_idx) {
    if (!file.is_open()) { TINYML_EXCEPTION("Dense Generation file is not open");}
    file << "\n//Layer " << layer_idx << " Relu\n";
    const auto qparam = qrelu.param();
    file << "static const ReluLayer relu" << layer_idx << "_layer = {\n";
    file << "\t.zp = " << static_cast<int>(qparam->zero_point) << ",\n";
    file << "};\n";
}
