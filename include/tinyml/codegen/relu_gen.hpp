#pragma once
#include <fstream>
#include "tinyml/quant/qrelu.hpp"

namespace tinyml::codegen {

void generate_relu(std::ofstream& file, const quant::QRelu& qrelu, std::size_t layer_idx);

}