#pragma once
#include "tinyml/quant/qdense.hpp"
#include <fstream>
#include <string>

namespace tinyml::codegen {

void generate_dense(std::ofstream& file, const quant::QDense& qlayer, const std::size_t layer_idx);

}
