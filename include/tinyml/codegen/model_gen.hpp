#pragma once
#include "tinyml/quant/qsequential.hpp"
#include <fstream>


namespace tinyml::codegen {

void generate_model(const quant::QSequential& model, std::ofstream& file);

}