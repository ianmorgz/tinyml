#pragma once
#include "tinyml/quant/qsequential.hpp"
namespace tinyml::codegen {

void generate(quant::QSequential& model, std::string template_path, std::string out_path, std::string model_name);
void copy_files(std::string template_path, std::string out_path);

}