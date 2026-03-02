#pragma once
#include "tinyml/quant/qsequential.hpp"
namespace tinyml::codegen {

void generate(const quant::QSequential& model, const std::string& template_path, const std::string& out_path, const std::string& model_name);
void copy_files(const std::string &template_path, const std::string& out_path);
void generate_data(const quant::QSequential& model, const std::string& out_folder);

}