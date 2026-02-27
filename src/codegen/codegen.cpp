#include "tinyml/codegen/codegen.hpp"
#include "tinyml/core/config.hpp"

#include <filesystem>
#include <iostream>

namespace tinyml::codegen {

namespace fs = std::filesystem;

void generate(quant::QSequential& model, std::string template_folder, std::string out_folder, std::string model_name) {
    copy_files(template_folder, out_folder);
}

void copy_files(std::string template_folder, std::string out_folder) {
    std::vector<std::string> files_to_copy = {"arena.h", "model.h", "model.c", "model_config.c", "model_config.h" };
    try {
        fs::path template_path = template_folder;
        fs::path out_path = out_folder;
        for (std::string file : files_to_copy) {
            fs::path f = file;
            fs::copy_file(template_path / file, out_path / file, fs::copy_options::overwrite_existing);
        }
    } catch (const std::exception& e) {
        TINYML_EXCEPTION("code generation failed to copy file");
    }
}

}