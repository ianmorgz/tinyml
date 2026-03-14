#include "tinyml/codegen/codegen.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/quant/qsequential.hpp"
#include "tinyml/quant/qlayer.hpp"
#include "tinyml/quant/qdense.hpp"
#include "tinyml/quant/qrelu.hpp"
#include "tinyml/codegen/dense_gen.hpp"
#include "tinyml/codegen/relu_gen.hpp"

#include <filesystem>
#include <fstream>

#include "tinyml/codegen/model_gen.hpp"

namespace tinyml::codegen {

namespace fs = std::filesystem;

void generate(const quant::QSequential& model, const std::string &template_folder, const std::string &out_folder) {
    // copy_files(template_folder, out_folder);
    generate_data(model, out_folder);

}

void copy_files(const std::string &template_folder, const std::string& out_folder) {
    std::vector<std::string> files_to_copy = {"arena.h", "model.h", "model.c", "model_config.c", "model_config.h", "model_data.h" };
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

void generate_data(const quant::QSequential& model, const std::string& out_folder) {
    std::ofstream data_file(out_folder + "/model_data.c");
    if (!data_file.is_open()) { TINYML_EXCEPTION("Model Generation failed to open file"); }

    const std::size_t n = model.num_layers();
    for (std::size_t i = 0; i < n; i++) {
        const auto& l = model.get_layer(i);
        switch (l.type()) {
            case quant::QLayer_Type::QDense: {
                const auto& d = static_cast<const quant::QDense&>(l);
                generate_dense(data_file, d, i);
                break;
            }
            case quant::QLayer_Type::QReLu: {
                const auto& r = static_cast<const quant::QRelu&>(l);
                generate_relu(data_file, r, i);
                break;
            }
            default: {
                TINYML_EXCEPTION("Code Generation Unknown layer type");
            }
        }
    }

    generate_model(model, data_file);
}
}
