#include "tinyml/codegen/model_gen.hpp"

void tinyml::codegen::generate_model(const quant::QSequential &model, std::ofstream &file) {
    if (!file.is_open()) { TINYML_EXCEPTION("Dense Generation file is not open");}
    file << "\n// Model\n";
    file << "static const Model model = {\n";

    std::size_t num_l = model.num_layers();

}

