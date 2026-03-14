#include "tinyml/codegen/model_gen.hpp"

void tinyml::codegen::generate_model(const quant::QSequential &model, std::ofstream &file) {
    if (!file.is_open()) { TINYML_EXCEPTION("Dense Generation file is not open");}
    file << "\n// Model\n";
    file << "static const Model model = {\n";
    file << "\t.in_scale = " << model.in_param()->scale << "\n";
    file << "\t.in_zp = " << static_cast<int>(model.in_param()->zero_point) << "\n";

    file << "\t.out_scale = " << model.out_param()->scale << "\n";
    file << "\t.out_zp = " << static_cast<int>(model.out_param()->zero_point) << "\n";
    file << "};\n";

    std::size_t num_l = model.num_layers();

}

