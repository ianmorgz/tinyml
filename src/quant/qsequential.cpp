#include "tinyml/quant/qsequential.hpp"

#include <iostream>

#include "tinyml/dataset/dataset.hpp"
#include "tinyml/model/dense.hpp"
#include "tinyml/model/layer.hpp"
#include "tinyml/quant/qdense.hpp"
#include "tinyml/quant/qrelu.hpp"
#include "tinyml/quant/qtanh.hpp"
#include "tinyml/model/sequential.hpp"

namespace tinyml::quant {
QSequential::QSequential(model::Sequential& net, dataset::Dataset& ds, const std::size_t callibration_size) {
    std::cout << "=================== TINY ML QUANTIZATION ===================\n";

    std::cout << "\rQuantizing..." << std::flush;

    // step 1 create the layers and quantize dense weights
    make_layers(net);

    c_min = std::numeric_limits<float>::infinity();
    c_max = -std::numeric_limits<float>::infinity();

    // step 2 callibrate pass on the float network
    callibrate(net, ds, callibration_size);

    // step 3 finalize model qparams + build LUTS
    finalize(net);
    // step 4 quantize dense biases

    std::cout << "\r\033[2K" << "Quantization complete\n"; // clear the testing... placeholder

}
void QSequential::make_layers(const model::Sequential &net) {
    num_layers = net.num_layers();
    layer_sizes_ = net.layer_sizes();
    max_layer_size_ = net.max_features();
    layers_.reserve(num_layers);

    for (size_t i = 0; i < num_layers; ++i) {
        const auto& L = net.layer(i);
        // create empty qLayers by type
        switch (L.type()) {
            case model::LayerType::Dense: {
                const auto& d = static_cast<const model::Dense&>(L);
                layers_.push_back(std::make_unique<QDense>(d));
                break;
            }

            case model::LayerType::ReLu: {
                layers_.push_back(std::make_unique<QRelu>());
                break;
            }

            default:
                TINYML_EXCEPTION("Quantized Network make layers undefined layer type used");
        }
    }
}

void QSequential::callibrate(model::Sequential &net, dataset::Dataset &ds, const std::size_t callibration_size) {
    const std::size_t max_batch_size = net.max_batch();
    dataset::BatchView callibration_batch;
    ds.shuffle_training(42);

    tensor::Tensor<float> act_a({max_batch_size, max_layer_size_});
    tensor::Tensor<float> act_b({max_batch_size, max_layer_size_});

    for (std::size_t i = 0; i < callibration_size; i += max_batch_size) {
        const std::size_t remaining = callibration_size - i;
        const std::size_t batch_size = (remaining < max_batch_size) ? remaining : max_batch_size;
        ds.next_training_batch(batch_size, callibration_batch);

        const float* o = callibration_batch.input.data();
        const std::size_t n = callibration_batch.input.size();

        for (std::size_t j = 0; j < n; j++) {
            if (o[j] > c_max) { c_max = o[j]; }
            if (o[j] < c_min) { c_min = o[j]; }
        }

        tensor::TensorView<const float> in_view = callibration_batch.input;

        bool write_to_a = true;

        for (std::size_t j = 0; j < num_layers; ++j) {
            tensor::TensorView out_view(write_to_a ? act_a.data() : act_b.data(), {batch_size,  layer_sizes_[j+1]});

            net.layer(j).forward(in_view, out_view);

            layers_[j]->observe_fp32_output(out_view.as_const());

            in_view = out_view.as_const();
            write_to_a = !write_to_a;
        }
    }
}

void QSequential::finalize(model::Sequential &net) {
    input_param_ = QParam(c_max, c_min, QType::Asymmetric);
    QParam qp = input_param_;

    for (std::size_t i = 0; i < num_layers; ++i) {
        qp = layers_.at(i)->finalize_callibration(net.layer(i), qp);
        if (i == num_layers - 1) {
            output_param_ = qp;
        }
    }
}

// forward pass
void QSequential::forward(const tensor::TensorView<const float> &in, tensor::TensorView<float> &out) const{
    // no batches for forward quantized as it's not realistic
    if (in.size() != layer_sizes_[0]) {
        TINYML_EXCEPTION("Quantized Sequential model forward given invalid size");
    }

    tensor::Tensor<int8_t> act_a({max_layer_size_});
    tensor::Tensor<int8_t> act_b({max_layer_size_});

    bool write_to_a = true;

    tensor::Tensor<int8_t> q_in(in.shape());
    QParam::quantize_i8(in, q_in.view(), input_param_);

    auto in_view = q_in.view().as_const();

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        core::Shape out_shape({layer_sizes_[i+1]});

        tensor::TensorView<int8_t> out_view = write_to_a
        ? tensor::TensorView(act_a.data(), out_shape)
        : tensor::TensorView(act_b.data(), out_shape);

        layers_[i]->forward(in_view, out_view);
        in_view = out_view.as_const();
        write_to_a = !write_to_a;
    }


    QParam::dequantize_i8(in_view, out, output_param_);

}

}
