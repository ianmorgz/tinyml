#pragma once
#include "tinyml/dataset/loader.hpp"
#include "tinyml/dataset/batch.hpp"
#include "tinyml/core/config.hpp"

#include <string>
#include <fstream>
#include <vector>

using namespace tinyml;

class MNISTLoader : public dataset::Loader {
public:
    MNISTLoader(std::string images_path, std::string labels_path) : images_path_(std::move(images_path)), labels_path_(std::move(labels_path)) {}

    void load(dataset::Batch& out) const override {
        // Read the labels
        std::ifstream lf(labels_path_, std::ios::binary);
        if (!lf) TINYML_EXCEPTION("MNIST: failed to open labels file");

        const std::uint32_t lbl_magic = read_be_u32_(lf);
        const std::uint32_t lbl_count = read_be_u32_(lf);
        if (lbl_magic != 2049u) TINYML_EXCEPTION("MNIST: labels magic mismatch (expected 2049)");

        std::vector<std::uint8_t> labels(lbl_count);
        lf.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(labels.size()));
        if (!lf) TINYML_EXCEPTION("MNIST: failed to read labels payload");

        // Read the images
        std::ifstream im(images_path_, std::ios::binary);
        if (!im) TINYML_EXCEPTION("MNIST: failed to open images file");

        const std::uint32_t img_magic = read_be_u32_(im);
        const std::uint32_t img_count = read_be_u32_(im);
        const std::uint32_t rows = read_be_u32_(im);
        const std::uint32_t cols = read_be_u32_(im);
        if (img_magic != 2051u) TINYML_EXCEPTION("MNIST: images magic mismatch (expected 2051)");
        if (rows == 0 || cols == 0) TINYML_EXCEPTION("MNIST: invalid image dimensions");
        if (img_count != lbl_count) TINYML_EXCEPTION("MNIST: image/label count mismatch");

        const std::size_t pixels = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        const auto total = static_cast<std::size_t>(img_count);

        std::size_t N = total;

        // ---- Allocate output tensors ----
        out.size = N;

        out.input = tinyml::tensor::Tensor<float>(tinyml::core::Shape{
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(pixels)
        });

        constexpr std::size_t label_dim = 10;
        out.label = tensor::Tensor<float>(core::Shape{
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(label_dim)
        });

        float* X = out.input.data();
        float* Y = out.label.data();

        // ---- Stream images, write floats ----
        // Each image is pixels bytes.
        std::vector<std::uint8_t> buf(pixels);

        for (std::size_t i = 0; i < N; ++i) {
            im.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            if (!im) TINYML_EXCEPTION("MNIST: failed to read image payload");



            // X row i
            float* xrow = X + i * pixels;
            for (std::size_t p = 0; p < pixels; ++p) {
                xrow[p] = normalize(buf[p]);
            }

            // Y row i
            const std::uint8_t lbl = labels[i];
            if (lbl > 9) TINYML_EXCEPTION("MNIST: label out of range");
            float* yrow = Y + i * label_dim;

            // zero then set correct label
            for (std::size_t k = 0; k < 10; ++k) yrow[k] = 0.0f;
            yrow[lbl] = 1.0f;
        }
    }

private:
    static float normalize(const float in) { return in * (1.0f / 255.0f); }

    static std::uint32_t read_be_u32_(std::ifstream& f) {
        std::uint8_t b[4]{};
        f.read(reinterpret_cast<char*>(b), 4);
        if (!f) TINYML_EXCEPTION("MNIST: failed to read u32");
        return (static_cast<std::uint32_t>(b[0]) << 24) |
               (static_cast<std::uint32_t>(b[1]) << 16) |
               (static_cast<std::uint32_t>(b[2]) <<  8) |
               (static_cast<std::uint32_t>(b[3]) <<  0);
    }

    std::string images_path_;
    std::string labels_path_;
};
