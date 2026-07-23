#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace triposplat {

struct safetensor_info {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

// A direct safetensors -> ggml loader. Tensor dimensions are reversed because
// safetensors/PyTorch stores the last dimension contiguously while ggml calls
// that contiguous dimension ne[0]. No payload transpose is necessary.
class safetensors_file {
public:
    explicit safetensors_file(const std::string & path);
    ~safetensors_file();
    safetensors_file(const safetensors_file &) = delete;
    safetensors_file & operator=(const safetensors_file &) = delete;

    const std::string & path() const { return path_; }
    const std::vector<safetensor_info> & tensors() const { return tensors_; }
    const safetensor_info & at(const std::string & name) const;
    const uint8_t * data(const safetensor_info & info) const;

private:
    std::string path_;
    int fd_ = -1;
    size_t size_ = 0;
    uint64_t data_offset_ = 0;
    const uint8_t * mapping_ = nullptr;
    std::vector<safetensor_info> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

enum class weight_policy {
    native,
    f16, // convert F32/BF16 to F16 and I64 indices to Vulkan-friendly I32
};

class weight_store {
public:
    weight_store(ggml_backend_t backend, const std::string & path,
                 weight_policy policy = weight_policy::native);
    ~weight_store();
    weight_store(const weight_store &) = delete;
    weight_store & operator=(const weight_store &) = delete;

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * maybe(const std::string & name) const;
    size_t tensor_count() const { return tensors_.size(); }
    size_t bytes() const { return bytes_; }

private:
    std::unique_ptr<safetensors_file> file_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    size_t bytes_ = 0;
};

} // namespace triposplat
