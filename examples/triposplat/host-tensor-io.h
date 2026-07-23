#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace triposplat {

// Host-side tensor types use safetensors/PyTorch dimension order. The last
// dimension is contiguous, so payloads can be passed to ggml unchanged after
// reversing shape into ggml's ne[] order.
enum class host_dtype {
    u8,
    i8,
    i16,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,
};

const char * host_dtype_name(host_dtype type);
host_dtype host_dtype_from_name(const std::string & name);
size_t host_dtype_size(host_dtype type);

struct host_tensor {
    std::string name;
    host_dtype type = host_dtype::f32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;

    size_t element_count() const;
    size_t expected_byte_count() const;
    void validate() const;
};

// allocate_host_tensor is the convenient export path for an e2e CLI:
// allocate it, fill bytes with ggml_backend_tensor_get(), then save it.
host_tensor allocate_host_tensor(std::string name, host_dtype type,
                                 std::vector<int64_t> shape);

host_tensor copy_host_tensor(std::string name, host_dtype type,
                             std::vector<int64_t> shape,
                             const void * data, size_t byte_count);

template <typename T>
host_tensor copy_host_tensor(std::string name, host_dtype type,
                             std::vector<int64_t> shape,
                             const std::vector<T> & values) {
    return copy_host_tensor(std::move(name), type, std::move(shape),
                            values.data(), values.size() * sizeof(T));
}

struct host_tensor_archive {
    std::vector<host_tensor> tensors;
    std::map<std::string, std::string> metadata;

    const host_tensor * find(const std::string & name) const;
    const host_tensor & at(const std::string & name) const;
};

// save_safetensors emits tensors in lexical name order and pads the JSON
// header to eight bytes. Identical inputs therefore produce identical files.
void save_safetensors(const std::string & path,
                      const std::vector<host_tensor> & tensors,
                      const std::map<std::string, std::string> & metadata = {});

// The loader owns every payload and validates dtype, shape, bounds, overlap,
// and exact byte counts. It is intended for small e2e inputs/results; large
// model weights should continue to use the mmap-based safetensors_file.
host_tensor_archive load_host_safetensors(const std::string & path);

// PCG32 gives a stable integer/uniform stream for a fixed (seed, sequence).
// normal_f32 uses a fixed Box-Muller draw order and caches the second sample.
class deterministic_rng {
public:
    explicit deterministic_rng(uint64_t seed = 0, uint64_t sequence = 1);

    void reseed(uint64_t seed, uint64_t sequence = 1);
    uint32_t next_u32();
    uint32_t uniform_u32(uint32_t bound);
    float uniform_f32();
    float normal_f32();

    void fill_uniform(float * values, size_t count,
                      float lower = 0.0f, float upper = 1.0f);
    void fill_normal(float * values, size_t count,
                     float mean = 0.0f, float standard_deviation = 1.0f);

private:
    uint64_t state_ = 0;
    uint64_t increment_ = 1;
    float spare_normal_ = 0.0f;
    bool has_spare_normal_ = false;
};

} // namespace triposplat
