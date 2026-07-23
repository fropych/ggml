#include "host-tensor-io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace triposplat {
namespace {

constexpr uint64_t kMaximumHeaderBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kPcgMultiplier = 6364136223846793005ull;
constexpr double kTwoPi = 6.283185307179586476925286766559;

uint64_t read_u64_le(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | bytes[i];
    return value;
}

void write_u64_le(std::ostream & output, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = uint8_t(value & 0xffu);
        value >>= 8;
    }
    output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

size_t checked_element_count(const std::vector<int64_t> & shape) {
    size_t count = 1;
    bool zero = false;
    for (int64_t dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("tensor shape has a negative dimension");
        if (dimension == 0) {
            zero = true;
            continue;
        }
        if (!zero && count > std::numeric_limits<size_t>::max() / size_t(dimension)) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        if (!zero) count *= size_t(dimension);
    }
    return zero ? 0 : count;
}

uint64_t checked_add_u64(uint64_t first, uint64_t second, const char * description) {
    if (first > std::numeric_limits<uint64_t>::max() - second) {
        throw std::overflow_error(description);
    }
    return first + second;
}

void read_exact(std::istream & input, void * destination, size_t byte_count,
                const std::string & path, const char * description) {
    if (byte_count == 0) return;
    input.read(static_cast<char *>(destination), std::streamsize(byte_count));
    if (!input) throw std::runtime_error(path + ": truncated " + description);
}

} // namespace

const char * host_dtype_name(host_dtype type) {
    switch (type) {
        case host_dtype::u8:   return "U8";
        case host_dtype::i8:   return "I8";
        case host_dtype::i16:  return "I16";
        case host_dtype::i32:  return "I32";
        case host_dtype::i64:  return "I64";
        case host_dtype::f16:  return "F16";
        case host_dtype::bf16: return "BF16";
        case host_dtype::f32:  return "F32";
        case host_dtype::f64:  return "F64";
    }
    throw std::invalid_argument("unknown host tensor dtype");
}

host_dtype host_dtype_from_name(const std::string & name) {
    if (name == "U8")   return host_dtype::u8;
    if (name == "I8")   return host_dtype::i8;
    if (name == "I16")  return host_dtype::i16;
    if (name == "I32")  return host_dtype::i32;
    if (name == "I64")  return host_dtype::i64;
    if (name == "F16")  return host_dtype::f16;
    if (name == "BF16") return host_dtype::bf16;
    if (name == "F32")  return host_dtype::f32;
    if (name == "F64")  return host_dtype::f64;
    throw std::invalid_argument("unsupported safetensors dtype: " + name);
}

size_t host_dtype_size(host_dtype type) {
    switch (type) {
        case host_dtype::u8:
        case host_dtype::i8:
            return 1;
        case host_dtype::i16:
        case host_dtype::f16:
        case host_dtype::bf16:
            return 2;
        case host_dtype::i32:
        case host_dtype::f32:
            return 4;
        case host_dtype::i64:
        case host_dtype::f64:
            return 8;
    }
    throw std::invalid_argument("unknown host tensor dtype");
}

size_t host_tensor::element_count() const {
    return checked_element_count(shape);
}

size_t host_tensor::expected_byte_count() const {
    const size_t count = element_count();
    const size_t width = host_dtype_size(type);
    if (count > std::numeric_limits<size_t>::max() / width) {
        throw std::overflow_error("tensor byte count overflows size_t");
    }
    return count * width;
}

void host_tensor::validate() const {
    if (name.empty()) throw std::invalid_argument("tensor name must not be empty");
    if (name == "__metadata__") {
        throw std::invalid_argument("__metadata__ is reserved by safetensors");
    }
    const size_t expected = expected_byte_count();
    if (bytes.size() != expected) {
        throw std::invalid_argument("tensor " + name + " has " +
            std::to_string(bytes.size()) + " bytes; expected " + std::to_string(expected));
    }
}

host_tensor allocate_host_tensor(std::string name, host_dtype type,
                                 std::vector<int64_t> shape) {
    host_tensor result;
    result.name = std::move(name);
    result.type = type;
    result.shape = std::move(shape);
    result.bytes.resize(result.expected_byte_count());
    result.validate();
    return result;
}

host_tensor copy_host_tensor(std::string name, host_dtype type,
                             std::vector<int64_t> shape,
                             const void * data, size_t byte_count) {
    if (byte_count != 0 && data == nullptr) {
        throw std::invalid_argument("non-empty tensor payload has a null source");
    }
    host_tensor result = allocate_host_tensor(std::move(name), type, std::move(shape));
    if (result.bytes.size() != byte_count) {
        throw std::invalid_argument("source byte count does not match tensor dtype and shape");
    }
    if (byte_count != 0) std::memcpy(result.bytes.data(), data, byte_count);
    return result;
}

const host_tensor * host_tensor_archive::find(const std::string & name) const {
    for (const host_tensor & tensor : tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

const host_tensor & host_tensor_archive::at(const std::string & name) const {
    const host_tensor * tensor = find(name);
    if (!tensor) throw std::out_of_range("missing host tensor: " + name);
    return *tensor;
}

void save_safetensors(const std::string & path,
                      const std::vector<host_tensor> & tensors,
                      const std::map<std::string, std::string> & metadata) {
    std::vector<const host_tensor *> ordered;
    ordered.reserve(tensors.size());
    for (const host_tensor & tensor : tensors) {
        tensor.validate();
        ordered.push_back(&tensor);
    }
    std::sort(ordered.begin(), ordered.end(), [](const host_tensor * first,
                                                  const host_tensor * second) {
        return first->name < second->name;
    });
    for (size_t i = 1; i < ordered.size(); ++i) {
        if (ordered[i - 1]->name == ordered[i]->name) {
            throw std::invalid_argument("duplicate tensor name: " + ordered[i]->name);
        }
    }

    nlohmann::ordered_json header = nlohmann::ordered_json::object();
    if (!metadata.empty()) header["__metadata__"] = metadata;
    uint64_t offset = 0;
    for (const host_tensor * tensor : ordered) {
        const uint64_t end = checked_add_u64(
            offset, uint64_t(tensor->bytes.size()), "safetensors payload offset overflow");
        header[tensor->name] = {
            {"dtype", host_dtype_name(tensor->type)},
            {"shape", tensor->shape},
            {"data_offsets", {offset, end}},
        };
        offset = end;
    }

    std::string header_text = header.dump();
    while ((header_text.size() % 8) != 0) header_text.push_back(' ');
    if (header_text.size() > kMaximumHeaderBytes) {
        throw std::runtime_error("safetensors header exceeds 64 MiB");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open " + path + " for writing");
    write_u64_le(output, uint64_t(header_text.size()));
    output.write(header_text.data(), std::streamsize(header_text.size()));
    for (const host_tensor * tensor : ordered) {
        if (!tensor->bytes.empty()) {
            output.write(reinterpret_cast<const char *>(tensor->bytes.data()),
                         std::streamsize(tensor->bytes.size()));
        }
    }
    output.flush();
    if (!output) throw std::runtime_error("failed to write " + path);
}

host_tensor_archive load_host_safetensors(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path + " for reading");
    const std::streamoff end_position = input.tellg();
    if (end_position < 8) throw std::runtime_error(path + ": truncated safetensors file");
    const uint64_t file_size = uint64_t(end_position);
    input.seekg(0);

    uint8_t encoded_header_size[8];
    read_exact(input, encoded_header_size, sizeof(encoded_header_size), path, "header length");
    const uint64_t header_size = read_u64_le(encoded_header_size);
    if (header_size > kMaximumHeaderBytes || header_size > file_size - 8) {
        throw std::runtime_error(path + ": invalid safetensors header size");
    }
    std::string header_text(size_t(header_size), '\0');
    read_exact(input, header_text.data(), header_text.size(), path, "JSON header");

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_text);
    } catch (const std::exception & error) {
        throw std::runtime_error(path + ": invalid safetensors JSON: " + error.what());
    }
    if (!header.is_object()) throw std::runtime_error(path + ": safetensors header is not an object");

    struct range {
        uint64_t begin;
        uint64_t end;
        size_t tensor_index;
    };
    host_tensor_archive archive;
    std::vector<range> ranges;
    const uint64_t data_offset = 8 + header_size;
    const uint64_t payload_size = file_size - data_offset;

    for (auto item = header.begin(); item != header.end(); ++item) {
        if (item.key() == "__metadata__") {
            if (!item.value().is_object()) {
                throw std::runtime_error(path + ": __metadata__ must be an object");
            }
            for (auto entry = item.value().begin(); entry != item.value().end(); ++entry) {
                if (!entry.value().is_string()) {
                    throw std::runtime_error(path + ": metadata values must be strings");
                }
                archive.metadata.emplace(entry.key(), entry.value().get<std::string>());
            }
            continue;
        }

        try {
            if (!item.value().is_object()) throw std::invalid_argument("descriptor is not an object");
            host_tensor tensor;
            tensor.name = item.key();
            tensor.type = host_dtype_from_name(item.value().at("dtype").get<std::string>());
            tensor.shape = item.value().at("shape").get<std::vector<int64_t>>();
            const std::vector<uint64_t> offsets =
                item.value().at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > payload_size) {
                throw std::invalid_argument("invalid data_offsets");
            }
            const uint64_t byte_count = offsets[1] - offsets[0];
            if (byte_count > std::numeric_limits<size_t>::max()) {
                throw std::overflow_error("tensor payload is too large for this host");
            }
            if (tensor.expected_byte_count() != size_t(byte_count)) {
                throw std::invalid_argument("payload byte count does not match dtype and shape");
            }
            tensor.bytes.resize(size_t(byte_count));
            archive.tensors.push_back(std::move(tensor));
            ranges.push_back({offsets[0], offsets[1], archive.tensors.size() - 1});
        } catch (const std::exception & error) {
            throw std::runtime_error(path + ": invalid tensor " + item.key() + ": " + error.what());
        }
    }

    std::sort(ranges.begin(), ranges.end(), [](const range & first, const range & second) {
        if (first.begin != second.begin) return first.begin < second.begin;
        // Zero-sized tensors may share the next non-empty tensor's offset.
        return first.end < second.end;
    });
    uint64_t expected_begin = 0;
    for (const range & current : ranges) {
        if (current.begin != expected_begin) {
            throw std::runtime_error(path + ": tensor payloads overlap or leave a hole");
        }
        expected_begin = current.end;
        host_tensor & tensor = archive.tensors[current.tensor_index];
        input.clear();
        input.seekg(std::streamoff(data_offset + current.begin));
        if (!input) throw std::runtime_error(path + ": invalid tensor payload seek");
        read_exact(input, tensor.bytes.data(), tensor.bytes.size(), path, "tensor payload");
        tensor.validate();
    }
    if (expected_begin != payload_size) {
        throw std::runtime_error(path + ": unindexed bytes at end of tensor payload");
    }
    std::sort(archive.tensors.begin(), archive.tensors.end(),
              [](const host_tensor & first, const host_tensor & second) {
        return first.name < second.name;
    });
    return archive;
}

deterministic_rng::deterministic_rng(uint64_t seed, uint64_t sequence) {
    reseed(seed, sequence);
}

void deterministic_rng::reseed(uint64_t seed, uint64_t sequence) {
    state_ = 0;
    increment_ = (sequence << 1u) | 1u;
    has_spare_normal_ = false;
    (void) next_u32();
    state_ += seed;
    (void) next_u32();
}

uint32_t deterministic_rng::next_u32() {
    const uint64_t old_state = state_;
    state_ = old_state * kPcgMultiplier + increment_;
    const uint32_t xorshifted = uint32_t(((old_state >> 18u) ^ old_state) >> 27u);
    const uint32_t rotation = uint32_t(old_state >> 59u);
    return (xorshifted >> rotation) | (xorshifted << ((0u - rotation) & 31u));
}

uint32_t deterministic_rng::uniform_u32(uint32_t bound) {
    if (bound == 0) throw std::invalid_argument("uniform_u32 bound must be positive");
    const uint32_t threshold = uint32_t(0u - bound) % bound;
    for (;;) {
        const uint32_t value = next_u32();
        if (value >= threshold) return value % bound;
    }
}

float deterministic_rng::uniform_f32() {
    return float(next_u32() >> 8u) * (1.0f / 16777216.0f);
}

float deterministic_rng::normal_f32() {
    if (has_spare_normal_) {
        has_spare_normal_ = false;
        return spare_normal_;
    }
    const double first = (double(next_u32()) + 1.0) / 4294967297.0;
    const double second = (double(next_u32()) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(first));
    const double angle = kTwoPi * second;
    spare_normal_ = float(radius * std::sin(angle));
    has_spare_normal_ = true;
    return float(radius * std::cos(angle));
}

void deterministic_rng::fill_uniform(float * values, size_t count,
                                     float lower, float upper) {
    if (count != 0 && values == nullptr) throw std::invalid_argument("null uniform output");
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
        throw std::invalid_argument("invalid uniform interval");
    }
    const float scale = upper - lower;
    for (size_t i = 0; i < count; ++i) values[i] = lower + scale * uniform_f32();
}

void deterministic_rng::fill_normal(float * values, size_t count,
                                    float mean, float standard_deviation) {
    if (count != 0 && values == nullptr) throw std::invalid_argument("null normal output");
    if (!std::isfinite(mean) || !std::isfinite(standard_deviation) ||
        standard_deviation < 0.0f) {
        throw std::invalid_argument("invalid normal distribution parameters");
    }
    for (size_t i = 0; i < count; ++i) {
        values[i] = mean + standard_deviation * normal_f32();
    }
}

} // namespace triposplat
