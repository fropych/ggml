#include "safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#ifndef _WIN32
#include <fcntl.h>
#endif
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace triposplat {
namespace {

uint64_t read_u64_le(const uint8_t * p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | p[i];
    }
    return value;
}

#ifdef _WIN32
std::string windows_error_message(DWORD code) {
    char * message = nullptr;
    const DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char *>(&message), 0, nullptr);
    std::string result = size && message ? std::string(message, size) :
        "Windows error " + std::to_string(code);
    if (message) LocalFree(message);
    while (!result.empty() &&
           (result.back() == '\r' || result.back() == '\n' || result.back() == ' ')) {
        result.pop_back();
    }
    return result;
}
#endif

ggml_type native_type(const std::string & dtype) {
    if (dtype == "F16")  return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "F32")  return GGML_TYPE_F32;
    if (dtype == "I64")  return GGML_TYPE_I64;
    if (dtype == "I32")  return GGML_TYPE_I32;
    throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<ggml_fp16_t> convert_to_f16(const safetensors_file & file,
                                        const safetensor_info & info) {
    size_t count = 1;
    for (int64_t dim : info.shape) count *= size_t(dim);
    std::vector<ggml_fp16_t> out(count);
    const uint8_t * src = file.data(info);
    if (info.dtype == "F32") {
        const float * values = reinterpret_cast<const float *>(src);
        for (size_t i = 0; i < count; ++i) out[i] = ggml_fp32_to_fp16(values[i]);
    } else if (info.dtype == "BF16") {
        const uint16_t * values = reinterpret_cast<const uint16_t *>(src);
        for (size_t i = 0; i < count; ++i) out[i] = ggml_fp32_to_fp16(bf16_to_float(values[i]));
    } else {
        throw std::runtime_error("cannot convert " + info.dtype + " to F16");
    }
    return out;
}

std::vector<int32_t> convert_to_i32(const safetensors_file & file,
                                    const safetensor_info & info) {
    size_t count = 1;
    for (int64_t dim : info.shape) count *= size_t(dim);
    std::vector<int32_t> out(count);
    const int64_t * values = reinterpret_cast<const int64_t *>(file.data(info));
    for (size_t i = 0; i < count; ++i) {
        if (values[i] < std::numeric_limits<int32_t>::min() ||
            values[i] > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("I64 value out of I32 range in " + info.name);
        }
        out[i] = (int32_t) values[i];
    }
    return out;
}

} // namespace

safetensors_file::safetensors_file(const std::string & path) : path_(path) {
#ifdef _WIN32
    const std::wstring wide_path = std::filesystem::u8path(path).wstring();
    HANDLE file = CreateFileW(wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("open " + path + ": " +
                                 windows_error_message(GetLastError()));
    }
    file_handle_ = file;
    LARGE_INTEGER file_size {};
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0 ||
        uint64_t(file_size.QuadPart) > std::numeric_limits<size_t>::max()) {
        const DWORD error = GetLastError();
        release_mapping();
        throw std::runtime_error("get size " + path + ": " +
                                 windows_error_message(error));
    }
    size_ = size_t(file_size.QuadPart);
    if (size_ < 8) {
        release_mapping();
        throw std::runtime_error(path + ": truncated safetensors file");
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        const DWORD error = GetLastError();
        release_mapping();
        throw std::runtime_error("map " + path + ": " +
                                 windows_error_message(error));
    }
    mapping_handle_ = mapping;
    mapping_ = static_cast<const uint8_t *>(
        MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (!mapping_) {
        const DWORD error = GetLastError();
        release_mapping();
        throw std::runtime_error("map view " + path + ": " +
                                 windows_error_message(error));
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        const int error = errno;
        release_mapping();
        throw std::runtime_error("fstat " + path + ": " + std::strerror(error));
    }
    size_ = size_t(st.st_size);
    if (size_ < 8) {
        release_mapping();
        throw std::runtime_error(path + ": truncated safetensors file");
    }
    void * mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        const int error = errno;
        release_mapping();
        throw std::runtime_error("mmap " + path + ": " + std::strerror(error));
    }
    mapping_ = static_cast<const uint8_t *>(mapped);
#endif

    try {
        const uint64_t header_size = read_u64_le(mapping_);
        if (header_size > size_ - 8) throw std::runtime_error(path + ": invalid header size");
        data_offset_ = 8 + header_size;
        const auto header = nlohmann::json::parse(
            reinterpret_cast<const char *>(mapping_ + 8),
            reinterpret_cast<const char *>(mapping_ + data_offset_));
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            safetensor_info info;
            info.name = it.key();
            info.dtype = it.value().at("dtype").get<std::string>();
            info.shape = it.value().at("shape").get<std::vector<int64_t>>();
            const auto offsets = it.value().at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] ||
                data_offset_ + offsets[1] > size_) {
                throw std::runtime_error(path + ": invalid offsets for " + info.name);
            }
            info.begin = offsets[0];
            info.end = offsets[1];
            index_.emplace(info.name, tensors_.size());
            tensors_.push_back(std::move(info));
        }
    } catch (...) {
        release_mapping();
        throw;
    }
}

safetensors_file::~safetensors_file() {
    release_mapping();
}

void safetensors_file::release_mapping() noexcept {
#ifdef _WIN32
    if (mapping_) UnmapViewOfFile(mapping_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    if (mapping_) ::munmap(const_cast<uint8_t *>(mapping_), size_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    mapping_ = nullptr;
    size_ = 0;
}

const safetensor_info & safetensors_file::at(const std::string & name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error(path_ + ": missing tensor " + name);
    return tensors_[it->second];
}

const uint8_t * safetensors_file::data(const safetensor_info & info) const {
    return mapping_ + data_offset_ + info.begin;
}

weight_store::weight_store(ggml_backend_t backend, const std::string & path, weight_policy policy)
    : file_(std::make_unique<safetensors_file>(path)) {
    const size_t metadata_size = 4u * 1024u * 1024u +
        file_->tensors().size() * (ggml_tensor_overhead() + 256u);
    ggml_init_params params { metadata_size, nullptr, true };
    ctx_ = ggml_init(params);
    if (!ctx_) throw std::runtime_error("failed to allocate ggml weight metadata");

    for (const auto & info : file_->tensors()) {
        ggml_type type = native_type(info.dtype);
        if (policy == weight_policy::f16 && (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16)) {
            type = GGML_TYPE_F16;
        } else if (policy == weight_policy::f16 && type == GGML_TYPE_I64) {
            type = GGML_TYPE_I32;
        }
        std::vector<int64_t> dims(info.shape.rbegin(), info.shape.rend());
        if (dims.empty()) dims.push_back(1);
        ggml_tensor * tensor = ggml_new_tensor(ctx_, type, int(dims.size()), dims.data());
        ggml_set_name(tensor, info.name.c_str());
        tensors_.emplace(info.name, tensor);
        bytes_ += ggml_nbytes(tensor);
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) throw std::runtime_error("failed to allocate Vulkan weight buffer for " + path);
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto & info : file_->tensors()) {
        ggml_tensor * tensor = tensors_.at(info.name);
        if (tensor->type == GGML_TYPE_F16 && info.dtype != "F16") {
            const auto converted = convert_to_f16(*file_, info);
            ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(ggml_fp16_t));
        } else if (tensor->type == GGML_TYPE_I32 && info.dtype == "I64") {
            const auto converted = convert_to_i32(*file_, info);
            ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(int32_t));
        } else {
            const size_t source_bytes = size_t(info.end - info.begin);
            if (source_bytes != ggml_nbytes(tensor)) {
                throw std::runtime_error("byte size mismatch for " + info.name);
            }
            ggml_backend_tensor_set(tensor, file_->data(info), 0, source_bytes);
        }
    }
}

weight_store::~weight_store() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor * weight_store::get(const std::string & name) const {
    auto * tensor = maybe(name);
    if (!tensor) throw std::runtime_error("missing model weight: " + name);
    return tensor;
}

ggml_tensor * weight_store::maybe(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

} // namespace triposplat
