#include "voxel-converter.h"

#include "voxel_finalize.spv.h"
#include "voxel_integrate.spv.h"
#include "voxel_preprocess.spv.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace triposplat {
namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr uint32_t kLocalSize = 256;

void vk_require(VkResult result, const char * operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with Vulkan error " +
                                 std::to_string(int(result)));
    }
}

bool host_is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

struct gaussian_cpu {
    std::array<float, 3> mean{};
    std::array<float, 3> scale{};
    std::array<float, 4> rotation{};
    std::array<float, 3> color{};
    float opacity = 0.0f;
};

size_t ply_scalar_size(const std::string & type) {
    if (type == "char" || type == "int8" ||
        type == "uchar" || type == "uint8") {
        return 1;
    }
    if (type == "short" || type == "int16" ||
        type == "ushort" || type == "uint16") {
        return 2;
    }
    if (type == "int" || type == "int32" ||
        type == "uint" || type == "uint32" ||
        type == "float" || type == "float32") {
        return 4;
    }
    if (type == "double" || type == "float64") {
        return 8;
    }
    throw std::invalid_argument("unsupported PLY scalar type: " + type);
}

float load_f32(const uint8_t * data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

std::vector<gaussian_cpu> load_gaussian_ply(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open Gaussian PLY: " + path);
    }
    std::string line;
    if (!std::getline(stream, line) || line != "ply") {
        throw std::invalid_argument("input is not a PLY file: " + path);
    }

    struct property {
        std::string type;
        std::string name;
        size_t offset = 0;
    };
    std::vector<property> properties;
    std::string format;
    uint64_t vertex_count = 0;
    size_t record_size = 0;
    bool in_vertex = false;
    bool saw_vertex = false;
    bool ended = false;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("format ", 0) == 0) {
            const size_t start = 7;
            const size_t end = line.find(' ', start);
            format = line.substr(start, end - start);
        } else if (line.rfind("element ", 0) == 0) {
            const size_t name_start = 8;
            const size_t name_end = line.find(' ', name_start);
            if (name_end == std::string::npos) {
                throw std::invalid_argument("malformed PLY element line");
            }
            const std::string name =
                line.substr(name_start, name_end - name_start);
            in_vertex = name == "vertex";
            if (in_vertex) {
                if (saw_vertex) {
                    throw std::invalid_argument(
                        "PLY contains more than one vertex element");
                }
                vertex_count =
                    std::stoull(line.substr(name_end + 1));
                saw_vertex = true;
            }
        } else if (line.rfind("property ", 0) == 0 && in_vertex) {
            const size_t type_start = 9;
            const size_t type_end = line.find(' ', type_start);
            if (type_end == std::string::npos ||
                line.compare(type_start, 4, "list") == 0) {
                throw std::invalid_argument(
                    "vertex list properties are unsupported");
            }
            const std::string type =
                line.substr(type_start, type_end - type_start);
            const std::string name = line.substr(type_end + 1);
            properties.push_back({type, name, record_size});
            record_size += ply_scalar_size(type);
        } else if (line == "end_header") {
            ended = true;
            break;
        }
    }
    if (!ended || format != "binary_little_endian" ||
        !saw_vertex || properties.empty()) {
        throw std::invalid_argument(
            "expected a binary_little_endian PLY vertex table");
    }
    if (vertex_count == 0 ||
        vertex_count > std::numeric_limits<size_t>::max() / record_size) {
        throw std::invalid_argument("invalid PLY vertex count");
    }

    std::map<std::string, property> by_name;
    for (const property & item : properties) {
        by_name.emplace(item.name, item);
    }
    const std::array<const char *, 14> required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    };
    for (const char * name : required) {
        const auto iterator = by_name.find(name);
        if (iterator == by_name.end()) {
            throw std::invalid_argument(
                std::string("Gaussian PLY is missing property ") + name);
        }
        if (iterator->second.type != "float" &&
            iterator->second.type != "float32") {
            throw std::invalid_argument(
                std::string("Gaussian PLY property must be float32: ") + name);
        }
    }

    std::vector<uint8_t> bytes(size_t(vertex_count) * record_size);
    stream.read(reinterpret_cast<char *>(bytes.data()),
                std::streamsize(bytes.size()));
    if (stream.gcount() != std::streamsize(bytes.size())) {
        throw std::runtime_error("truncated Gaussian PLY vertex table");
    }
    auto field = [&](const uint8_t * record, const char * name) {
        return load_f32(record + by_name.at(name).offset);
    };

    std::vector<gaussian_cpu> gaussians(
        static_cast<size_t>(vertex_count));
    for (size_t index = 0; index < gaussians.size(); ++index) {
        const uint8_t * record = bytes.data() + index * record_size;
        gaussian_cpu & gaussian = gaussians[index];
        gaussian.mean = {
            field(record, "x"), field(record, "y"), field(record, "z")};
        gaussian.scale = {
            std::exp(field(record, "scale_0")),
            std::exp(field(record, "scale_1")),
            std::exp(field(record, "scale_2"))};
        gaussian.rotation = {
            field(record, "rot_0"), field(record, "rot_1"),
            field(record, "rot_2"), field(record, "rot_3")};
        gaussian.opacity = stable_sigmoid(field(record, "opacity"));
        gaussian.color = {
            std::clamp(0.5f + kShC0 * field(record, "f_dc_0"), 0.0f, 1.0f),
            std::clamp(0.5f + kShC0 * field(record, "f_dc_1"), 0.0f, 1.0f),
            std::clamp(0.5f + kShC0 * field(record, "f_dc_2"), 0.0f, 1.0f)};
        double norm_squared = 0.0;
        for (float value : gaussian.rotation) {
            norm_squared += double(value) * value;
        }
        const float norm = float(std::sqrt(norm_squared));
        if (!(norm > 1e-12f)) {
            throw std::invalid_argument(
                "Gaussian PLY contains a zero quaternion");
        }
        for (float & value : gaussian.rotation) value /= norm;
        for (float value : gaussian.mean) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Gaussian PLY contains non-finite means");
            }
        }
        for (float value : gaussian.scale) {
            if (!std::isfinite(value) || !(value > 0.0f)) {
                throw std::invalid_argument(
                    "Gaussian PLY contains invalid scales");
            }
        }
    }
    return gaussians;
}

struct alignas(16) raw_gaussian_gpu {
    float mean_opacity[4];
    float scale_color_r[4];
    float rotation[4];
    float color_gb[4];
};
static_assert(sizeof(raw_gaussian_gpu) == 64);

struct alignas(16) prepared_gaussian_gpu {
    float data[32];
    uint32_t bounds[8];
};
static_assert(sizeof(prepared_gaussian_gpu) == 160);

struct voxel_record_v1 {
    uint32_t linear_index;
    float color[3];
};
static_assert(sizeof(voxel_record_v1) == 16);

#pragma pack(push, 1)
struct voxel_file_header_v1 {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t resolution;
    uint32_t axis_order;
    uint32_t color_type;
    uint32_t record_bytes;
    uint64_t occupied_count;
    uint64_t record_count;
    float origin[3];
    float voxel_size;
    float iso;
    float opacity_threshold;
    float tolerance;
    float color_weight_power;
    uint32_t integration_steps;
    uint32_t flags;
    uint64_t source_gaussian_count;
    uint64_t payload_bytes;
    uint64_t reserved[3];
};
#pragma pack(pop)
static_assert(sizeof(voxel_file_header_v1) == 128);

struct push_constants {
    uint32_t gaussian_count;
    uint32_t total_pairs;
    uint32_t resolution;
    uint32_t integration_steps;
    float iso;
    float tolerance;
    float color_weight_power;
    float opacity_threshold;
};
static_assert(sizeof(push_constants) == 32);

class vulkan_context;

struct vk_buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    void * mapped = nullptr;
    bool coherent = false;
    uint64_t * current_bytes = nullptr;

    vk_buffer() = default;
    vk_buffer(const vk_buffer &) = delete;
    vk_buffer & operator=(const vk_buffer &) = delete;

    vk_buffer(vk_buffer && other) noexcept {
        *this = std::move(other);
    }
    vk_buffer & operator=(vk_buffer && other) noexcept {
        if (this == &other) return *this;
        release();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        memory = std::exchange(other.memory, VK_NULL_HANDLE);
        size = std::exchange(other.size, 0);
        allocation_size = std::exchange(other.allocation_size, 0);
        mapped = std::exchange(other.mapped, nullptr);
        coherent = other.coherent;
        current_bytes = std::exchange(other.current_bytes, nullptr);
        return *this;
    }
    ~vk_buffer() {
        release();
    }

    void release() {
        if (device == VK_NULL_HANDLE) return;
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        if (current_bytes) *current_bytes -= uint64_t(allocation_size);
        device = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        mapped = nullptr;
        size = 0;
        allocation_size = 0;
        current_bytes = nullptr;
    }
};

class vulkan_context {
public:
    explicit vulkan_context(int requested_device) {
        const VkApplicationInfo application{
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "triposplat voxel converter",
            VK_MAKE_VERSION(1, 0, 0),
            "triposplat",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2,
        };
        const VkInstanceCreateInfo instance_info{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &application,
            0,
            nullptr,
            0,
            nullptr,
        };
        vk_require(vkCreateInstance(&instance_info, nullptr, &instance_),
                   "vkCreateInstance");

        uint32_t physical_count = 0;
        vk_require(vkEnumeratePhysicalDevices(
                       instance_, &physical_count, nullptr),
                   "vkEnumeratePhysicalDevices");
        if (physical_count == 0) {
            throw std::runtime_error("no Vulkan physical device");
        }
        std::vector<VkPhysicalDevice> physical_devices(physical_count);
        vk_require(vkEnumeratePhysicalDevices(
                       instance_, &physical_count, physical_devices.data()),
                   "vkEnumeratePhysicalDevices");
        if (requested_device < 0 ||
            uint32_t(requested_device) >= physical_count) {
            throw std::invalid_argument(
                "Vulkan device index is out of range");
        }
        physical_ = physical_devices[size_t(requested_device)];
        vkGetPhysicalDeviceProperties(physical_, &properties_);
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties_);

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_, &family_count, families.data());
        queue_family_ = std::numeric_limits<uint32_t>::max();
        for (uint32_t index = 0; index < family_count; ++index) {
            if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                queue_family_ = index;
                break;
            }
        }
        if (queue_family_ == std::numeric_limits<uint32_t>::max()) {
            for (uint32_t index = 0; index < family_count; ++index) {
                if (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    queue_family_ = index;
                    break;
                }
            }
        }
        if (queue_family_ == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                "Vulkan device exposes no compute queue");
        }
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_info{
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            queue_family_,
            1,
            &priority,
        };
        const VkDeviceCreateInfo device_info{
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            1,
            &queue_info,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
        };
        vk_require(vkCreateDevice(
                       physical_, &device_info, nullptr, &device_),
                   "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        const VkCommandPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_,
        };
        vk_require(vkCreateCommandPool(
                       device_, &pool_info, nullptr, &command_pool_),
                   "vkCreateCommandPool");

        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        const VkDescriptorSetLayoutCreateInfo descriptor_layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            uint32_t(bindings.size()),
            bindings.data(),
        };
        vk_require(vkCreateDescriptorSetLayout(
                       device_, &descriptor_layout_info, nullptr,
                       &descriptor_layout_),
                   "vkCreateDescriptorSetLayout");
        const VkPushConstantRange push_range{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants)};
        const VkPipelineLayoutCreateInfo pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptor_layout_,
            1,
            &push_range,
        };
        vk_require(vkCreatePipelineLayout(
                       device_, &pipeline_layout_info, nullptr,
                       &pipeline_layout_),
                   "vkCreatePipelineLayout");

        preprocess_pipeline_ = create_pipeline(
            triposplat_voxel_preprocess_spv,
            triposplat_voxel_preprocess_spv_size);
        integrate_pipeline_ = create_pipeline(
            triposplat_voxel_integrate_spv,
            triposplat_voxel_integrate_spv_size);
        finalize_pipeline_ = create_pipeline(
            triposplat_voxel_finalize_spv,
            triposplat_voxel_finalize_spv_size);

        const VkDescriptorPoolSize pool_size{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
        const VkDescriptorPoolCreateInfo descriptor_pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            1,
            &pool_size,
        };
        vk_require(vkCreateDescriptorPool(
                       device_, &descriptor_pool_info, nullptr,
                       &descriptor_pool_),
                   "vkCreateDescriptorPool");
        const VkDescriptorSetAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool_,
            1,
            &descriptor_layout_,
        };
        vk_require(vkAllocateDescriptorSets(
                       device_, &allocate_info, &descriptor_set_),
                   "vkAllocateDescriptorSets");

        if (properties_.limits.timestampComputeAndGraphics) {
            const VkQueryPoolCreateInfo query_info{
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                nullptr,
                0,
                VK_QUERY_TYPE_TIMESTAMP,
                4,
                0,
            };
            vk_require(vkCreateQueryPool(
                           device_, &query_info, nullptr, &query_pool_),
                       "vkCreateQueryPool");
        }
    }

    vulkan_context(const vulkan_context &) = delete;
    vulkan_context & operator=(const vulkan_context &) = delete;

    ~vulkan_context() {
        if (device_) vkDeviceWaitIdle(device_);
        if (query_pool_) vkDestroyQueryPool(device_, query_pool_, nullptr);
        if (descriptor_pool_) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        }
        if (preprocess_pipeline_) {
            vkDestroyPipeline(device_, preprocess_pipeline_, nullptr);
        }
        if (integrate_pipeline_) {
            vkDestroyPipeline(device_, integrate_pipeline_, nullptr);
        }
        if (finalize_pipeline_) {
            vkDestroyPipeline(device_, finalize_pipeline_, nullptr);
        }
        if (pipeline_layout_) {
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        }
        if (descriptor_layout_) {
            vkDestroyDescriptorSetLayout(
                device_, descriptor_layout_, nullptr);
        }
        if (command_pool_) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        if (device_) vkDestroyDevice(device_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    const char * device_name() const {
        return properties_.deviceName;
    }

    VkPipeline preprocess_pipeline() const {
        return preprocess_pipeline_;
    }
    VkPipeline integrate_pipeline() const {
        return integrate_pipeline_;
    }
    VkPipeline finalize_pipeline() const {
        return finalize_pipeline_;
    }
    VkPipelineLayout pipeline_layout() const {
        return pipeline_layout_;
    }
    VkDescriptorSet descriptor_set() const {
        return descriptor_set_;
    }

    uint64_t peak_bytes() const {
        return peak_bytes_;
    }

    vk_buffer create_buffer(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0) {
        if (size == 0) size = 4;
        vk_buffer result;
        result.device = device_;
        result.size = size;
        result.current_bytes = &current_bytes_;
        const VkBufferCreateInfo buffer_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            size,
            usage,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
        };
        vk_require(vkCreateBuffer(
                       device_, &buffer_info, nullptr, &result.buffer),
                   "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            device_, result.buffer, &requirements);
        const uint32_t memory_type =
            find_memory_type(requirements.memoryTypeBits,
                             required, preferred);
        const VkMemoryAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            requirements.size,
            memory_type,
        };
        vk_require(vkAllocateMemory(
                       device_, &allocate_info, nullptr, &result.memory),
                   "vkAllocateMemory");
        vk_require(vkBindBufferMemory(
                       device_, result.buffer, result.memory, 0),
                   "vkBindBufferMemory");
        result.allocation_size = requirements.size;
        const VkMemoryPropertyFlags flags =
            memory_properties_.memoryTypes[memory_type].propertyFlags;
        result.coherent =
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            vk_require(vkMapMemory(
                           device_, result.memory, 0,
                           VK_WHOLE_SIZE, 0, &result.mapped),
                       "vkMapMemory");
        }
        current_bytes_ += uint64_t(requirements.size);
        peak_bytes_ = std::max(peak_bytes_, current_bytes_);
        return result;
    }

    void flush(const vk_buffer & buffer) const {
        if (!buffer.mapped || buffer.coherent) return;
        const VkMappedMemoryRange range{
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            nullptr,
            buffer.memory,
            0,
            VK_WHOLE_SIZE,
        };
        vk_require(vkFlushMappedMemoryRanges(
                       device_, 1, &range),
                   "vkFlushMappedMemoryRanges");
    }

    void invalidate(const vk_buffer & buffer) const {
        if (!buffer.mapped || buffer.coherent) return;
        const VkMappedMemoryRange range{
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            nullptr,
            buffer.memory,
            0,
            VK_WHOLE_SIZE,
        };
        vk_require(vkInvalidateMappedMemoryRanges(
                       device_, 1, &range),
                   "vkInvalidateMappedMemoryRanges");
    }

    void update_binding(uint32_t binding, const vk_buffer & buffer) {
        const VkDescriptorBufferInfo buffer_info{
            buffer.buffer, 0, buffer.size};
        const VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set_,
            binding,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &buffer_info,
            nullptr,
        };
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkCommandBuffer begin_commands() const {
        VkCommandBuffer command = VK_NULL_HANDLE;
        const VkCommandBufferAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            command_pool_,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1,
        };
        vk_require(vkAllocateCommandBuffers(
                       device_, &allocate_info, &command),
                   "vkAllocateCommandBuffers");
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        vk_require(vkBeginCommandBuffer(command, &begin_info),
                   "vkBeginCommandBuffer");
        return command;
    }

    void submit_and_wait(VkCommandBuffer command) const {
        vk_require(vkEndCommandBuffer(command), "vkEndCommandBuffer");
        const VkSubmitInfo submit{
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &command,
            0,
            nullptr,
        };
        vk_require(vkQueueSubmit(
                       queue_, 1, &submit, VK_NULL_HANDLE),
                   "vkQueueSubmit");
        vk_require(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
    }

    void copy_buffer(const vk_buffer & source,
                     const vk_buffer & destination,
                     VkDeviceSize bytes) const {
        VkCommandBuffer command = begin_commands();
        const VkBufferCopy copy{0, 0, bytes};
        vkCmdCopyBuffer(command, source.buffer,
                        destination.buffer, 1, &copy);
        submit_and_wait(command);
    }

    void begin_timestamp(VkCommandBuffer command, uint32_t query) const {
        if (!query_pool_) return;
        vkCmdResetQueryPool(command, query_pool_, query, 2);
        vkCmdWriteTimestamp(command,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            query_pool_, query);
    }

    void end_timestamp(VkCommandBuffer command, uint32_t query) const {
        if (!query_pool_) return;
        vkCmdWriteTimestamp(command,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            query_pool_, query + 1);
    }

    double timestamp_milliseconds(uint32_t query) const {
        if (!query_pool_) return 0.0;
        std::array<uint64_t, 2> values{};
        vk_require(vkGetQueryPoolResults(
                       device_, query_pool_, query, 2,
                       sizeof(values), values.data(),
                       sizeof(uint64_t),
                       VK_QUERY_RESULT_64_BIT |
                           VK_QUERY_RESULT_WAIT_BIT),
                   "vkGetQueryPoolResults");
        return double(values[1] - values[0]) *
               double(properties_.limits.timestampPeriod) / 1.0e6;
    }

    uint32_t max_dispatch_x() const {
        return properties_.limits.maxComputeWorkGroupCount[0];
    }

private:
    uint32_t find_memory_type(
        uint32_t allowed, VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred) const {
        auto find = [&](VkMemoryPropertyFlags desired)
            -> uint32_t {
            for (uint32_t index = 0;
                 index < memory_properties_.memoryTypeCount; ++index) {
                if ((allowed & (1u << index)) == 0) continue;
                const VkMemoryPropertyFlags flags =
                    memory_properties_.memoryTypes[index].propertyFlags;
                if ((flags & desired) == desired) return index;
            }
            return std::numeric_limits<uint32_t>::max();
        };
        uint32_t result = find(required | preferred);
        if (result == std::numeric_limits<uint32_t>::max()) {
            result = find(required);
        }
        if (result == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                "no compatible Vulkan memory type");
        }
        return result;
    }

    VkPipeline create_pipeline(
        const uint8_t * code, size_t bytes) const {
        if (bytes == 0 || bytes % sizeof(uint32_t) != 0) {
            throw std::runtime_error("invalid embedded SPIR-V");
        }
        const VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            bytes,
            reinterpret_cast<const uint32_t *>(code),
        };
        VkShaderModule shader = VK_NULL_HANDLE;
        vk_require(vkCreateShaderModule(
                       device_, &shader_info, nullptr, &shader),
                   "vkCreateShaderModule");
        const VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            shader,
            "main",
            nullptr,
        };
        const VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            stage,
            pipeline_layout_,
            VK_NULL_HANDLE,
            -1,
        };
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult status = vkCreateComputePipelines(
            device_, VK_NULL_HANDLE, 1, &pipeline_info,
            nullptr, &pipeline);
        vkDestroyShaderModule(device_, shader, nullptr);
        vk_require(status, "vkCreateComputePipelines");
        return pipeline;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline preprocess_pipeline_ = VK_NULL_HANDLE;
    VkPipeline integrate_pipeline_ = VK_NULL_HANDLE;
    VkPipeline finalize_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    uint64_t current_bytes_ = 0;
    uint64_t peak_bytes_ = 0;
};

uint32_t dispatch_groups(uint64_t work_items,
                         const vulkan_context & context) {
    const uint64_t groups =
        (work_items + kLocalSize - 1) / kLocalSize;
    if (groups > context.max_dispatch_x()) {
        throw std::overflow_error(
            "Vulkan dispatch exceeds maxComputeWorkGroupCount.x");
    }
    return uint32_t(groups);
}

void bind_compute(VkCommandBuffer command,
                  const vulkan_context & context,
                  VkPipeline pipeline,
                  const push_constants & constants) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline);
    const VkDescriptorSet set = context.descriptor_set();
    vkCmdBindDescriptorSets(
        command, VK_PIPELINE_BIND_POINT_COMPUTE,
        context.pipeline_layout(), 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(
        command, context.pipeline_layout(),
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(constants), &constants);
}

void shader_barrier(VkCommandBuffer command) {
    const VkMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(
        command,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void write_voxel_file(
    const voxel_conversion_options & options,
    const std::array<float, 3> & origin,
    float voxel_size, uint64_t gaussian_count,
    const std::vector<voxel_record_v1> & records) {
    voxel_file_header_v1 header{};
    const std::array<char, 8> magic{
        'T', 'S', 'V', 'O', 'X', 'E', 'L', '\0'};
    std::copy(magic.begin(), magic.end(), header.magic);
    header.version = 1;
    header.header_bytes = sizeof(header);
    header.resolution = options.resolution;
    header.axis_order = 0; // occupancy[z,y,x], x-fastest linear index
    header.color_type = 1; // RGB float32
    header.record_bytes = sizeof(voxel_record_v1);
    header.occupied_count = records.size();
    header.record_count = records.size();
    std::copy(origin.begin(), origin.end(), header.origin);
    header.voxel_size = voxel_size;
    header.iso = options.iso;
    header.opacity_threshold = options.opacity_threshold;
    header.tolerance = options.tolerance;
    header.color_weight_power = options.color_weight_power;
    header.integration_steps = options.integration_steps;
    header.flags = 1u | 2u; // little-endian, unordered sparse records
    header.source_gaussian_count = gaussian_count;
    header.payload_bytes =
        uint64_t(records.size()) * sizeof(voxel_record_v1);

    const std::filesystem::path path(options.output_path);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open voxel output: " + options.output_path);
    }
    stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!records.empty()) {
        stream.write(
            reinterpret_cast<const char *>(records.data()),
            std::streamsize(records.size() * sizeof(voxel_record_v1)));
    }
    if (!stream) {
        throw std::runtime_error(
            "failed writing voxel output: " + options.output_path);
    }
}

} // namespace

voxel_conversion_result convert_gaussian_ply_to_voxels(
    const voxel_conversion_options & options) {
    if (!host_is_little_endian()) {
        throw std::runtime_error(
            "TSVOXEL v1 currently requires a little-endian host");
    }
    if (options.input_ply.empty() || options.output_path.empty()) {
        throw std::invalid_argument(
            "voxel conversion requires input and output paths");
    }
    if (options.resolution < 2 ||
        (options.resolution & (options.resolution - 1)) != 0) {
        throw std::invalid_argument(
            "voxel resolution must be a power of two");
    }
    if (!(options.iso > 0.0f) ||
        !(options.opacity_threshold >= 0.0f &&
          options.opacity_threshold <= 1.0f) ||
        !(options.tolerance > 0.0f) ||
        options.integration_steps == 0 ||
        !(options.color_weight_power > 0.0f)) {
        throw std::invalid_argument(
            "invalid voxel conversion parameters");
    }

    const std::vector<gaussian_cpu> gaussians =
        load_gaussian_ply(options.input_ply);
    if (gaussians.size() >
        std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("too many Gaussians for Vulkan");
    }
    std::array<float, 3> minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    std::array<float, 3> maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (const gaussian_cpu & gaussian : gaussians) {
        for (size_t axis = 0; axis < 3; ++axis) {
            minimum[axis] =
                std::min(minimum[axis], gaussian.mean[axis]);
            maximum[axis] =
                std::max(maximum[axis], gaussian.mean[axis]);
        }
    }
    const float largest_span = std::max({
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2]});
    if (!(largest_span > 0.0f)) {
        throw std::invalid_argument(
            "Gaussian means have zero spatial extent");
    }
    const float voxel_size =
        largest_span / float(options.resolution - 1);
    std::array<float, 3> origin{};
    for (size_t axis = 0; axis < 3; ++axis) {
        origin[axis] = minimum[axis] - 0.5f * voxel_size;
    }
    const float world_extent =
        float(options.resolution) * voxel_size;
    const float world_to_spc = 2.0f / world_extent;

    std::vector<raw_gaussian_gpu> raw(gaussians.size());
    for (size_t index = 0; index < gaussians.size(); ++index) {
        const gaussian_cpu & source = gaussians[index];
        raw_gaussian_gpu & destination = raw[index];
        for (size_t axis = 0; axis < 3; ++axis) {
            destination.mean_opacity[axis] =
                (source.mean[axis] - origin[axis]) *
                    world_to_spc -
                1.0f;
            destination.scale_color_r[axis] =
                source.scale[axis] * world_to_spc;
        }
        destination.mean_opacity[3] = source.opacity;
        destination.scale_color_r[3] = source.color[0];
        std::copy(source.rotation.begin(), source.rotation.end(),
                  destination.rotation);
        destination.color_gb[0] = source.color[1];
        destination.color_gb[1] = source.color[2];
        destination.color_gb[2] = 0.0f;
        destination.color_gb[3] = 0.0f;
    }

    const auto setup_begin = std::chrono::steady_clock::now();
    vulkan_context context(options.vulkan_device);
    const auto setup_end = std::chrono::steady_clock::now();
    const auto conversion_begin = setup_end;

    const VkMemoryPropertyFlags host_memory =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags coherent =
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags device_memory =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkBufferUsageFlags storage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags transfer_source =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkBufferUsageFlags transfer_destination =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    vk_buffer raw_buffer = context.create_buffer(
        raw.size() * sizeof(raw_gaussian_gpu), storage,
        host_memory, coherent);
    std::memcpy(raw_buffer.mapped, raw.data(),
                raw.size() * sizeof(raw_gaussian_gpu));
    context.flush(raw_buffer);
    vk_buffer prepared_buffer = context.create_buffer(
        raw.size() * sizeof(prepared_gaussian_gpu),
        storage, device_memory);
    vk_buffer count_buffer = context.create_buffer(
        raw.size() * sizeof(uint32_t), storage,
        host_memory, coherent);
    context.update_binding(0, raw_buffer);
    context.update_binding(1, prepared_buffer);
    context.update_binding(2, count_buffer);

    push_constants constants{
        uint32_t(raw.size()),
        0,
        options.resolution,
        options.integration_steps,
        options.iso,
        options.tolerance,
        options.color_weight_power,
        options.opacity_threshold,
    };
    VkCommandBuffer preprocess = context.begin_commands();
    context.begin_timestamp(preprocess, 0);
    bind_compute(preprocess, context,
                 context.preprocess_pipeline(), constants);
    vkCmdDispatch(preprocess,
                  dispatch_groups(raw.size(), context), 1, 1);
    context.end_timestamp(preprocess, 0);
    context.submit_and_wait(preprocess);
    context.invalidate(count_buffer);

    const uint32_t * counts =
        static_cast<const uint32_t *>(count_buffer.mapped);
    std::vector<uint32_t> offsets(raw.size() + 1, 0);
    uint64_t total_pairs_64 = 0;
    for (size_t index = 0; index < raw.size(); ++index) {
        total_pairs_64 += counts[index];
        if (total_pairs_64 >
            std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Gaussian AABB pair count exceeds uint32");
        }
        offsets[index + 1] = uint32_t(total_pairs_64);
    }
    constants.total_pairs = uint32_t(total_pairs_64);

    vk_buffer offset_staging = context.create_buffer(
        offsets.size() * sizeof(uint32_t), transfer_source,
        host_memory, coherent);
    std::memcpy(offset_staging.mapped, offsets.data(),
                offsets.size() * sizeof(uint32_t));
    context.flush(offset_staging);
    vk_buffer offset_buffer = context.create_buffer(
        offsets.size() * sizeof(uint32_t),
        storage | transfer_destination, device_memory);
    context.copy_buffer(offset_staging, offset_buffer,
                        offsets.size() * sizeof(uint32_t));

    const uint64_t voxel_count_64 =
        uint64_t(options.resolution) *
        options.resolution * options.resolution;
    if (voxel_count_64 > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(
            "voxel grid exceeds uint32 linear indexing");
    }
    const uint32_t voxel_count = uint32_t(voxel_count_64);
    vk_buffer accumulation_buffer = context.create_buffer(
        uint64_t(voxel_count) * 5 * sizeof(uint32_t),
        storage | transfer_destination, device_memory);
    vk_buffer record_buffer = context.create_buffer(
        uint64_t(voxel_count) * sizeof(voxel_record_v1),
        storage | transfer_source, device_memory);
    vk_buffer counter_buffer = context.create_buffer(
        sizeof(uint32_t), storage | transfer_destination,
        host_memory, coherent);
    context.update_binding(2, offset_buffer);
    context.update_binding(3, accumulation_buffer);
    context.update_binding(4, record_buffer);
    context.update_binding(5, counter_buffer);

    VkCommandBuffer integrate = context.begin_commands();
    vkCmdFillBuffer(integrate, accumulation_buffer.buffer,
                    0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(integrate, counter_buffer.buffer,
                    0, sizeof(uint32_t), 0);
    const VkMemoryBarrier transfer_to_compute{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(
        integrate, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &transfer_to_compute,
        0, nullptr, 0, nullptr);
    context.begin_timestamp(integrate, 2);
    bind_compute(integrate, context,
                 context.integrate_pipeline(), constants);
    if (constants.total_pairs > 0) {
        vkCmdDispatch(
            integrate,
            dispatch_groups(constants.total_pairs, context), 1, 1);
    }
    shader_barrier(integrate);
    bind_compute(integrate, context,
                 context.finalize_pipeline(), constants);
    vkCmdDispatch(integrate,
                  dispatch_groups(voxel_count, context), 1, 1);
    context.end_timestamp(integrate, 2);
    const VkMemoryBarrier compute_to_host{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(
        integrate, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &compute_to_host,
        0, nullptr, 0, nullptr);
    context.submit_and_wait(integrate);
    context.invalidate(counter_buffer);
    const uint32_t occupied =
        *static_cast<const uint32_t *>(counter_buffer.mapped);
    if (occupied > voxel_count) {
        throw std::runtime_error(
            "Vulkan voxel counter exceeded grid size");
    }

    vk_buffer readback_buffer = context.create_buffer(
        uint64_t(std::max(occupied, 1u)) *
            sizeof(voxel_record_v1),
        transfer_destination, host_memory, coherent);
    if (occupied > 0) {
        context.copy_buffer(
            record_buffer, readback_buffer,
            uint64_t(occupied) * sizeof(voxel_record_v1));
        context.invalidate(readback_buffer);
    }
    std::vector<voxel_record_v1> records(occupied);
    if (occupied > 0) {
        std::memcpy(records.data(), readback_buffer.mapped,
                    records.size() * sizeof(voxel_record_v1));
    }
    for (const voxel_record_v1 & record : records) {
        if (record.linear_index >= voxel_count ||
            !std::isfinite(record.color[0]) ||
            !std::isfinite(record.color[1]) ||
            !std::isfinite(record.color[2])) {
            throw std::runtime_error(
                "Vulkan produced an invalid voxel record");
        }
    }
    const auto conversion_end = std::chrono::steady_clock::now();
    write_voxel_file(
        options, origin, voxel_size,
        gaussians.size(), records);

    voxel_conversion_result result;
    result.device_name = context.device_name();
    result.gaussian_count = gaussians.size();
    result.aabb_candidate_pairs = total_pairs_64;
    result.occupied_voxels = occupied;
    result.output_bytes =
        sizeof(voxel_file_header_v1) +
        uint64_t(records.size()) * sizeof(voxel_record_v1);
    result.converter_gpu_bytes = context.peak_bytes();
    result.setup_milliseconds =
        std::chrono::duration<double, std::milli>(
            setup_end - setup_begin).count();
    result.conversion_milliseconds =
        std::chrono::duration<double, std::milli>(
            conversion_end - conversion_begin).count();
    result.gpu_milliseconds =
        context.timestamp_milliseconds(0) +
        context.timestamp_milliseconds(2);
    return result;
}

} // namespace triposplat
