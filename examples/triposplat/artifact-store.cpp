#include "artifact-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

namespace triposplat {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using system_clock = std::chrono::system_clock;

constexpr int k_manifest_schema = 1;

std::string random_hex() {
    static std::mutex mutex;
    static std::random_device random;
    static std::mt19937_64 generator(
        (uint64_t(random()) << 32) ^ uint64_t(random()));
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(16) << generator()
           << std::setw(16) << generator();
    return output.str();
}

std::string make_id() {
    return "art_" + random_hex();
}

std::string sanitize_original_name(const std::string & name) {
    if (name.empty()) return {};
    return fs::path(name).filename().string();
}

std::string extension_for_type(const std::string & type) {
    if (type == "input_image") return ".image";
    if (type == "gaussian_ply") return ".ply";
    if (type == "splat") return ".splat";
    if (type == "tsvox") return ".tsvox";
    if (type == "log") return ".log";
    throw std::invalid_argument("invalid artifact type: " + type);
}

system_clock::time_point parse_time(const std::string & value) {
    std::tm tm{};
    std::istringstream input(value);
    input >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    if (!input ||
        input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid artifact timestamp: " + value);
    }
#ifdef _WIN32
    const std::time_t seconds = _mkgmtime(&tm);
#else
    const std::time_t seconds = timegm(&tm);
#endif
    if (seconds == std::time_t(-1)) {
        throw std::runtime_error("artifact timestamp is out of range");
    }
    return system_clock::from_time_t(seconds);
}

json record_json(const artifact_record & record) {
    return {
        {"schema_version", k_manifest_schema},
        {"id", record.id},
        {"type", record.type},
        {"state", record.state},
        {"size_bytes", record.size_bytes},
        {"created_at", artifact_store::format_time(record.created_at)},
        {"expires_at", artifact_store::format_time(record.expires_at)},
        {"original_name", record.original_name},
        {"producer_job_id", record.producer_job_id},
        {"parent_artifact_id", record.parent_artifact_id},
        {"file_name", record.file_name},
    };
}

artifact_record parse_record(
    const json & value, const fs::path & directory) {
    if (value.at("schema_version").get<int>() != k_manifest_schema) {
        throw std::runtime_error("unsupported artifact manifest schema");
    }
    artifact_record record;
    record.id = value.at("id").get<std::string>();
    record.type = value.at("type").get<std::string>();
    record.state = value.at("state").get<std::string>();
    record.size_bytes = value.at("size_bytes").get<uint64_t>();
    record.created_at = parse_time(value.at("created_at").get<std::string>());
    record.expires_at = parse_time(value.at("expires_at").get<std::string>());
    record.original_name = value.value("original_name", std::string{});
    record.producer_job_id =
        value.value("producer_job_id", std::string{});
    record.parent_artifact_id =
        value.value("parent_artifact_id", std::string{});
    record.file_name = value.at("file_name").get<std::string>();
    record.data_path = directory / record.file_name;

    if (record.id != directory.filename().string() ||
        !artifact_store::valid_type(record.type) ||
        record.state != "ready" ||
        fs::path(record.file_name).filename() != record.file_name ||
        !fs::is_regular_file(record.data_path) ||
        fs::file_size(record.data_path) != record.size_bytes) {
        throw std::runtime_error(
            "invalid artifact manifest in " + directory.string());
    }
    return record;
}

void write_manifest(
    const fs::path & path, const artifact_record & record) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to create artifact manifest: " + path.string());
    }
    output << record_json(record).dump(2) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed to write artifact manifest: " + path.string());
    }
}

} // namespace

struct artifact_store::implementation {
    artifact_store_config config;
    clock_function clock;
    fs::path root;
    fs::path temporary;
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<artifact_record>> records;
    uint64_t bytes = 0;

    implementation(
        artifact_store_config input_config,
        clock_function input_clock)
        : config(std::move(input_config)),
          clock(std::move(input_clock)) {
        if (config.directory.empty()) {
            throw std::invalid_argument("artifact directory is empty");
        }
        if (config.ttl <= std::chrono::seconds::zero()) {
            throw std::invalid_argument("artifact TTL must be positive");
        }
        if (config.max_artifacts == 0) {
            throw std::invalid_argument("max-artifacts must be positive");
        }
        if (config.max_bytes == 0) {
            throw std::invalid_argument(
                "max-artifact-bytes must be positive");
        }
        if (!clock) throw std::invalid_argument("artifact clock is empty");

        std::error_code error;
        root = fs::absolute(config.directory, error);
        if (error) {
            throw std::runtime_error(
                "failed to resolve artifact directory: " +
                error.message());
        }
        config.directory = root;
        temporary = root / ".tmp";
        fs::create_directories(temporary);

        for (const fs::directory_entry & entry :
             fs::directory_iterator(temporary)) {
            fs::remove_all(entry.path());
        }

        for (const fs::directory_entry & entry :
             fs::directory_iterator(root)) {
            if (!entry.is_directory() ||
                entry.path().filename() == ".tmp") {
                continue;
            }
            const fs::path manifest = entry.path() / "manifest.json";
            std::ifstream input(manifest, std::ios::binary);
            if (!input) {
                throw std::runtime_error(
                    "artifact directory has no manifest: " +
                    entry.path().string());
            }
            json value;
            input >> value;
            artifact_record record = parse_record(value, entry.path());
            if (records.count(record.id) != 0) {
                throw std::runtime_error(
                    "duplicate artifact id: " + record.id);
            }
            if (record.size_bytes >
                std::numeric_limits<uint64_t>::max() - bytes) {
                throw std::overflow_error(
                    "artifact store size overflow");
            }
            bytes += record.size_bytes;
            const std::string id = record.id;
            records.emplace(
                id,
                std::make_shared<artifact_record>(std::move(record)));
        }
        cleanup_locked();
    }

    artifact_cleanup_result cleanup_locked(
        const std::string & protected_id = {}) {
        artifact_cleanup_result result;
        const auto now = clock();
        std::vector<std::shared_ptr<artifact_record>> candidates;
        candidates.reserve(records.size());
        for (const auto & item : records) {
            if (item.first != protected_id &&
                item.second.use_count() == 1) {
                candidates.push_back(item.second);
            }
        }
        std::sort(
            candidates.begin(), candidates.end(),
            [](const auto & left, const auto & right) {
                if (left->created_at != right->created_at) {
                    return left->created_at < right->created_at;
                }
                return left->id < right->id;
            });

        auto erase_record = [&](const std::shared_ptr<artifact_record> & item) {
            const auto found = records.find(item->id);
            if (found == records.end() ||
                found->second.get() != item.get() ||
                found->second.use_count() != 2) {
                return false;
            }
            const fs::path directory = root / item->id;
            fs::remove_all(directory);
            bytes -= item->size_bytes;
            records.erase(found);
            ++result.removed;
            return true;
        };

        for (const auto & item : candidates) {
            if (item->expires_at <= now) erase_record(item);
        }
        for (const auto & item : candidates) {
            if (records.size() <= config.max_artifacts &&
                bytes <= config.max_bytes) {
                break;
            }
            erase_record(item);
        }
        result.remaining = records.size();
        result.remaining_bytes = bytes;
        result.limits_satisfied =
            records.size() <= config.max_artifacts &&
            bytes <= config.max_bytes;
        return result;
    }

    fs::path unique_temporary_path(
        const std::string & prefix,
        const std::string & suffix,
        bool directory) {
        for (int attempt = 0; attempt < 100; ++attempt) {
            const fs::path path =
                temporary / (prefix + random_hex() + suffix);
            std::error_code error;
            if (directory) {
                if (fs::create_directory(path, error)) return path;
            } else {
                if (!fs::exists(path, error)) return path;
            }
            if (error) {
                throw std::runtime_error(
                    "failed to allocate artifact temporary path: " +
                    error.message());
            }
        }
        throw std::runtime_error(
            "failed to allocate a unique artifact temporary path");
    }
};

artifact_store::~artifact_store() = default;

artifact_lease::artifact_lease(
    std::shared_ptr<const artifact_record> record)
    : record_(std::move(record)) {}

artifact_lease::operator bool() const {
    return bool(record_);
}

const artifact_record & artifact_lease::record() const {
    if (!record_) throw std::logic_error("artifact lease is empty");
    return *record_;
}

artifact_store::artifact_store(
    artifact_store_config config, clock_function clock)
    : impl_(std::make_unique<implementation>(
          std::move(config), std::move(clock))) {}

const artifact_store_config & artifact_store::config() const {
    return impl_->config;
}

const fs::path & artifact_store::temporary_directory() const {
    return impl_->temporary;
}

fs::path artifact_store::make_temporary_file(
    const std::string & suffix) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->unique_temporary_path("upload_", suffix, false);
}

fs::path artifact_store::make_temporary_directory(
    const std::string & label) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string prefix = "job_";
    for (char character : label) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-') {
            prefix.push_back(character);
        }
    }
    prefix.push_back('_');
    return impl_->unique_temporary_path(prefix, ".partial", true);
}

artifact_record artifact_store::commit_file(
    const fs::path & temporary_path,
    const std::string & type,
    const std::string & original_name,
    const std::string & producer_job_id,
    const std::string & parent_artifact_id) {
    if (!valid_type(type)) {
        throw std::invalid_argument("invalid artifact type: " + type);
    }
    if (!fs::is_regular_file(temporary_path)) {
        throw std::invalid_argument(
            "artifact temporary file does not exist: " +
            temporary_path.string());
    }
    const uint64_t size = fs::file_size(temporary_path);
    if (size > impl_->config.max_bytes) {
        throw std::runtime_error(
            "artifact exceeds max-artifact-bytes");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string id;
    fs::path staging;
    fs::path final_directory;
    for (int attempt = 0; attempt < 100; ++attempt) {
        id = make_id();
        final_directory = impl_->root / id;
        if (impl_->records.count(id) == 0 &&
            !fs::exists(final_directory)) {
            staging = impl_->temporary / (id + ".partial");
            if (fs::create_directory(staging)) break;
        }
        id.clear();
    }
    if (id.empty()) {
        throw std::runtime_error(
            "failed to allocate a unique artifact id");
    }

    artifact_record record;
    record.id = id;
    record.type = type;
    record.original_name = sanitize_original_name(original_name);
    record.producer_job_id = producer_job_id;
    record.parent_artifact_id = parent_artifact_id;
    record.file_name = "data" + extension_for_type(type);
    record.size_bytes = size;
    record.created_at = impl_->clock();
    record.expires_at = record.created_at + impl_->config.ttl;

    try {
        const fs::path staged_data = staging / record.file_name;
        std::error_code error;
        fs::rename(temporary_path, staged_data, error);
        if (error) {
            fs::copy_file(
                temporary_path, staged_data,
                fs::copy_options::overwrite_existing);
            fs::remove(temporary_path);
        }
        write_manifest(staging / "manifest.json", record);
        fs::rename(staging, final_directory);
        record.data_path = final_directory / record.file_name;
        auto stored =
            std::make_shared<artifact_record>(record);
        impl_->records.emplace(record.id, stored);
        impl_->bytes += record.size_bytes;
        const artifact_cleanup_result cleanup =
            impl_->cleanup_locked(record.id);
        if (!cleanup.limits_satisfied) {
            impl_->records.erase(record.id);
            impl_->bytes -= record.size_bytes;
            fs::remove_all(final_directory);
            throw std::runtime_error(
                "artifact store is full while existing artifacts are in use");
        }
        return record;
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
}

std::optional<artifact_record> artifact_store::get(
    const std::string & id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->cleanup_locked();
    const auto found = impl_->records.find(id);
    if (found == impl_->records.end()) return std::nullopt;
    return *found->second;
}

artifact_lease artifact_store::acquire(const std::string & id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->cleanup_locked();
    const auto found = impl_->records.find(id);
    if (found == impl_->records.end()) return {};
    return artifact_lease(found->second);
}

artifact_remove_result artifact_store::remove(
    const std::string & id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->records.find(id);
    if (found == impl_->records.end()) {
        return artifact_remove_result::not_found;
    }
    if (found->second.use_count() != 1) {
        return artifact_remove_result::in_use;
    }
    const uint64_t size = found->second->size_bytes;
    fs::remove_all(impl_->root / id);
    impl_->records.erase(found);
    impl_->bytes -= size;
    return artifact_remove_result::removed;
}

artifact_cleanup_result artifact_store::cleanup() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->cleanup_locked();
}

size_t artifact_store::count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->records.size();
}

uint64_t artifact_store::total_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->bytes;
}

bool artifact_store::valid_type(const std::string & type) {
    static constexpr std::array<const char *, 5> types = {
        "input_image", "gaussian_ply", "splat", "tsvox", "log"};
    return std::find(types.begin(), types.end(), type) != types.end();
}

std::string artifact_store::format_time(
    system_clock::time_point time) {
    const std::time_t seconds = system_clock::to_time_t(time);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &seconds);
#else
    gmtime_r(&seconds, &tm);
#endif
    std::ostringstream output;
    output << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

} // namespace triposplat
