#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace triposplat {

struct artifact_store_config {
    std::filesystem::path directory = "artifacts";
    std::chrono::seconds ttl = std::chrono::hours(24);
    size_t max_artifacts = 100;
    uint64_t max_bytes = 50ull * 1024ull * 1024ull * 1024ull;
};

struct artifact_record {
    std::string id;
    std::string type;
    std::string state = "ready";
    std::string original_name;
    std::string producer_job_id;
    std::string parent_artifact_id;
    std::string file_name;
    uint64_t size_bytes = 0;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point expires_at;
    std::filesystem::path data_path;
};

struct artifact_cleanup_result {
    size_t removed = 0;
    size_t remaining = 0;
    uint64_t remaining_bytes = 0;
    bool limits_satisfied = true;
};

enum class artifact_remove_result {
    removed,
    not_found,
    in_use,
};

class artifact_lease {
public:
    artifact_lease() = default;

    explicit operator bool() const;
    const artifact_record & record() const;

private:
    friend class artifact_store;
    explicit artifact_lease(std::shared_ptr<const artifact_record> record);
    std::shared_ptr<const artifact_record> record_;
};

class artifact_store {
public:
    using clock_function =
        std::function<std::chrono::system_clock::time_point()>;

    explicit artifact_store(
        artifact_store_config config,
        clock_function clock = std::chrono::system_clock::now);
    ~artifact_store();

    const artifact_store_config & config() const;
    const std::filesystem::path & temporary_directory() const;

    std::filesystem::path make_temporary_file(
        const std::string & suffix = ".partial");
    std::filesystem::path make_temporary_directory(
        const std::string & label);

    artifact_record commit_file(
        const std::filesystem::path & temporary_path,
        const std::string & type,
        const std::string & original_name = {},
        const std::string & producer_job_id = {},
        const std::string & parent_artifact_id = {});

    std::optional<artifact_record> get(const std::string & id) const;
    artifact_lease acquire(const std::string & id);
    artifact_remove_result remove(const std::string & id);
    artifact_cleanup_result cleanup();

    size_t count() const;
    uint64_t total_bytes() const;

    static bool valid_type(const std::string & type);
    static std::string format_time(
        std::chrono::system_clock::time_point time);

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace triposplat
