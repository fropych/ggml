#include "artifact-store.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using triposplat::artifact_remove_result;
using triposplat::artifact_store;
using triposplat::artifact_store_config;

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct temporary_tree {
    fs::path path;

    temporary_tree() {
        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() /
            ("triposplat-artifact-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~temporary_tree() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

fs::path write_temporary(
    artifact_store & store,
    const std::string & contents) {
    const fs::path path = store.make_temporary_file();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), std::streamsize(contents.size()));
    output.close();
    check(bool(output), "failed to create test artifact");
    return path;
}

std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

void test_persistence_ttl_fifo_and_leases(const fs::path & root) {
    auto now = std::chrono::system_clock::from_time_t(1'700'000'000);
    artifact_store_config config;
    config.directory = root;
    config.ttl = std::chrono::seconds(10);
    config.max_artifacts = 2;
    config.max_bytes = 100;

    std::string newest_id;
    fs::path stale_temporary;
    {
        artifact_store store(config, [&] { return now; });
        const auto oldest = store.commit_file(
            write_temporary(store, "old"),
            "input_image", "../unsafe/input.webp");
        check(
            oldest.original_name == "input.webp",
            "original file name was not sanitized");

        now += std::chrono::seconds(1);
        const auto middle = store.commit_file(
            write_temporary(store, "middle"),
            "gaussian_ply", "middle.ply");
        auto oldest_lease = store.acquire(oldest.id);
        check(bool(oldest_lease), "could not acquire artifact lease");

        now += std::chrono::seconds(1);
        const auto newest = store.commit_file(
            write_temporary(store, "newest"),
            "tsvox", "newest.tsvox");
        newest_id = newest.id;

        check(store.count() == 2, "count quota was not enforced");
        check(!store.get(middle.id), "oldest unleased artifact was not evicted");
        check(bool(store.get(oldest.id)), "leased artifact was evicted");
        check(bool(store.get(newest.id)), "new artifact was not committed");
        check(
            store.remove(oldest.id) == artifact_remove_result::in_use,
            "leased artifact was deletable");

        oldest_lease = {};
        check(
            store.remove(oldest.id) == artifact_remove_result::removed,
            "released artifact was not deletable");

        stale_temporary = store.make_temporary_file(".stale");
        std::ofstream(stale_temporary, std::ios::binary) << "partial";
        check(fs::exists(stale_temporary), "stale temporary file missing");
    }

    {
        artifact_store store(config, [&] { return now; });
        check(store.count() == 1, "artifact manifest did not reload");
        const std::string manifest = read_file(
            root / newest_id / "manifest.json");
        const auto reloaded = store.get(newest_id);
        check(
            bool(reloaded),
            "reloaded artifact is missing at " +
                artifact_store::format_time(now) + " (remaining " +
                std::to_string(store.count()) + ", requested " +
                newest_id + "): " + manifest);
        check(
            !fs::exists(stale_temporary),
            "stale temporary entry survived restart");

        now += std::chrono::seconds(11);
        check(
            !store.get(newest_id),
            "expired artifact remained readable");
        check(store.count() == 0, "expired artifact was not removed");
        check(store.total_bytes() == 0, "expired bytes were not released");
    }
}

void test_byte_quota_and_full_store_rollback(const fs::path & root) {
    auto now = std::chrono::system_clock::from_time_t(1'700'100'000);
    artifact_store_config config;
    config.directory = root;
    config.ttl = std::chrono::hours(1);
    config.max_artifacts = 1;
    config.max_bytes = 5;

    artifact_store store(config, [&] { return now; });
    const auto first = store.commit_file(
        write_temporary(store, "12345"),
        "input_image", "first.webp");
    auto lease = store.acquire(first.id);
    check(bool(lease), "failed to lease quota fixture");

    bool rejected = false;
    try {
        store.commit_file(
            write_temporary(store, "x"),
            "input_image", "second.webp");
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "commit succeeded while every eviction target was leased");
    check(store.count() == 1, "failed commit changed artifact count");
    check(store.total_bytes() == 5, "failed commit changed byte count");
    check(bool(store.get(first.id)), "failed commit removed existing artifact");

    bool oversized = false;
    try {
        store.commit_file(
            write_temporary(store, "123456"),
            "input_image", "oversized.webp");
    } catch (const std::runtime_error &) {
        oversized = true;
    }
    check(oversized, "oversized artifact was accepted");
}

} // namespace

int main() {
    try {
        temporary_tree tree;
        test_persistence_ttl_fifo_and_leases(tree.path / "persistent");
        test_byte_quota_and_full_store_rollback(tree.path / "quota");
        std::cout << "artifact store contract passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "artifact store contract failed: "
                  << error.what() << '\n';
        return 1;
    }
}
