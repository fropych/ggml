#pragma once

#include "ggml-backend.h"

#include <string>

namespace triposplat {

bool is_e2e_worker_mode(const std::string & mode);
int run_e2e_worker(const std::string & mode, int argc, char ** argv,
                   ggml_backend_t backend);

} // namespace triposplat
