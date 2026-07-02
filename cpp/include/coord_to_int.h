#pragma once

#include <map>
#include <string>
#include <utility>

struct InterchangeDevice;

std::map<std::pair<uint32_t, uint32_t>, uint32_t> build_coord_to_int(
    const InterchangeDevice& device,
    const std::string& device_path);
