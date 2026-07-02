#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct InterchangeDevice;

struct TileRRG {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> capacities;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> distances;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> wl_scores;
};

TileRRG build_tile_rrg_from_path(const std::string& device_path, const InterchangeDevice& device);
