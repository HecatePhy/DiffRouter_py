#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "coord_to_int.h"
#include "device_reader.h"
#include "tile_rrg_builder.h"

void write_rrg_json(
    const std::string& output_path,
    const InterchangeDevice& device,
    const TileRRG& rrg,
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& coord_to_int);
