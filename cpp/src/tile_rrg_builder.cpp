#include "tile_rrg_builder.h"

#include "device_reader.h"

#include <DeviceResources.capnp.h>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

static bool allowed_distance(uint32_t dist) {
    return dist == 1 || dist == 2 || dist == 4 || dist == 12;
}

// FPGA24 contest PIP wirelength scores (wa.py / xcvup_device_data.py).
static uint32_t contest_wl_score(uint32_t dist, bool horizontal) {
    switch (dist) {
        case 1:
            return 1;
        case 2:
            return horizontal ? 5 : 3;
        case 4:
            return horizontal ? 10 : 5;
        case 12:
            return horizontal ? 14 : 12;
        default:
            return 0;
    }
}

static bool is_blocklisted_wire_name(const std::string& name) {
    static const char* blocked[] = {
        "GLOBAL", "GCLK", "BUFG", "REFCLK", "CLK", "VCC", "GND", "HROUTE", "HDISTR",
        "VDISTR", "VROUTE", "LEAF", nullptr};
    for (const char** p = blocked; *p; ++p) {
        if (name.find(*p) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::pair<uint32_t, uint32_t> undirected_key(uint32_t a, uint32_t b) {
    if (a <= b) {
        return {a, b};
    }
    return {b, a};
}

}  // namespace

TileRRG build_tile_rrg_from_path(
    const std::string& device_path,
    const InterchangeDevice& dev_meta) {
    const std::string raw = read_gzip_file(device_path);

    std::stringstream stream(raw, std::ios::in | std::ios::out | std::ios::binary);
    kj::std::StdInputStream istream(stream);

    capnp::ReaderOptions opts;
    opts.nestingLimit = std::numeric_limits<int>::max();
    opts.traversalLimitInWords = std::numeric_limits<uint64_t>::max();

    capnp::InputStreamMessageReader reader(istream, opts);
    auto device = reader.getRoot<DeviceResources::Device>();

    const auto& str_list = dev_meta.str_list;
    const auto tile_type_list = device.getTileTypeList();
    std::vector<std::string> tile_type_names(tile_type_list.size());
    for (uint32_t i = 0; i < tile_type_list.size(); ++i) {
        tile_type_names[i] = str_list[tile_type_list[i].getName()];
    }

    const auto wire_types = device.getWireTypes();
    std::vector<bool> wire_type_routable(wire_types.size(), false);
    for (uint32_t i = 0; i < wire_types.size(); ++i) {
        const auto wt = wire_types[i];
        if (wt.getCategory() != DeviceResources::Device::WireCategory::GENERAL) {
            continue;
        }
        const std::string name = str_list[wt.getName()];
        if (is_blocklisted_wire_name(name)) {
            continue;
        }
        wire_type_routable[i] = true;
    }

    const auto tile_list = device.getTileList();
    std::unordered_map<uint32_t, uint32_t> tile_str_to_idx;
    tile_str_to_idx.reserve(tile_list.size());
    std::vector<uint32_t> tile_rows(tile_list.size());
    std::vector<uint32_t> tile_cols(tile_list.size());
    std::vector<uint32_t> tile_type_idx(tile_list.size());

    for (uint32_t ti = 0; ti < tile_list.size(); ++ti) {
        const auto tile = tile_list[ti];
        tile_str_to_idx[tile.getName()] = ti;
        tile_rows[ti] = tile.getRow();
        tile_cols[ti] = tile.getCol();
        tile_type_idx[ti] = tile.getType();
    }

    const auto wire_list = device.getWires();
    const auto node_list = device.getNodes();

    TileRRG rrg;
    uint64_t skipped_category = 0;
    uint64_t skipped_geometry = 0;
    uint64_t skipped_same_tile = 0;
    uint64_t accepted = 0;

    for (const auto node : node_list) {
        const auto wires = node.getWires();
        if (wires.size() == 0) {
            continue;
        }

        const uint32_t begin_wire_idx = wires[0];
        if (begin_wire_idx >= wire_list.size()) {
            continue;
        }
        const auto begin_wire = wire_list[begin_wire_idx];
        const uint32_t begin_wt = begin_wire.getType();
        if (begin_wt >= wire_type_routable.size() || !wire_type_routable[begin_wt]) {
            ++skipped_category;
            continue;
        }

        int32_t begin_int = -1;
        int32_t end_int = -1;
        bool finding_end = true;

        for (const uint32_t wire_idx : wires) {
            if (wire_idx >= wire_list.size()) {
                continue;
            }
            const auto wire = wire_list[wire_idx];
            const auto it = tile_str_to_idx.find(wire.getTile());
            if (it == tile_str_to_idx.end()) {
                continue;
            }
            const uint32_t ti = it->second;
            if (tile_type_idx[ti] >= tile_type_names.size()) {
                continue;
            }
            if (tile_type_names[tile_type_idx[ti]] != "INT") {
                continue;
            }
            const uint64_t key = pack_coord(tile_rows[ti], tile_cols[ti]);
            const auto int_it = dev_meta.int_coord_to_idx.find(key);
            if (int_it == dev_meta.int_coord_to_idx.end()) {
                continue;
            }
            const int32_t int_idx = static_cast<int32_t>(int_it->second);

            if (finding_end) {
                if (begin_int < 0) {
                    begin_int = int_idx;
                }
                end_int = int_idx;
                if (begin_int >= 0 && end_int != begin_int) {
                    // Potter: stop after recording a second INT tile
                    finding_end = false;
                }
            }
        }

        if (begin_int < 0 || end_int < 0) {
            continue;
        }
        if (begin_int == end_int) {
            ++skipped_same_tile;
            continue;
        }

        const auto& ta = dev_meta.int_tiles[static_cast<uint32_t>(begin_int)];
        const auto& tb = dev_meta.int_tiles[static_cast<uint32_t>(end_int)];
        const uint32_t dy = (ta.fabric_y > tb.fabric_y) ? (ta.fabric_y - tb.fabric_y)
                                                        : (tb.fabric_y - ta.fabric_y);
        const uint32_t dx = (ta.fabric_x > tb.fabric_x) ? (ta.fabric_x - tb.fabric_x)
                                                        : (tb.fabric_x - ta.fabric_x);

        if (dy != 0 && dx != 0) {
            ++skipped_geometry;
            continue;
        }
        const uint32_t dist = dy + dx;
        if (!allowed_distance(dist)) {
            ++skipped_geometry;
            continue;
        }

        const auto key = undirected_key(static_cast<uint32_t>(begin_int), static_cast<uint32_t>(end_int));
        rrg.capacities[key]++;
        if (rrg.distances.find(key) == rrg.distances.end()) {
            const bool horizontal = (dy == 0);
            rrg.distances[key] = dist;
            rrg.wl_scores[key] = contest_wl_score(dist, horizontal);
        }
        ++accepted;
    }

    rrg.edges.clear();
    rrg.edges.reserve(rrg.capacities.size());
    for (const auto& kv : rrg.capacities) {
        rrg.edges.push_back(kv.first);
    }
    std::sort(rrg.edges.begin(), rrg.edges.end());

    std::cerr << "RRG edges: " << rrg.edges.size() << " (accepted nodes=" << accepted
              << " skipped_category=" << skipped_category
              << " skipped_same_tile=" << skipped_same_tile
              << " skipped_geometry=" << skipped_geometry << ")" << std::endl;

    return rrg;
}
