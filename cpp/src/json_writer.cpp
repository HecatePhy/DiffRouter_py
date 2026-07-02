#include "json_writer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

}  // namespace

void write_rrg_json(
    const std::string& output_path,
    const InterchangeDevice& device,
    const TileRRG& rrg,
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& coord_to_int) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open output: " + output_path);
    }

    out << "{\n";
    out << "  \"device_name\": \"" << json_escape(device.device_name) << "\",\n";
    out << "  \"device_rows\": " << device.device_rows << ",\n";
    out << "  \"device_cols\": " << device.device_cols << ",\n";
    out << "  \"int_only\": true,\n";

    out << "  \"tiles\": [\n";
    for (uint32_t i = 0; i < device.int_tiles.size(); ++i) {
        const auto& t = device.int_tiles[i];
        // Fabric lattice Y/X (from INT_X#Y# name); used for routing geometry.
        out << "    [" << t.fabric_y << ", " << t.fabric_x << ", \"" << json_escape(t.name)
            << "\", true]";
        if (i + 1 < device.int_tiles.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"int_interchange\": [\n";
    for (uint32_t i = 0; i < device.int_tiles.size(); ++i) {
        const auto& t = device.int_tiles[i];
        out << "    [" << t.row << ", " << t.col << "]";
        if (i + 1 < device.int_tiles.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"edges\": [\n";
    for (size_t i = 0; i < rrg.edges.size(); ++i) {
        const auto& e = rrg.edges[i];
        out << "    [" << e.first << ", " << e.second << "]";
        if (i + 1 < rrg.edges.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"edge_capacities\": {\n";
    size_t ci = 0;
    for (const auto& kv : rrg.capacities) {
        out << "    \"" << kv.first.first << "_" << kv.first.second << "\": " << kv.second;
        if (++ci < rrg.capacities.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  },\n";

    out << "  \"edge_distances\": {\n";
    ci = 0;
    for (const auto& kv : rrg.distances) {
        out << "    \"" << kv.first.first << "_" << kv.first.second << "\": " << kv.second;
        if (++ci < rrg.distances.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  },\n";

    out << "  \"edge_wl_scores\": {\n";
    ci = 0;
    for (const auto& kv : rrg.wl_scores) {
        out << "    \"" << kv.first.first << "_" << kv.first.second << "\": " << kv.second;
        if (++ci < rrg.wl_scores.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  },\n";

    out << "  \"coord_to_int\": {\n";
    size_t mi = 0;
    for (const auto& kv : coord_to_int) {
        out << "    \"" << kv.first.first << "_" << kv.first.second << "\": " << kv.second;
        if (++mi < coord_to_int.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";

    std::cerr << "Wrote RRG JSON: " << output_path << std::endl;
}
