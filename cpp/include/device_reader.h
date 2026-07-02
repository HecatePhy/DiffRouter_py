#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct IntTileInfo {
    uint32_t row;       // Interchange device grid (pin / coord_to_int)
    uint32_t col;
    uint32_t fabric_x;  // Parsed from INT_X#Y# tile name (routing lattice)
    uint32_t fabric_y;
    std::string name;
};

struct InterchangeDevice {
    std::string device_name;
    uint32_t device_rows = 0;
    uint32_t device_cols = 0;

    std::vector<std::string> str_list;
    std::vector<IntTileInfo> int_tiles;
    std::unordered_map<uint64_t, uint32_t> int_coord_to_idx;

    // Site / tile name -> device grid (row, col)
    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> site_to_coord;
    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> tile_name_to_coord;

    // (row,col) -> dense INT index for any grid coord that has an INT tile
    std::vector<std::vector<int32_t>> int_grid;

    void load(const std::string& device_path);

    bool site_coord(const std::string& site_name, uint32_t& row, uint32_t& col) const;
    bool tile_coord(const std::string& tile_name, uint32_t& row, uint32_t& col) const;
};

std::string read_gzip_file(const std::string& path);

uint64_t pack_coord(uint32_t row, uint32_t col);

// Parse fabric indices from tile names like INT_X3Y220 -> x=3, y=220.
bool parse_int_fabric_xy(const std::string& name, uint32_t& fabric_x, uint32_t& fabric_y);
