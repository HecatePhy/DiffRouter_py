#include "device_reader.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>
#include <zlib.h>

#include <DeviceResources.capnp.h>

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

uint64_t pack_coord(uint32_t row, uint32_t col) {
    return (static_cast<uint64_t>(row) << 32) | col;
}

bool parse_int_fabric_xy(const std::string& name, uint32_t& fabric_x, uint32_t& fabric_y) {
    const size_t xpos = name.find("_X");
    if (xpos == std::string::npos) {
        return false;
    }
    const size_t ypos = name.find('Y', xpos + 2);
    if (ypos == std::string::npos) {
        return false;
    }
    try {
        fabric_x = static_cast<uint32_t>(std::stoul(name.substr(xpos + 2, ypos - xpos - 2)));
        fabric_y = static_cast<uint32_t>(std::stoul(name.substr(ypos + 1)));
    } catch (...) {
        return false;
    }
    return true;
}

std::string read_gzip_file(const std::string& path) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == Z_NULL) {
        throw std::runtime_error("Failed to open gzip device file: " + path);
    }
    std::string data;
    data.reserve(64 * 1024 * 1024);
    char buf[65536];
    int ret;
    while ((ret = gzread(file, buf, sizeof(buf))) > 0) {
        data.append(buf, static_cast<size_t>(ret));
    }
    if (ret < 0) {
        gzclose(file);
        throw std::runtime_error("Error reading gzip device file: " + path);
    }
    if (gzclose(file) != Z_OK) {
        throw std::runtime_error("Error closing gzip device file: " + path);
    }
    return data;
}

void InterchangeDevice::load(const std::string& device_path) {
    std::cerr << "Loading device: " << device_path << std::endl;
    const std::string raw = read_gzip_file(device_path);

    std::stringstream stream(raw, std::ios::in | std::ios::out | std::ios::binary);
    kj::std::StdInputStream istream(stream);

    capnp::ReaderOptions opts;
    opts.nestingLimit = std::numeric_limits<int>::max();
    opts.traversalLimitInWords = std::numeric_limits<uint64_t>::max();

    capnp::InputStreamMessageReader reader(istream, opts);
    auto device = reader.getRoot<DeviceResources::Device>();

    device_name = device.getName().cStr();
    const auto str_list_reader = device.getStrList();
    str_list.reserve(str_list_reader.size());
    for (const auto s : str_list_reader) {
        str_list.emplace_back(s.cStr());
    }

    const auto tile_type_list = device.getTileTypeList();
    std::vector<std::string> tile_type_names(tile_type_list.size());
    for (uint32_t i = 0; i < tile_type_list.size(); ++i) {
        tile_type_names[i] = str_list[tile_type_list[i].getName()];
    }

    const auto tile_list = device.getTileList();
    device_rows = 0;
    device_cols = 0;

    site_to_coord.reserve(tile_list.size() * 2);
    tile_name_to_coord.reserve(tile_list.size());

    for (const auto tile : tile_list) {
        const uint32_t row = tile.getRow();
        const uint32_t col = tile.getCol();
        device_rows = std::max(device_rows, row + 1);
        device_cols = std::max(device_cols, col + 1);

        const std::string tile_name = str_list[tile.getName()];
        tile_name_to_coord.emplace(tile_name, std::make_pair(row, col));

        for (const auto tile_site : tile.getSites()) {
            const std::string site_name = str_list[tile_site.getName()];
            site_to_coord.emplace(site_name, std::make_pair(row, col));
        }

        const uint32_t type_idx = tile.getType();
        if (type_idx >= tile_type_names.size()) {
            continue;
        }
        if (tile_type_names[type_idx] != "INT") {
            continue;
        }
        const uint32_t int_idx = static_cast<uint32_t>(int_tiles.size());
        IntTileInfo info;
        info.row = row;
        info.col = col;
        info.name = tile_name;
        if (!parse_int_fabric_xy(info.name, info.fabric_x, info.fabric_y)) {
            throw std::runtime_error("Failed to parse INT fabric X/Y from tile name: " + info.name);
        }
        int_tiles.push_back(info);
        int_coord_to_idx[pack_coord(row, col)] = int_idx;
    }

    int_grid.assign(device_rows, std::vector<int32_t>(device_cols, -1));
    for (uint32_t i = 0; i < int_tiles.size(); ++i) {
        int_grid[int_tiles[i].row][int_tiles[i].col] = static_cast<int32_t>(i);
    }

    std::cerr << "Device: " << device_name << " grid " << device_rows << "x" << device_cols
              << " INT tiles: " << int_tiles.size()
              << " sites: " << site_to_coord.size() << std::endl;
}

bool InterchangeDevice::site_coord(
    const std::string& site_name, uint32_t& row, uint32_t& col) const {
    const auto it = site_to_coord.find(site_name);
    if (it == site_to_coord.end()) {
        return false;
    }
    row = it->second.first;
    col = it->second.second;
    return true;
}

bool InterchangeDevice::tile_coord(
    const std::string& tile_name, uint32_t& row, uint32_t& col) const {
    const auto it = tile_name_to_coord.find(tile_name);
    if (it == tile_name_to_coord.end()) {
        return false;
    }
    row = it->second.first;
    col = it->second.second;
    return true;
}
