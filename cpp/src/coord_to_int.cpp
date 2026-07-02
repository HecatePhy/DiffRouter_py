#include "coord_to_int.h"

#include "device_reader.h"

#include <DeviceResources.capnp.h>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>
#include <zlib.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

std::map<std::pair<uint32_t, uint32_t>, uint32_t> build_coord_to_int(
    const InterchangeDevice& dev_meta,
    const std::string& device_path) {
    const std::string raw = read_gzip_file(device_path);

    std::stringstream stream(raw, std::ios::in | std::ios::out | std::ios::binary);
    kj::std::StdInputStream istream(stream);

    capnp::ReaderOptions opts;
    opts.nestingLimit = std::numeric_limits<int>::max();
    opts.traversalLimitInWords = std::numeric_limits<uint64_t>::max();

    capnp::InputStreamMessageReader reader(istream, opts);
    auto device = reader.getRoot<DeviceResources::Device>();

    const auto tile_list = device.getTileList();
    const uint32_t rows = dev_meta.device_rows;
    const uint32_t cols = dev_meta.device_cols;

    std::vector<std::vector<bool>> has_tile(rows, std::vector<bool>(cols, false));
    for (const auto tile : tile_list) {
        const uint32_t r = tile.getRow();
        const uint32_t c = tile.getCol();
        if (r < rows && c < cols) {
            has_tile[r][c] = true;
        }
    }

    std::vector<std::vector<int32_t>> nearest_int(rows, std::vector<int32_t>(cols, -1));
    std::queue<std::pair<uint32_t, uint32_t>> q;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            if (dev_meta.int_grid[r][c] >= 0) {
                nearest_int[r][c] = dev_meta.int_grid[r][c];
                q.emplace(r, c);
            }
        }
    }

    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        const auto [r, c] = q.front();
        q.pop();
        const int32_t src_int = nearest_int[r][c];
        for (int k = 0; k < 4; ++k) {
            const int nr = static_cast<int>(r) + dr[k];
            const int nc = static_cast<int>(c) + dc[k];
            if (nr < 0 || nc < 0 || nr >= static_cast<int>(rows) || nc >= static_cast<int>(cols)) {
                continue;
            }
            if (!has_tile[static_cast<uint32_t>(nr)][static_cast<uint32_t>(nc)]) {
                continue;
            }
            if (nearest_int[static_cast<uint32_t>(nr)][static_cast<uint32_t>(nc)] >= 0) {
                continue;
            }
            nearest_int[static_cast<uint32_t>(nr)][static_cast<uint32_t>(nc)] = src_int;
            q.emplace(static_cast<uint32_t>(nr), static_cast<uint32_t>(nc));
        }
    }

    std::map<std::pair<uint32_t, uint32_t>, uint32_t> coord_to_int;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            if (!has_tile[r][c]) {
                continue;
            }
            const int32_t int_idx = nearest_int[r][c];
            if (int_idx < 0) {
                continue;
            }
            coord_to_int[{r, c}] = static_cast<uint32_t>(int_idx);
        }
    }

    std::cerr << "coord_to_int mappings: " << coord_to_int.size() << std::endl;
    return coord_to_int;
}
