#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct InterchangeDevice;

struct StubNetRecord {
    std::string name;
    uint32_t src_int_idx = UINT32_MAX;
    std::vector<uint32_t> sink_int_idxs;
    uint32_t min_col = 0;
    uint32_t max_col = 0;
    uint32_t min_row = 0;
    uint32_t max_row = 0;
    uint32_t fanout = 0;
};

struct StubNetList {
    uint32_t phys_nets_scanned = 0;
    uint32_t nets_kept = 0;
    std::vector<StubNetRecord> nets;
};

StubNetList extract_stub_nets(
    const std::string& phys_path,
    const InterchangeDevice& device,
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& coord_to_int,
    double expansion_ratio,
    uint32_t min_fanout,
    uint32_t max_nets = 0);
