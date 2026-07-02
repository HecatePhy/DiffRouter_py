#pragma once

#include <string>

#include "phys_netlist_reader.h"

void write_stub_net_list_json(
    const std::string& output_path,
    const StubNetList& net_list,
    double expansion_ratio,
    uint32_t min_fanout,
    uint32_t device_rows,
    uint32_t device_cols);
