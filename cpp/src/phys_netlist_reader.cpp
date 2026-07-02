#include "phys_netlist_reader.h"

#include "coord_to_int.h"
#include "device_reader.h"

#include <PhysicalNetlist.capnp.h>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using SitePin = std::pair<std::string, std::string>;

void extract_site_pins(
    std::vector<SitePin>& site_pins,
    const std::vector<std::string>& str_list,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches) {
    std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Reader> queue;
    for (const auto route_branch : branches) {
        queue.push(route_branch);
    }
    while (!queue.empty()) {
        const auto route_branch = queue.front();
        queue.pop();
        const auto route_segment = route_branch.getRouteSegment();
        if (route_segment.which() ==
            PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::Which::SITE_PIN) {
            const auto sp = route_segment.getSitePin();
            site_pins.emplace_back(
                str_list[sp.getSite()],
                str_list[sp.getPin()]);
        }
        for (const auto branch : route_branch.getBranches()) {
            queue.push(branch);
        }
    }
}

void extract_site_pins_one_by_one(
    std::vector<SitePin>& site_pins,
    const std::vector<std::string>& str_list,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches) {
    for (const auto route_branch : branches) {
        std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Reader> queue;
        queue.push(route_branch);
        while (!queue.empty()) {
            const auto rb = queue.front();
            queue.pop();
            const auto route_segment = rb.getRouteSegment();
            if (route_segment.which() ==
                PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::Which::SITE_PIN) {
                const auto sp = route_segment.getSitePin();
                site_pins.emplace_back(
                    str_list[sp.getSite()],
                    str_list[sp.getPin()]);
                break;
            }
            for (const auto branch : rb.getBranches()) {
                queue.push(branch);
            }
        }
    }
}

void collect_pip_tiles(
    std::vector<std::pair<uint32_t, uint32_t>>& tiles,
    const InterchangeDevice& device,
    const std::vector<std::string>& str_list,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches) {
    std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Reader> queue;
    for (const auto rb : branches) {
        queue.push(rb);
    }
    while (!queue.empty()) {
        const auto rb = queue.front();
        queue.pop();
        const auto rs = rb.getRouteSegment();
        if (rs.isPip()) {
            const std::string tile_name = str_list[rs.getPip().getTile()];
            uint32_t row = 0;
            uint32_t col = 0;
            if (device.tile_coord(tile_name, row, col)) {
                tiles.emplace_back(row, col);
            }
        }
        for (const auto child : rb.getBranches()) {
            queue.push(child);
        }
    }
}

bool lookup_int_idx(
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& coord_to_int,
    uint32_t row,
    uint32_t col,
    uint32_t& out_idx) {
    const auto it = coord_to_int.find(std::make_pair(row, col));
    if (it == coord_to_int.end()) {
        return false;
    }
    out_idx = it->second;
    return true;
}

void add_tile_coord(
    std::vector<std::pair<uint32_t, uint32_t>>& tiles,
    uint32_t row,
    uint32_t col) {
    tiles.emplace_back(row, col);
}

bool site_to_tile(
    const InterchangeDevice& device,
    const std::string& site_name,
    uint32_t& row,
    uint32_t& col) {
    return device.site_coord(site_name, row, col);
}

}  // namespace

StubNetList extract_stub_nets(
    const std::string& phys_path,
    const InterchangeDevice& device,
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& coord_to_int,
    double expansion_ratio,
    uint32_t min_fanout,
    uint32_t max_nets) {
    const std::string raw = read_gzip_file(phys_path);

    std::stringstream stream(raw, std::ios::in | std::ios::out | std::ios::binary);
    kj::std::StdInputStream istream(stream);

    capnp::ReaderOptions opts;
    opts.nestingLimit = std::numeric_limits<int>::max();
    opts.traversalLimitInWords = std::numeric_limits<uint64_t>::max();

    capnp::InputStreamMessageReader reader(istream, opts);
    const auto netlist = reader.getRoot<PhysicalNetlist::PhysNetlist>();
    const auto str_list_reader = netlist.getStrList();
    std::vector<std::string> str_list;
    str_list.reserve(str_list_reader.size());
    for (const auto s : str_list_reader) {
        str_list.emplace_back(s.cStr());
    }
    const auto phys_nets = netlist.getPhysNets();

    StubNetList result;
    result.phys_nets_scanned = static_cast<uint32_t>(phys_nets.size());

    for (const auto phys_net : phys_nets) {
        const auto stubs = phys_net.getStubs();
        const auto sources = phys_net.getSources();
        const auto type = phys_net.getType();

        if (type != PhysicalNetlist::PhysNetlist::NetType::SIGNAL || stubs.size() == 0) {
            continue;
        }

        std::vector<SitePin> sink_pins;
        extract_site_pins_one_by_one(sink_pins, str_list, stubs);
        if (sink_pins.empty()) {
            continue;
        }

        std::vector<SitePin> source_pins;
        extract_site_pins(source_pins, str_list, sources);
        if (source_pins.empty()) {
            continue;
        }

        const uint32_t fanout = static_cast<uint32_t>(sink_pins.size());
        if (fanout <= min_fanout) {
            continue;
        }

        std::vector<std::pair<uint32_t, uint32_t>> tile_coords;
        for (const auto& src_pin : source_pins) {
            uint32_t row = 0;
            uint32_t col = 0;
            if (site_to_tile(device, src_pin.first, row, col)) {
                add_tile_coord(tile_coords, row, col);
            }
        }
        for (const auto& sink_pin : sink_pins) {
            uint32_t row = 0;
            uint32_t col = 0;
            if (site_to_tile(device, sink_pin.first, row, col)) {
                add_tile_coord(tile_coords, row, col);
            }
        }
        collect_pip_tiles(tile_coords, device, str_list, sources);

        if (tile_coords.size() < 2) {
            continue;
        }

        uint32_t min_row = UINT32_MAX;
        uint32_t max_row = 0;
        uint32_t min_col = UINT32_MAX;
        uint32_t max_col = 0;
        for (const auto& rc : tile_coords) {
            min_row = std::min(min_row, rc.first);
            max_row = std::max(max_row, rc.first);
            min_col = std::min(min_col, rc.second);
            max_col = std::max(max_col, rc.second);
        }

        const uint32_t w = std::max(1u, max_col - min_col);
        const uint32_t h = std::max(1u, max_row - min_row);
        const uint32_t exp_w = std::max(1u, static_cast<uint32_t>(w * expansion_ratio));
        const uint32_t exp_h = std::max(1u, static_cast<uint32_t>(h * expansion_ratio));
        min_col = (min_col > exp_w) ? (min_col - exp_w) : 0;
        max_col = std::min(device.device_cols, max_col + exp_w);
        min_row = (min_row > exp_h) ? (min_row - exp_h) : 0;
        max_row = std::min(device.device_rows, max_row + exp_h);

        uint32_t src_int_idx = UINT32_MAX;
        {
            uint32_t row = 0;
            uint32_t col = 0;
            if (!site_to_tile(device, source_pins[0].first, row, col)) {
                continue;
            }
            if (!lookup_int_idx(coord_to_int, row, col, src_int_idx)) {
                continue;
            }
        }

        std::vector<uint32_t> sink_int_idxs;
        sink_int_idxs.reserve(sink_pins.size());
        for (const auto& sink_pin : sink_pins) {
            uint32_t row = 0;
            uint32_t col = 0;
            if (!site_to_tile(device, sink_pin.first, row, col)) {
                continue;
            }
            uint32_t idx = UINT32_MAX;
            if (lookup_int_idx(coord_to_int, row, col, idx)) {
                sink_int_idxs.push_back(idx);
            }
        }
        if (sink_int_idxs.empty()) {
            continue;
        }

        StubNetRecord rec;
        rec.name = str_list[phys_net.getName()];
        rec.src_int_idx = src_int_idx;
        rec.sink_int_idxs = std::move(sink_int_idxs);
        rec.min_col = min_col;
        rec.max_col = max_col;
        rec.min_row = min_row;
        rec.max_row = max_row;
        rec.fanout = fanout;
        result.nets.push_back(std::move(rec));
        result.nets_kept++;

        if (max_nets > 0 && result.nets.size() >= max_nets) {
            break;
        }
    }

    return result;
}
