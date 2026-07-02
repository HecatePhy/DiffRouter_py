#include "net_list_writer.h"

#include <fstream>
#include <iomanip>
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

void write_stub_net_list_json(
    const std::string& output_path,
    const StubNetList& net_list,
    double expansion_ratio,
    uint32_t min_fanout,
    uint32_t device_rows,
    uint32_t device_cols) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open output: " + output_path);
    }

    out << std::setprecision(17);
    out << "{\n";
    out << "  \"format_version\": 2,\n";
    out << "  \"route_filter\": \"stubs\",\n";
    out << "  \"expansion_ratio\": " << expansion_ratio << ",\n";
    out << "  \"min_fanout\": " << min_fanout << ",\n";
    out << "  \"device_rows\": " << device_rows << ",\n";
    out << "  \"device_cols\": " << device_cols << ",\n";
    out << "  \"phys_nets_scanned\": " << net_list.phys_nets_scanned << ",\n";
    out << "  \"nets_kept\": " << net_list.nets_kept << ",\n";
    out << "  \"nets\": [\n";

    for (size_t i = 0; i < net_list.nets.size(); ++i) {
        const auto& net = net_list.nets[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(net.name) << "\",\n";
        out << "      \"src_int_idx\": " << net.src_int_idx << ",\n";
        out << "      \"sink_int_idxs\": [";
        for (size_t j = 0; j < net.sink_int_idxs.size(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << net.sink_int_idxs[j];
        }
        out << "],\n";
        out << "      \"bbox\": [" << net.min_col << ", " << net.max_col << ", "
            << net.min_row << ", " << net.max_row << "],\n";
        out << "      \"fanout\": " << net.fanout << "\n";
        out << "    }";
        if (i + 1 < net_list.nets.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}
