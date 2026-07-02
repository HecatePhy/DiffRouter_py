#include "coord_to_int.h"
#include "device_reader.h"
#include "net_list_writer.h"
#include "phys_netlist_reader.h"

#include <iostream>
#include <string>

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " -d <device.device> -i <input.phys> -o <output.json>\n"
        << "       [--expansion-ratio R] [--min-fanout N] [--max-nets N]\n";
}

int main(int argc, char** argv) {
    std::string device_path;
    std::string phys_path;
    std::string output_path;
    double expansion_ratio = 0.1;
    uint32_t min_fanout = 0;
    uint32_t max_nets = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_path = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            phys_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--expansion-ratio" && i + 1 < argc) {
            expansion_ratio = std::stod(argv[++i]);
        } else if (arg == "--min-fanout" && i + 1 < argc) {
            min_fanout = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--max-nets" && i + 1 < argc) {
            max_nets = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    if (device_path.empty() || phys_path.empty() || output_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        InterchangeDevice device;
        device.load(device_path);

        std::cerr << "Reading physical netlist: " << phys_path << std::endl;
        const auto coord_to_int = build_coord_to_int(device, device_path);

        StubNetList net_list = extract_stub_nets(
            phys_path,
            device,
            coord_to_int,
            expansion_ratio,
            min_fanout,
            max_nets);

        write_stub_net_list_json(
            output_path,
            net_list,
            expansion_ratio,
            min_fanout,
            device.device_rows,
            device.device_cols);

        std::cerr << "Scanned " << net_list.phys_nets_scanned << " phys nets, kept "
                  << net_list.nets.size() << " stub nets -> " << output_path << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
