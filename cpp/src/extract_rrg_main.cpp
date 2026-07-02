#include "coord_to_int.h"
#include "device_reader.h"
#include "json_writer.h"
#include "tile_rrg_builder.h"

#include <iostream>
#include <string>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " -i <device.device> -o <output.json>\n";
}

int main(int argc, char** argv) {
    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        InterchangeDevice device;
        device.load(input_path);

        TileRRG rrg = build_tile_rrg_from_path(input_path, device);
        auto coord_to_int = build_coord_to_int(device, input_path);

        write_rrg_json(output_path, device, rrg, coord_to_int);

        std::cerr << "Done. INT tiles=" << device.int_tiles.size()
                  << " edges=" << rrg.edges.size() << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
