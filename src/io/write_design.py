"""Write routed designs to FPGA interchange and DCP formats."""

import os
from typing import Any


def ensure_result_dir(testcase: str, base: str = "results") -> str:
    path = os.path.join(base, testcase)
    os.makedirs(path, exist_ok=True)
    checkpoint = os.path.join(path, "checkpoint")
    os.makedirs(checkpoint, exist_ok=True)
    return path


def write_routed_dcp(design: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    design.writeCheckpoint(output_path)
    print(f"Routed DCP saved to: {output_path}")


def write_routed_phys(design: Any, output_path: str) -> None:
    from com.xilinx.rapidwright.interchange import PhysNetlistWriter

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PhysNetlistWriter.writePhysNetlist(design, output_path)
    print(f"Routed PHYS saved to: {output_path}")
