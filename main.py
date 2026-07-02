"""
Legacy entry point — use run_exp.py for the full DiffRouter pipeline.

  python run_exp.py --testcase boom_soc_v2 --quiet
"""

import sys

if __name__ == "__main__":
    print("Use: python run_exp.py --help")
    print("Global-only: python src/GlobalRoute.py")
    sys.exit(1)
