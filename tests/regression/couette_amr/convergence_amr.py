#!/usr/bin/env python3
"""convergence_amr.py — Grid-convergence study for the AMR Couette test.

Runs test_couette_amr with a given inputs file (default: ./inputs, three
levels with refinement boxes fixed in physical space and n_error_buf = 0,
so the coarse/fine interfaces sit at the same physical location at every
resolution) for several level-0 resolutions Ny, with lbm.tau pinned so
nu_lat is constant, and stops every run at the same dimensionless time
nu*t/H^2. The composite inf-norm error printed by the test ("Final Steady
state error") is used to fit the observed order between consecutive
resolutions.

Usage:
    cd tests/regression/couette_amr
    make                                  # builds test_couette_amr2d.gnu.ex
    python3 convergence_amr.py --ny-list 32 64 128 --min-order 1.8
    python3 convergence_amr.py --inputs inputs_poiseuille amr.interp=lincc

Trailing KEY=VALUE arguments are passed to every run as extra overrides.

Requires only the Python standard library.
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS2 = 1.0 / 3.0
ERR_RE = re.compile(r"^Final Steady state error:\s*([-+0-9.eE]+)", re.MULTILINE)
DRIFT_RE = re.compile(r"^Final mass drift:\s*([-+0-9.eE]+)", re.MULTILINE)


def find_exe(pattern: str) -> Path:
    hits = sorted(HERE.glob(pattern))
    if not hits:
        sys.exit(f"no executable matching {pattern} in {HERE}; run `make` first")
    return hits[0]


def run_one(exe: Path, inputs: Path, ny: int, nx: int, tau: float, target_time: float,
            workdir: Path, extra: list) -> dict:
    nu_lat = (tau - 0.5) * CS2
    step = max(1, round(target_time * ny * ny / nu_lat))
    work = workdir / f"Ny{ny}"
    work.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"grid.ncells={nx} {ny}",
        f"grid.max_grid_size={ny}",
        f"lbm.tau={tau}",
        f"sim.max_step={step}",
        "amr.plot_int=-1",
        "amr.chk_int=-1",
        # thresholds in the inputs file are tuned for its own resolution
        "couette_amr.max_err=-1",
        *extra,
    ]
    print(f"Ny={ny:4d} Nx={nx:3d} steps={step:6d} -> {work}")
    out = subprocess.run([str(exe), str(inputs), *overrides], cwd=work,
                         capture_output=True, text=True)
    (work / "run.log").write_text(out.stdout + out.stderr)
    m = ERR_RE.search(out.stdout)
    if out.returncode != 0 or m is None:
        sys.exit(f"run Ny={ny} failed (exit {out.returncode}); see {work / 'run.log'}")
    d = DRIFT_RE.search(out.stdout)
    return {"Ny": ny, "step": step, "err": float(m.group(1)),
            "drift": float(d.group(1)) if d else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ny-list", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--target-time", type=float, default=0.05,
                    help="dimensionless time nu*t/H^2 of the comparison (default 0.05)")
    ap.add_argument("--inputs", type=Path, default=HERE / "inputs")
    ap.add_argument("--aspect", type=float, default=0.125,
                    help="Nx/Ny (default 0.125, as in ./inputs); Nx is at least 4")
    ap.add_argument("--min-order", type=float, default=None,
                    help="exit non-zero if the worst observed order is below this")
    ap.add_argument("--exe", type=Path, default=None,
                    help="test executable (default: test_couette_amr*2d*.ex in this directory)")
    ap.add_argument("--workdir", type=Path, default=HERE / "convergence_runs",
                    help="directory for the run subdirectories (default ./convergence_runs)")
    ap.add_argument("overrides", nargs="*", help="extra KEY=VALUE overrides for every run")
    args = ap.parse_args()

    exe = args.exe.resolve() if args.exe else find_exe("test_couette_amr*2d*.ex")
    rows = []
    for ny in sorted(args.ny_list):
        nx = max(4, round(ny * args.aspect))
        rows.append(run_one(exe, args.inputs.resolve(), ny, nx, args.tau, args.target_time,
                            args.workdir.resolve(), args.overrides))

    print(f"\n{'Ny':>6}{'steps':>8}{'err':>14}{'order':>8}{'mass drift':>13}")
    orders = []
    for i, r in enumerate(rows):
        order = None
        if i > 0:
            p = rows[i - 1]
            order = math.log(p["err"] / r["err"]) / math.log(r["Ny"] / p["Ny"])
            orders.append(order)
        o = f"{order:.2f}" if order is not None else "-"
        print(f"{r['Ny']:>6}{r['step']:>8}{r['err']:>14.6e}{o:>8}{r['drift']:>13.3e}")

    if args.min_order is not None and orders and min(orders) < args.min_order:
        print(f"FAIL: worst observed order {min(orders):.2f} < {args.min_order}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
