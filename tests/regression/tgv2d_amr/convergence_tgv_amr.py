#!/usr/bin/env python3
"""convergence_tgv_amr.py — Grid-convergence study for the AMR Taylor-Green test.

Runs test_tgv2d_amr with ./inputs (refinement boxes fixed in physical space,
n_error_buf = 0, so the coarse/fine interfaces sit at the same physical
location at every resolution) for several level-0 resolutions N, with
lbm.tau pinned so nu_lat is constant and units.ref_length = N (as in
tests/regression/tgv2d/convergence.py, which makes u0_lat = nu_lat Re / N:
diffusive scaling). Every run stops at the same dimensionless decay time
nu k^2 t. The composite inf-norm and L2 errors printed by the test (relative
to the current vortex amplitude) are used to fit the observed order between
consecutive resolutions.

Usage:
    cd tests/regression/tgv2d_amr
    make                                   # builds test_tgv2d_amr2d.gnu.ex
    python3 convergence_tgv_amr.py --n-list 32 64 128 --min-order 1.8
    python3 convergence_tgv_amr.py --exe ../../../build2d/tests/test_tgv2d_amr amr.interp=quadratic

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
PATTERNS = {
    "err": re.compile(r"^Final velocity error:\s*([-+0-9.eE]+)", re.MULTILINE),
    "l2": re.compile(r"^Final L2 velocity error:\s*([-+0-9.eE]+)", re.MULTILINE),
    "drift": re.compile(r"^Final mass drift:\s*([-+0-9.eE]+)", re.MULTILINE),
}


def find_exe(pattern: str) -> Path:
    hits = sorted(HERE.glob(pattern))
    if not hits:
        sys.exit(f"no executable matching {pattern} in {HERE}; run `make` first")
    return hits[0]


def run_one(exe: Path, inputs: Path, n: int, tau: float, decay_time: float,
            workdir: Path, extra: list) -> dict:
    nu_lat = (tau - 0.5) * CS2
    k = 2.0 * math.pi / n
    step = max(1, round(decay_time / (nu_lat * k * k)))
    work = workdir / f"N{n}"
    work.mkdir(parents=True, exist_ok=True)
    # Suppress tgv_amr thresholds (tuned for N=64) when the caller overrides
    # amr.max_level (e.g. single-level baseline) so min_finest does not trip,
    # otherwise leave them active so a broken AMR is caught.
    overrides_extra = list(extra)
    if any("max_level" in s for s in extra):
        overrides_extra += ["tgv_amr.max_err=-1", "tgv_amr.max_l2_err=-1",
                            "tgv_amr.min_finest=-1", "tgv_amr.max_mass_drift=-1"]
    else:
        overrides_extra += ["tgv_amr.max_err=-1", "tgv_amr.max_l2_err=-1"]
    overrides = [
        f"grid.ncells={n} {n}",
        f"units.ref_length={n}",
        f"lbm.tau={tau}",
        f"sim.max_step={step}",
        "amr.plot_int=-1",
        "amr.chk_int=-1",
        *overrides_extra,
    ]
    proc = subprocess.run([str(exe), str(inputs), *overrides], cwd=work,
                          capture_output=True, text=True)
    (work / "run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.exit(f"run N={n} failed (exit {proc.returncode}); see {work / 'run.log'}")
    out = {"n": n, "step": step}
    for key, pat in PATTERNS.items():
        m = pat.search(proc.stdout)
        if not m:
            sys.exit(f"could not parse '{key}' from N={n} output; see {work / 'run.log'}")
        out[key] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, default=None,
                    help="test executable (default: ./test_tgv2d_amr2d*.ex)")
    ap.add_argument("--inputs", type=Path, default=HERE / "inputs")
    ap.add_argument("--n-list", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--decay-time", type=float, default=0.5,
                    help="dimensionless stop time nu k^2 t (default 0.5)")
    ap.add_argument("--workdir", type=Path, default=HERE / "convergence_runs")
    ap.add_argument("--min-order", type=float, default=None,
                    help="fail (exit 1) if any observed inf-norm order is below this")
    ap.add_argument("overrides", nargs="*", help="extra KEY=VALUE overrides")
    args = ap.parse_args()

    exe = args.exe.resolve() if args.exe else find_exe("test_tgv2d_amr2d*.ex")
    inputs = args.inputs.resolve()
    results = [run_one(exe, inputs, n, args.tau, args.decay_time, args.workdir, args.overrides)
               for n in args.n_list]

    print(f"{'N':>6} {'steps':>7} {'inf err':>12} {'order':>6} {'L2 err':>12} {'order':>6} {'mass drift':>11}")
    orders = []
    prev = None
    for r in results:
        o_inf = o_l2 = ""
        if prev is not None:
            ratio = math.log(r["n"] / prev["n"])
            oi = math.log(prev["err"] / r["err"]) / ratio
            ol = math.log(prev["l2"] / r["l2"]) / ratio
            orders.append(oi)
            o_inf, o_l2 = f"{oi:.2f}", f"{ol:.2f}"
        print(f"{r['n']:>6} {r['step']:>7} {r['err']:>12.4e} {o_inf:>6} {r['l2']:>12.4e} {o_l2:>6} {r['drift']:>11.2e}")
        prev = r

    if args.min_order is not None and any(o < args.min_order for o in orders):
        print(f"FAIL: observed order below {args.min_order}")
        sys.exit(1)


if __name__ == "__main__":
    main()
