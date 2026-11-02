#!/usr/bin/env python3
"""convergence_shearwave_amr.py — Grid-convergence study for the shearwave AMR test.

Runs test_shearwave_amr with ./inputs (refinement boxes fixed in physical
space, n_error_buf=0) for several level-0 resolutions N (where N is the value
for grid.ncells in the y-direction; aspect ratio of the inputs is preserved,
with e.g. z = N/2 for the 3D case), with lbm.tau pinned so nu_lat is constant
and units.ref_length = N (as in tgv2d_amr/convergence_tgv_amr.py, which makes
u0_lat = nu_lat Re / N via diffusive scaling). Every run stops at the same
dimensionless decay time nu k^2 t (0.5 by default). The composite inf-norm and
L2 errors printed by the test (relative to the current amplitude) are used to
fit the observed order between consecutive resolutions. Also reports mass drift
and momentum defect from the post-collision f totals.

Usage:
    cd tests/regression/shearwave_amr
    make DIM=2                           # builds test_shearwave_amr2d.gnu.ex
    python3 convergence_shearwave_amr.py --n-list 32 64 128 --min-order 1.8
    python3 convergence_shearwave_amr.py --exe /tmp/mu2d_test/tests/test_shearwave_amr amr.interp=quadratic

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
    "mom": re.compile(r"^Final momentum defect:\s*([-+0-9.eE]+)", re.MULTILINE),
}


def find_exe(pattern: str) -> Path:
    hits = sorted(HERE.glob(pattern))
    if not hits:
        sys.exit(f"no executable matching {pattern} in {HERE}; run `make` first")
    return hits[0]


def parse_ncells(inputs: Path):
    """Read grid.ncells from inputs file to get aspect ratio."""
    text = inputs.read_text()
    m = re.search(r"^\s*grid\.ncells\s*=\s*(.+)", text, re.MULTILINE)
    if not m:
        sys.exit(f"could not find grid.ncells in {inputs}")
    vals = [int(x) for x in m.group(1).split()]
    return vals


def run_one(exe: Path, inputs: Path, n: int, tau: float, decay_time: float,
            workdir: Path, extra: list, base_ncells) -> dict:
    nu_lat = (tau - 0.5) * CS2
    k = 2.0 * math.pi / n
    step = max(1, round(decay_time / (nu_lat * k * k)))
    work = workdir / f"N{n}"
    work.mkdir(parents=True, exist_ok=True)

    # Scale grid.ncells preserving aspect ratio relative to base
    # base_ncells e.g. [64,64] (2D) or [32,32,16] (3D). The y-direction defines N.
    base_ny = base_ncells[1] if len(base_ncells) > 1 else base_ncells[0]
    scaled = [max(1, round(c * n / base_ny)) for c in base_ncells]
    ncells_str = " ".join(str(c) for c in scaled)

    # Suppress shearwave thresholds when caller overrides max_level
    overrides_extra = list(extra)
    if any("max_level" in s for s in extra):
        overrides_extra += ["shearwave.max_err=-1", "shearwave.max_l2_err=-1",
                            "shearwave.min_finest=-1", "shearwave.max_mass_drift=-1",
                            "shearwave.max_mom_defect=-1"]
    else:
        overrides_extra += ["shearwave.max_err=-1", "shearwave.max_l2_err=-1",
                            "shearwave.max_mass_drift=-1", "shearwave.max_mom_defect=-1"]
    overrides = [
        f"grid.ncells={ncells_str}",
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
                    help="test executable (default: ./test_shearwave_amr*2d*.ex)")
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

    exe = args.exe.resolve() if args.exe else find_exe("test_shearwave_amr*2d*.ex")
    inputs = args.inputs.resolve()
    base_ncells = parse_ncells(inputs)
    results = [run_one(exe, inputs, n, args.tau, args.decay_time, args.workdir, args.overrides, base_ncells)
               for n in args.n_list]

    print(f"{'N':>6} {'steps':>7} {'inf err':>12} {'order':>6} {'L2 err':>12} {'order':>6} {'mass drift':>11} {'mom defect':>11}")
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
        print(f"{r['n']:>6} {r['step']:>7} {r['err']:>12.4e} {o_inf:>6} {r['l2']:>12.4e} {o_l2:>6} {r['drift']:>11.2e} {r['mom']:>11.2e}")
        prev = r

    if args.min_order is not None and any(o < args.min_order for o in orders):
        print(f"FAIL: observed order below {args.min_order}")
        sys.exit(1)


if __name__ == "__main__":
    main()
