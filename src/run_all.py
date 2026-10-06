"""Run the whole pipeline: Task 1 -> Task 2 -> Task 3 -> Task 4.

    python -m src.run_all --config configs/default.yaml --student-id YOUR_ID

Use ``--skip task2 task4`` etc. to leave stages out (the video demo can show saved outputs
instead of re-running the expensive Monte Carlo stages).  Task 3 is skipped with a warning
if the course CSVs are not in ``data/raw/``.
"""
from __future__ import annotations

import time

from . import task1_ridge, task2_alternative, task3_hidden_prediction, task4_nn_dynamics
from .config import base_parser

STAGES = ("task1", "task2", "task3", "task4")


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--student-id", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGES, help="stages to skip")
    args = ap.parse_args(argv)
    common = ["--config", args.config] + (["--output-dir", args.output_dir] if args.output_dir else [])
    t3 = common + (["--student-id", args.student_id] if args.student_id else []) \
        + (["--raw-dir", args.raw_dir] if args.raw_dir else [])
    plan = {"task1": (task1_ridge.main, common), "task2": (task2_alternative.main, common),
            "task3": (task3_hidden_prediction.main, t3), "task4": (task4_nn_dynamics.main, common)}
    for name in STAGES:
        if name in args.skip:
            print(f"== skipping {name}")
            continue
        print(f"== running {name}")
        t0 = time.time()
        fn, a = plan[name]
        fn(a)
        print(f"== {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
