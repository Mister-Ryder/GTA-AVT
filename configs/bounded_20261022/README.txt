Bounded initial GTA-AVT delivery plan, 2026-10-22

One seed20261022, 1.0mm cache, patch96, batch2. Pilot:200 updates, three predeclared cohort validation patients. Formal:5000 updates, all8validation patients at2500/5000. No external networks or automatic held-out test.

Role main: ordinary attribute fusion, full soft GTA, matched constant soft GTA.
Role ablations: full soft GTA platform bridge, scale-only, structure-only, original hard anchor.

Use tools/run_bounded.py with explicit project/cache roots, Python interpreter, role and phase. Review --help and --plan-only. Pilot and formal are separate invocations; prerequisites check current-source fixtures and actual cache hashes. Completion never starts another phase or test.

Results pending. The bridge does not establish exact optimization equivalence across environments.
