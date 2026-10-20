# oracle-v1 lock

These files define the frozen evaluation. Later rounds must not change them; any change means a
new version (oracle-v2) with its own tag and a re-run of every reported number.

| file | sha256 |
|---|---|
| validation/sprint/oracle.py | 6bf031954f35b34155a7bf448892d906a09fa58d80507c5859a120d8cc891e70 |
| validation/sprint/conventions.py | bd9569415e609297319efb2f0d7b4ec0df18882affbaeaae00937492c99e6ed8 |
| validation/sprint/splits_v1.json | f7106767622a838569421ed1f0617eebb3040b9ccea98481c3accdcbf576809c |

Check: `sha256sum validation/sprint/oracle.py validation/sprint/conventions.py validation/sprint/splits_v1.json`.
Tag: `oracle-v1`. Not locked (may change without a version bump as long as they do not change
the numbers): eval.py, baselines.py (new methods may be added; `gt`, `gt_rot`, `basis` must not
change), splits.py, probe_conventions.py, repro_legacy.py.

# oracle-v2 lock (2026-10-20)

oracle-v2 imports oracle-v1 unchanged (hashes above still hold) and only adds columns:
GT-bone-length FK for canonical-rest rotation outputs and the skate / ground / pop / swivel
columns. `eval.py` now defaults to `--oracle v2`; `--oracle v1` reproduces the Round 1 tables.
Gate: v1 numeric columns of gt, gt_rot and basis are identical under v2 (checked key by key).
Tag: `oracle-v2`.

| file | sha256 |
|---|---|
| validation/sprint/oracle_v2.py | fcde4cbc2f86c2a7d2dea4f4f4a41acb3c91953b97f41d943e13e73fce04bda9 |
| validation/sprint/rest.py | f18004216711d083bcd0c72b640ecc83c5f519daecfc22ddf4a2b862119d907c |
