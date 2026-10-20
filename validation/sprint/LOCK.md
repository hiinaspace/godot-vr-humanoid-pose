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
