# Eight-model Kaggriculture runtime

The competition agent image is not the same as the current Kaggle Notebook
image. The validation worker uses Python 3.11 and preloads NumPy 2.4.6 before
executing `main.py`. Removing NumPy from `sys.modules` or vendoring a different
NumPy build is unsafe and does not replace the already-loaded extension.

Build the archive with Python 3.11 manylinux wheels for every compiled package
listed in `runtime-requirements-py311.txt`, but do not include a NumPy wheel.
The known-good remote combination is SciPy 1.16.3, scikit-learn 1.6.1,
Pandas 2.2.3, Matplotlib 3.10.8, and contourpy 1.3.2. PyTabKit and Skorch are
pure-Python wheels and are also vendored.

Remote acceptance requires all of the following:

- `ENSEMBLE_RUNTIME_OK models=8` appears once in the agent log.
- Warm calls have non-trivial model latency (about 0.22 seconds in validation),
  rather than the roughly 0.00003-second deterministic fallback.
- Both seats end with `DONE` and retain positive overage time.
- No traceback is emitted by bundle loading or prediction.

Submission `55783103`, validation episode `99703638`, satisfied these checks.
