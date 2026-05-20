# Contributing to Miniature

Thanks for considering a contribution. Miniature follows the discipline of
the [rewrites.bio](https://rewrites.bio/) framework — even though it is an
original tool, we treat every refactor and optimisation as a "rewrite of
ourselves" and validate accordingly.

Please read this file before opening a PR.

## The non-negotiable rule

**Output must not regress.** Every change must preserve the equivalence
targets defined in [`docs/equivalence.md`](./docs/equivalence.md):

- SSIM ≥ 0.99 on output PNGs vs. the reference set.
- Mean ΔE2000 ≤ 1.0 over tissue pixels.
- Trustworthiness and perceptual trustworthiness within 0.005 of the
  recorded baseline.
- Mask coverage within ±1% of the baseline.

If a change alters output, it must either:

1. Fix a demonstrable bug (in which case update the baseline numbers in
   `docs/equivalence.md` in the same PR with a clear justification), or
2. Be opt-in behind a CLI flag that defaults to current behaviour.

PRs that change output without one of the above will be rejected.

## Compatibility surface (stable interfaces)

The following do not change in minor or patch releases:

- All CLI flags and their defaults (`--dimred`, `--n_components`,
  `--colormap`, etc.).
- Output PNG filename pattern (`{stem}.png`, `{stem}_{colormap}.png`).
- HDF5 keys: `mask`, `tissue_array`, `embedding`, `colors/{colormap}`.
- Exit codes.
- Python public API exported from `miniature/__init__.py`.

Breaking changes here require a major version bump and a CHANGELOG entry.

## Performance work

For perf PRs ("make X faster"):

1. **Capture a baseline first.** Run the benchmarks below on `main` and
   on your branch; include both numbers in the PR description.

   ```bash
   pytest tests/test_pipeline_benchmark.py tests/test_delta_e_scaling.py \
       -v -m benchmark -s
   ```

2. **One hot path per PR.** Keep changes focused so each speedup can be
   independently bisected and reverted.

3. **No language port without profile evidence.** Prefer pure-Python /
   NumPy fixes first; Numba only when profile shows a tight pure-Python
   loop; Rust / Cython only when both above have been exhausted and the
   evidence is in `docs/perf/`.

## How to propose a change

1. Open an issue first if the change is non-trivial.
2. For bug fixes, include a failing test against a minimal reproducible
   example, ideally using one of the small files in `data/`.
3. For new features, explain which real usage requires it. Unused
   features become maintenance debt.
4. Run the unit tests (`pytest tests/ -m "not benchmark"`). If you
   touched a perf-sensitive path, also run the benchmarks.

## AI assistance in contributions

You may use AI coding assistants. If you do:

- Disclose it in the PR description (tool + role: e.g. "Claude Code,
  Opus 4.7 — refactored `assign_colours_lab` vectorisation").
- The output-equivalence requirement is the same as for human-written
  code — no exemption for AI-assisted PRs.
- See the AI assistance disclosure block in `README.md` for the
  project-level statement on tooling and validation.

## Code style

- Python ≥ 3.11.
- `ruff` is configured in `pyproject.toml` — run `ruff check src/ tests/`
  and `ruff format src/ tests/` before pushing.
- Type hints on public functions; not required on internal helpers.

## Releasing

- Versioning: [semver](https://semver.org/).
- Tag releases (`vMAJOR.MINOR.PATCH`); update `CHANGELOG.md` under the
  new version heading; refresh the baseline numbers in
  `docs/equivalence.md` and `docs/perf/baseline.md` if any pin changed.
