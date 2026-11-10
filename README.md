# Partial label learning for building occupancy estimation with DT-CG

Code for the paper *Reducing Annotation Cost in Building Occupancy Estimation Through Partial Label Learning and Decision-Tree Candidate Generation* (M. Y. Efrit, M. Amayri, N. Bouguila; submitted to *Energy and Buildings*).

The single script `occupancy_pll_revised.py` builds the features, the time-blocked folds and the labeled seeds, generates DT-CG candidate sets, runs the four PLL methods (HTC-LTPLL, SoDisam, UCL, TERIAL; Algorithm 1 and Eqs. 7-16 of the paper) and the supervised baselines, and computes every table and statistic reported in the paper.

## Setup

```
pip install -r requirements.txt
```

Put the input CSVs in `data/` (see `data/README.md`). ROBOD is public; Floor S is available on institutional request.


Run from the repository root (each mode writes to `results_revision4_paper/` by default; change with `--out-dir`):

| Command | What it produces | Paper |
|---|---|---|
| `python occupancy_pll_revised.py revision --data-dir data` | Main run: all methods, budgets, candidate-set sources; folds, coverage, predictions, non-inferiority tests | Sections 6.1-6.7, Tables 5-12, B.14-E.18 |
| `python occupancy_pll_revised.py decomp --data-dir data` | Protocol decomposition (shuffled vs. time-blocked folds, scattered vs. whole-day seeds) | Table 9 |
| `python occupancy_pll_revised.py imbalance-sens --data-dir data` | No resampling vs. SMOTE vs. class weights | Section 6.8, Table F.19 |
| `python occupancy_pll_revised.py tau-sweep --data-dir data` | Threshold sensitivity, tau = 0.05-0.50 | Section 6.9, Table F.20 |
| `python occupancy_pll_revised.py tau-select-sens --data-dir data` | Seed-only tau selection rule | Section 6.9 |
| `python occupancy_pll_revised.py hardlabel --data-dir data` | LR on DT-CG hard labels (needs the main run's `fold_metrics.csv`) | Section 6.1, Tables 10 and F.21 |
| `python occupancy_pll_revised.py tables --data-dir data` | Summary tables from existing results | |

The main run takes several hours on a laptop; the other modes take minutes to about an hour. Use `--datasets ROBOD_R1 ROBOD_R2 ...` to run a subset, and `--quick` for a fast smoke test.

All randomness is seeded from `master_seed = 2026`; each fold, budget and seed placement uses a derived seed, so results are reproducible. `run_config.json` in each output folder records every setting.

## Results

`results/` contains, for the ROBOD rooms, the fold dates, per-fold metrics, candidate-set diagnostics, test-set predictions (`predictions_robod.csv.gz`) and summary tables of the final run. Files derived from Floor S are not redistributed.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT (see `LICENSE`).
