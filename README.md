# Partial label learning for building occupancy estimation with DT-CG

Code for the paper *Reducing Annotation Cost in Building Occupancy Estimation Through Partial Label Learning and Decision-Tree Candidate Generation* (M. Y. Efrit, M. Amayri, N. Bouguila; submitted to *Energy and Buildings*).

## Setup

```
pip install -r requirements.txt
```

Put the input CSVs in `data/` (see `data/README.md`). ROBOD is public; Floor S is available on institutional request.


Run from the repository root (each mode writes to `results_revision4_paper/` by default; change with `--out-dir`):



## Results

`results/` contains, for the ROBOD rooms, the fold dates, per-fold metrics, candidate-set diagnostics, test-set predictions (`predictions_robod.csv.gz`) and summary tables of the final run. Files derived from Floor S are not redistributed.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT (see `LICENSE`).
