# Preprocessing Method Comparison — Summary Report

Phase-contrast reference built from **5** images: mean=70.86, std=3.96, median(sampled)=71.00.

Evaluated on **2** brightfield (non-phase) images.

## Ranking (composite score, higher = better)

|   rank | run_label                                               |   composite_score |   bhattacharyya_to_phase_domain |   wasserstein_to_phase_domain |   background_unevenness_score |   edge_preservation_index |     gmsd |
|-------:|:--------------------------------------------------------|------------------:|--------------------------------:|------------------------------:|------------------------------:|--------------------------:|---------:|
|      1 | basic__nohistmatch                                      |          0.855893 |                         3.56544 |                       87.4224 |                       8.68623 |                  0.999222 | 0.949442 |
|      2 | polynomial__order2_background_percentile40__nohistmatch |          0.679565 |                         6.03203 |                      137.181  |                       7.34139 |                  1        | 0.999767 |
|      3 | gaussian__sigma25__nohistmatch                          |          0.649132 |                         6.13864 |                      137.181  |                       8.76418 |                  1        | 0.999609 |
|      4 | none__nohistmatch                                       |          0.548637 |                         5.76321 |                      137.181  |                     nan       |                  1        | 1        |
|      5 | morphological_opening__radius25__nohistmatch            |          0.33136  |                         6.46941 |                      137.181  |                      11.5233  |                  0.882808 | 0.978374 |
|      6 | rolling_ball__radius50__nohistmatch                     |          0.180734 |                         5.97874 |                      137.181  |                      26.1368  |                  0.968909 | 0.937486 |


## Recommendation

**Best-performing configuration: `basic__nohistmatch`** (composite score 0.856).

This configuration achieves the best *combined* balance of domain similarity to the phase-contrast training distribution (Bhattacharyya=3.565, Wasserstein=87.422) **and** structural fidelity (edge preservation=0.999, GMSD=0.949), while keeping the background-unevenness score (8.686) low — i.e. it does not introduce the large-scale patchwork/circular artifacts that disqualified rolling ball at every radius tested previously.

**Weakest configuration: `rolling_ball__radius50__nohistmatch`** (composite score 0.181) — kept in the table for context; see its per-image comparison figures in `figures/` to inspect failure mode directly.

## Rolling-ball failure-mode analysis

Rolling ball assumes a single uniform radius of curvature describes the
entire background surface. When tested at radii 50-150 px, the metrics in
`results_all_runs.csv` should be read together with the background-unevenness
score, not the histogram-distance columns alone: if `background_unevenness_score`
for rolling-ball runs is high even at the radius previously suggested by
Bhattacharyya/Wasserstein optimization (radius 150), that confirms the failure
is structural — the ball settles into locally-fit spherical caps across a
large image, producing visible seams between regions — rather than caused by
choosing the wrong radius. Three independent contributing factors are
distinguished by this experiment's design:
(1) radius vs. image size ratio (test by comparing unevenness score trend
across radii — if it does not monotonically improve with radius, the method
itself is saturating, not under-tuned);
(2) interaction with histogram matching (compare hist-matched vs.
non-hist-matched rolling-ball runs — histogram matching cannot repair
spatial unevenness, it can only redistribute the global histogram, so equal
or worse unevenness scores after matching would confirm this);
(3) cell size / resolution mismatch (cross-reference against the polynomial
and Gaussian background estimates at comparable spatial scale — if those
smooth, globally-fit methods score better on unevenness at every parameter
setting, it indicates the local/iterative nature of rolling ball — not the
radius choice — is the root cause for these images).


## Notes on method selection

- The composite score intentionally weights `background_unevenness_score` heavily (30%) specifically because histogram-only metrics (Bhattacharyya, Wasserstein) cannot detect spatially localized shading artifacts — a method can match the target histogram globally while still containing visible circular patches, which is exactly what was observed with rolling ball previously.
- CLAHE variants are included in the full results table for completeness but are expected to score worse on `edge_preservation_index`/background flatness once background noise is amplified, consistent with the earlier finding that CLAHE strengthens background texture.
- Inspect `figures/<run_label>/<image>_comparison.png` for any top-3 candidate before final adoption — quantitative ranking should always be confirmed visually per the project's stated evaluation philosophy.
