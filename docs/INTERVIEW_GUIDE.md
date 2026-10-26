# Ready Line: interview guide

Notes for presenting this project. Every number below comes from the pipeline's outputs (`site/data/*.json`). If you re-run the pipeline, re-check them against the site.

## The 60-second version

> "I wanted to know whether predicting jet-engine removals actually buys readiness, so I built the whole chain on public data. I ingested 624,000 FAA maintenance reports and NASA's 1,416 run-to-failure engines with PySpark into a DuckDB warehouse. I tested five hypotheses. The key one: engines wear out (Weibull shape 3.4), so early removal has value. Then I built a LightGBM + CNN ensemble that predicts remaining life to within 4.7 cycles in the window where maintenance decisions happen. I replayed those predictions in a GPU Monte Carlo of a 24-jet squadron. Predictive removal raised mission-capable rate from 59.9% to 63.8% and cut unscheduled removals by 98%. The surprise: once you have any decent warning, extra accuracy mostly saves money, not readiness, because spares and depot speed become the bottleneck. I finished with a MILP that sets spares levels and a depot induction schedule, and scored the schedule against NASA's withheld answers."

## How the project maps to the job posting

| Posting asks for | Where it shows up |
|---|---|
| Big data environment; Spark / Hive / SQL; distributed computing | `01_spark_etl.py`: PySpark ETL of 10 multi-line CSVs (624k rows), Spark ML k-means, Spark window functions for rolling features. `02_warehouse.py`: DuckDB star schema + SQL |
| Reconciling disparate datasets | Six operating regimes normalized into one scale across 4 C-MAPSS subsets; SDR codes decoded; SDR part counts mapped to notional fighter LRUs; Spark vs. pandas row-count reconciliation |
| EDA, latent factors, hypothesis testing | `03_explore.py`: regime analysis, sensor ranking, health index, unsupervised fault-mode discovery (GMM + BIC + dip test). `04_hypotheses.py`: H1–H5 with LRT, log-rank, logistic regression odds ratios, interrupted time series |
| Predictive models, deep learning, GPU | `05`–`07`: LightGBM with engine-grouped CV, 1D-CNN on Apple GPU (7.8× faster than CPU), ensemble, conformal intervals, TreeSHAP |
| Text mining / NLP | `08_nlp_sdr.py`: TF-IDF + linear classifier (89.9% on 39 systems, temporal holdout), leakage ablation, miscode detection, NMF topic model, risk-weighted Pareto |
| Time series and demand forecasting | `09_forecast.py`: ETS vs SARIMAX with COVID intervention; Croston / SBA / TSB for intermittent part demand |
| Modeling & simulation; PBL metrics (MC, NMCM, NMCS) | `sim_engine.py`, `10_simulate.py`: vectorized Monte Carlo, policy comparison, value of prognostics, spares × turnaround surfaces |
| Optimization solvers (Gurobi, CPLEX, IPOPT) | `11_optimize.py`: Pyomo models solved with HiGHS (one-line swap to Gurobi/CPLEX); spares MILP frontier; induction scheduling MILP |
| Quantitative framework for evaluating alternatives | Section 10 of the site: analysis of alternatives with P(meets PBL), cost, and a sensitivity tornado |
| Problem statements, GR&A, root cause, corrective actions | Site sections 01, 02 (GR&A table), 06 and 10 (root cause from write-ups) |
| Visualization & communication | The site itself: BLUF, squadron status board, interactive simulator |
| AI development tools / agentic coding | Built with an AI coding agent; verification checks listed in section 12 |

## Questions you should expect, with short answers

**Why cap RUL at 125 cycles?** Early in life the sensors carry no degradation signal, so a linear "cycles left" target asks the model to predict noise. A piecewise-linear target (flat at 125, then counting down) is standard for C-MAPSS. The cost is that test engines with more than 125 cycles left always look under-predicted. That's why the FD002/FD004 RMSE (about 23) is much higher than the decision-zone RMSE (4.7).

**How did you prevent leakage?** (1) Every feature uses only the engine's past: trailing windows and a baseline from its own first cycles. (2) Cross-validation folds are grouped by engine, so no engine appears in both training and validation. (3) Regime normalization uses training statistics only. (4) NASA's test answers were used once, for final scoring. (5) The simulator replays out-of-fold predictions, never in-sample ones.

**What is the NASA score?** The PHM08 scoring function. It's exponential in the error and penalizes late predictions (predicting more life than remains) harder than early ones, because a late call means an in-flight failure.

**Why conformal prediction?** It gives intervals with a finite-sample coverage guarantee and no distributional assumptions. I used Mondrian (group-conditional) conformal by operating profile and predicted-RUL band, calibrated on out-of-fold residuals. Coverage is close to nominal in the decision zone and too narrow only above 100 cycles, where both prediction and truth sit at the cap.

**Explain the fault-mode discovery.** Each engine's end-of-life signature (last 20 cycles minus first 30, across 14 sensors) went through PCA, then Gaussian mixtures with 1–4 components. BIC improves massively from 1 to 2 groups for FD003/FD004 (−492, −1118) and gets worse for FD001/FD002 (+20, +35). Hartigan's dip test agrees (p < 0.001 vs p = 0.85). I also ran a parametric-bootstrap LRT, but it rejected even the single-mode subsets, because it tests Gaussianity rather than the number of modes. That's why I relied on BIC and the dip test.

**Why did H2 come back null, and why keep it?** Flying in six regimes didn't change engine life (log-rank p = 0.87). It changed what the sensors read. That told me to put the effort into regime normalization rather than a regime-specific life model, and to use one life distribution in the simulator. A null result that changes the design is still a result.

**What does the simulator assume, and what drives the result?** See the GR&A table. The readiness gain comes almost entirely from GR-07: planned depot visits are shorter (75 vs 120 days) because an engine pulled before failure has less secondary damage. The sensitivity analysis shows that if planned visits are no shorter, the readiness gain drops to about zero, though the cost saving remains. That's the first number I'd measure on a real program.

**Why does hard-time removal do worse than run to failure on readiness?** It pulls healthy engines and wastes about a third of their life. The extra removals load the depot and drain the spare pool, and forced removals without a spare ground the jet. The textbook age-replacement math assumes unlimited spares and depot capacity; the simulator doesn't.

**How did you validate the simulator?** Steady state: 365- and 900-day warm-ups agree. Determinism: re-running reproduces results exactly. Cross-implementation: the JavaScript port matches the Python/GPU version within 0.04 MC points on four reference configurations (all |z| < 1). Calibration: the non-engine break rate was set by bisection so the status quo sits at 60% MC, inside GAO's 44–67% fleet range.

**Explain METRIC / Palm's theorem.** If failures are Poisson and repair times are independent, the number of units in repair is Poisson with mean demand rate × mean repair time, whatever the repair-time distribution (Palm's theorem). Stocking s spares, expected backorders are EBO(s) = Σ (x − s) P(X = x) over x > s. Aircraft availability is approximately the product over items of (1 − EBO/N). Maximizing log availability under a budget is separable, so it becomes a multiple-choice knapsack, which I solved as a MILP and cross-checked with greedy marginal analysis.

**Why Spark for 624k rows?** Honestly, pandas handles 624k rows. The point was to write the ETL the way it would run on a Hadoop/Spark cluster at fleet scale: multi-line CSV parsing, window functions, Spark ML. The same job runs unchanged with `spark-submit --master yarn`.

**How did you use AI tools?** I used an AI coding agent as a pair programmer for drafting code. I owned the questions, the methods and the ground rules, and I didn't trust any output until it passed a check: the reconciliations, ablations and cross-validations listed in section 12. Be ready to walk through any file line by line. If something is unclear, re-run that stage and read its output before the interview.

**What would you do with real F-35 data?** Replace C-MAPSS with fleet sensor and maintenance history, which means censoring (engines removed for other reasons). So: survival models (Cox, Weibull AFT) or deep survival models, Bayesian updating as each flight arrives, the real planned vs. unscheduled turnaround and cost, multi-echelon spares (VARI-METRIC), and a shadow-mode pilot before acting on predictions.

## Sustainment vocabulary

- **MC (mission capable):** share of time an aircraft can do at least one tasked mission. **FMC:** all missions.
- **NMCM / NMCS:** not mission capable for maintenance / for supply (waiting on a part).
- **PBL (performance-based logistics):** a contract that pays for outcomes (availability, MC rate) rather than parts and labor.
- **TAT:** depot turnaround time. **LRU:** line-replaceable unit, swapped on the flight line.
- **RUL:** remaining useful life. **PHM:** prognostics and health management.
- **GR&A:** ground rules and assumptions. **AoA:** analysis of alternatives. **COA:** course of action. **BLUF:** bottom line up front.
- **JASC / ATA chapter:** standard aircraft system codes (72 = turbine engine, 32 = landing gear, …).

## Resume bullets (pick 2–3)

- Built an end-to-end predictive sustainment study on public data (Ready Line): PySpark ETL of 624K FAA maintenance reports and 1,416 NASA turbofan engines into a DuckDB warehouse; published as an interactive website.
- Predicted jet-engine remaining useful life with a LightGBM + 1D-CNN ensemble (PyTorch on GPU, engine-grouped CV, conformal intervals), reaching 11.1-cycle RMSE on NASA's FD001 test set and 4.7 cycles in the maintenance decision zone.
- Simulated 24-jet squadron readiness with a GPU-vectorized Monte Carlo that replays the model's real predictions; predictive removal raised mission-capable rate from 59.9% to 63.8% and cut unscheduled engine removals 98%.
- Optimized spares stocking and depot induction with Pyomo/HiGHS MILPs: matched a rule-of-thumb plan's availability for 35% less, and cut realized in-service failures from 51 to 33 against withheld ground truth.
- Text-mined 624K maintenance write-ups: 89.9%-accurate system auto-coding (39 classes, temporal holdout), an NMF failure-mode taxonomy, and a risk-weighted root-cause Pareto.
