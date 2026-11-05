# MATH840 final project: the LGBT model (Light Gradient Boosting Transformer)

Andrii Boichuk. Hourly forecast of `y` for 2026-10-27 01:00 to 2026-10-29 00:00 (48 h) with a 95% interval.

**Model.** The mean of four tree models (LightGBM on `y`, LightGBM on `log(1+y)`, LightGBM on the ratio to the last-week mean with recency weights, XGBoost on `y`; no `cloud_cover`) blended 0.3 / 0.7 with Chronos-2 reading the weather (contexts of 1024 and 2048 h, averaged). Night-hour forecasts are multiplied by 0.85 to lower MAPE.
**Interval.** The 95th percentile of the absolute and relative errors of the 60 recent backtest origins, in 4 groups (night or day × day 1 or day 2), mean of the two, times 1.2. The lower bound is cut at 0.
**Submitted file.** `submissions/Boichuk_project_final.csv`: the Session 1 points on day 1 (night × 0.85), the average of the Session 1 and latest files on day 2, and the interval of the latest file. Winkler depends only on the interval and the actual values, so the points and the interval can come from different files.

## Files

| Path | What it is |
|---|---|
| `notebooks/01_final_model_generation.ipynb` | Generates the latest file (stage 2): backtest on 120 origins, interval recipe search, final fit, CSV. Colab, asks for `train.csv` and `weather_forecast.csv`. |
| `notebooks/02_full_pipeline_colab.ipynb` | End-to-end Colab pipeline: embeds the real Session 1 file (stage 1), runs notebook 01 (stage 2), combines them into the final file (stage 3), plots. |
| `notebooks/experiments/session1_experiments_v3.ipynb` | Experiments behind the Session 1 file (tree models without `cloud_cover`, pretrained models). |
| `scripts/combine_s1_points_latest_ci.py` | Standalone script for the final mix (Session 1 points, latest interval). |
| `scripts/residual_diagnostics_cell.py` | Residual diagnostics cell (Ljung-Box, ACF, QQ, bias by hour); paste it at the end of notebook 01 after the backtest cells. |
| `submissions/` | `Boichuk_project_final.csv` (final), `session1_Boichuk_project.csv` (Session 1), `latest_stage2_project.csv` (stage 2). |
| `data/` | `train.csv`, `weather_forecast.csv` as given in the course. |

## How to run

1. Open `notebooks/02_full_pipeline_colab.ipynb` in Google Colab (a GPU runtime makes Chronos-2 faster) and upload `data/train.csv` and `data/weather_forecast.csv` when asked.
2. Run all cells. The last stage writes and downloads the final CSV. Rename it to the name the template asks for.
3. To regenerate only the latest file, run `notebooks/01_final_model_generation.ipynb` instead.

Chronos-2 and the tree libraries can give slightly different numbers on other hardware or library versions.

## Main results (from the notebooks)

- Backtest skill (RMSE ÷ RMSE of the daily seasonal naive, 120 origins): trees 0.72, Chronos-2 with weather 0.66, final blend 0.65.
- Public hours 1 to 24: the latest file has RMSE 98.4, MAPE 25.5%, Winkler about 570 (the benchmark: 115.7, 84.3%, 521).
- Settings (blend weight, night factor, interval recipe) were chosen on the same 120 origins they are scored on, so the backtest is somewhat optimistic. The weather noise in the backtest is simulated and its size is an assumption.
- `cloud_cover` is left out of every final model: it behaves differently in `train.csv` and in `weather_forecast.csv`, and differently from the brief.

## Use of AI (LLM)

We used Claude (Anthropic) as a coding and analysis assistant for the whole project.

- **What Claude did:** wrote the notebooks (exploratory analysis, tree models, Chronos-2 and other pretrained-model runs, rolling backtest, interval and MAPE tests), fixed bugs we hit when running them, suggested the experiments, and drafted the slides and the report.
- **What I did:** ran every notebook in Colab, read the outputs, chose which variants to keep, decided what to submit and when, and checked the leaderboard results.
- **Data rule:** only `train.csv` and `weather_forecast.csv` were used, and I did not look for the origin of the series.
- The figures in the report are taken from the notebook outputs. Where a number is an estimate (for example the Session 1 RMSE of about 57, worked back from the leaderboard), the text says so.

## Report and slides

The report and the slides are in the course submission (PDF); add the links or files here if the course asks for them.
