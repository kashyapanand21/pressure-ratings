
# T20 Pressure Ratings

A ball-by-ball machine learning pipeline that defines **pressure** in T20 cricket and ranks batters and bowlers by how they perform when the game is on the line. It uses IPL and T20 International data from 2022 to 2026.

Built for Round 1 of the SupeRR Selector Season 2 challenge (Rajasthan Royals × Unstop): *identify the most reliable players under pressure*.

> **Data not included.** The competition data pack is for competition use only and must not be shared or published, so this repository contains code only. To run the pipeline, place the data pack files in `data/` exactly as supplied.

---

## The question

Some players lift when a match is tight; others fade. Raw strike rates and economy rates can't tell them apart, because a strike rate of 140 means something very different in the powerplay and in a death-over chase. This project:

1. Defines which balls are genuinely high-pressure.
2. Judges every ball against what an average player would do in exactly that situation.
3. Separates real skill from luck in small samples.
4. Tests whether the rankings hold up.

## Approach

### 1. Defining pressure

Pressure is measured differently in each innings, because the situations are different.

| Situation               | Pressure means    | How it is measured                                                                             |
| ----------------------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| **Chasing**       | Stakes            | How far one ball can swing the result: win chance after a four minus win chance after a wicket |
| **Batting first** | Trouble           | How far the projected total is below par (the median first-innings total for that competition) |
| **Bowling first** | Mirror of trouble | Bowlers are under pressure when the batting side is *ahead* of par                           |

- The **top 25% of balls**, ranked within each innings and competition, count as high pressure. Ranking this way stops the toss or the IPL's higher scoring from deciding who faces pressure.
- Five candidate definitions were compared against situations everyone agrees on: collapses, teams cruising, close finishes, and matches already decided. The chosen pair were the only ones to pass.
- **Self-made pressure is removed.** A batter who blocks for 20 balls drags his own team behind par, which would label his own later balls "pressure". First-innings balls are therefore re-labelled as if that player had been exactly average so far in the match.

### 2. Models

| Model                        | Type                                                      | Purpose                                                                                                                 |
| ---------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Win chance                   | Logistic regression with interaction terms                | Chase pressure; compared against plain logistic regression and gradient boosting                                        |
| Runs still to come           | Gradient-boosted regression (monotonic constraints)       | First-innings projection, par and the run cost of a wicket                                                              |
| Expected runs and dismissals | Gradient boosting (Poisson regression and classification) | What an average player does in the exact situation: over, wickets, run rates, competition, pace or spin, pressure level |

- Every model is evaluated with **match-grouped cross-validation**, so balls from one match never sit on both sides of a split.
- Every model also gets a **future test**: trained on 2022–25 and scored on 2026.
- Ball-level predictions are **out-of-fold**: each ball is scored by a model that never saw its match.

### 3. Rating players

- **Impact per ball** = runs above expected, with wickets valued in runs. A wicket's value comes from the projection model (about 10.5 runs in the powerplay, about 3 at the death).
- **Shrinkage (empirical Bayes):** each player's estimate starts from his own normal level and moves only as far as his pressure record justifies. How much to trust a sample is estimated from the data, not chosen by hand.
- **Rating** = 0.5 × skill + 0.25 × resilience + 0.25 × consistency (as z-scores):
  - **Skill:** estimated performance level under pressure.
  - **Resilience:** change from his normal level under pressure.
  - **Consistency:** share of pressure matches above expected.
- **Eligibility:** a minimum of 100 high-pressure balls. Associate-nation matches and run-outs are excluded.

### 4. Reliability checks

- **Future test:** rank players on 2022–25, then check against 2026 pressure performance. The same comparison is run for simple baselines (death-overs strike rate or economy, overall strike rate or economy).
- **Odd/even-match check:** rate players separately on odd- and even-numbered matches, then compare the two rankings.
- **Bootstrap:** 500 resamples of matches, reporting how often each player lands in the top 10.
- **Sensitivity:** how many of the top 10 survive changes to the pressure threshold (70% / 80%), the weights, the minimum balls, and the rating method.

## Project structure

```
src/
  config.py          paths and shared settings (threshold, minimum balls, weights)
  features.py        feature engineering shared by all models
  build_state.py     cleans ball-by-ball data and builds match state for every ball
  train_winprob.py   win-chance models: comparison, calibration, out-of-fold predictions
  inn1_total.py      runs-still-to-come model, total-to-win model, par
  pressure.py        pressure definitions, scorecard comparison, chosen labels
  rating_lib.py      shrinkage and rating maths (player-by-match aggregation)
  ratings.py         expected-outcome models, impact per ball, player ratings
  reliability.py     future test, odd/even check, bootstrap, sensitivity
data/                competition data pack (not included)
work/                intermediate files (generated)
models/              saved models (generated)
reports/figures/     charts (generated)
```

## How to run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Place the data pack in `data/`, then run the steps in order from the project root:

```bash
python src/build_state.py
python src/train_winprob.py
python src/inn1_total.py
python src/pressure.py
python src/ratings.py
python src/reliability.py
```

The full run takes a few minutes on a laptop.

**Outputs**

- `work/decision_batters.csv`, `work/decision_bowlers.csv`: final rankings with bootstrap and sensitivity columns.
- `work/ratings_batters.csv`, `work/ratings_bowlers.csv`: full ratings for all eligible players.
- `reports/figures/`: calibration, pressure proof, bootstrap and future-test charts.

Settings such as the pressure threshold, the minimum balls and the rating weights live in `src/config.py`.

## Limitations

- Team totals are approximate (about ±2.5 runs).
- Par is set per competition, not per venue, so a slow pitch can look like pressure.
- In chases, the wicket value comes from a first-innings model.
- A few innings end early, probably because balls are missing from the data.
- Samples per player are small, so shrinkage is strong.
- Pressure-specific performance is noisy: no method, including simple baselines, strongly predicts the following season.

## Author

Anand Kashyap
