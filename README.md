# IPL Match Outcome Prediction

Machine-learning analysis and a deployable match-winner prediction model for the
Indian Premier League (IPL), built on every match from **2008 to 2024**
(1,095 matches, 260k+ ball-by-ball deliveries).

> **AIT 582 — Team 4:** Sadiya Puspo · Sai Padma Sriya Pothula · Adi Mohan

## 🔮 Try the model

**Live predictor:** <https://sriya19.github.io/IPL-Match-Outcome-Prediction/>

Pick two teams, the venue city, and the toss result — the page returns calibrated
win probabilities with a full feature-contribution breakdown. Inference runs
entirely in your browser against the exported model coefficients
([`docs/model.json`](docs/model.json)); no server required.

The same model is available from the command line:

```bash
python src/predict.py \
    --team-a "Chennai Super Kings" \
    --team-b "Mumbai Indians" \
    --city Chennai \
    --toss-winner "Chennai Super Kings" \
    --toss-decision bat
```

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Kaggle [IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) — `matches.csv`, `deliveries.csv` |
| `notebooks/` | Original research notebook (EDA + the three research questions) |
| `reports/` | Final paper (PDF) and presentation (PPTX) |
| `src/train.py` | End-to-end training pipeline: feature engineering, model selection, evaluation, artifact export |
| `src/predict.py` | CLI predictor using the exported model |
| `models/` | Fitted scikit-learn model (`.joblib`) and evaluation metrics |
| `docs/` | GitHub Pages app (`index.html`) + exported model (`model.json`) |

## Research questions (paper & notebook)

1. **Does winning the toss affect the match outcome?** — Toss advantage is
   marginal; venue/team context dominates.
2. **Does powerplay performance determine the outcome?** — Powerplay runs and
   wickets carry meaningful in-match signal.
3. **Which individual performance metrics drive Player-of-the-Match awards?** —
   Batting impact (runs, strike rate) dominates over bowling economy.

## The deployed prediction model

### Feature engineering

Each match is encoded **from team A's perspective** with an antisymmetric
representation (swapping the teams negates every feature), which guarantees
`P(A beats B) + P(B beats A) = 1`:

| Feature | Description |
|---|---|
| `win_rate_diff` | Difference in career win rate (computed only from matches *before* the one being predicted — no leakage) |
| `form_diff` | Difference in win rate over each team's last 10 matches |
| `h2h_diff` | Normalized head-to-head record between the two teams |
| `toss_won` | +1 if team A won the toss, −1 otherwise |
| `bats_first` | +1 if team A bats first (derived from toss winner + decision) |
| `home_advantage` | +1/−1/0 from a franchise → home-city mapping |

Franchise renames (Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab
Kings, RCB Bangalore → Bengaluru, etc.) are normalized so each franchise has a
single history. Team-identity one-hots were evaluated and **rejected**: squad
strength resets at every mega-auction, so identity terms overfit history, while
rolling statistics adapt with the data.

### Model selection & honest evaluation

Candidates (regularized logistic regression on the compact feature set, on the
full set with team one-hots, and a random forest) are tuned with
`TimeSeriesSplit` cross-validation and compared on **expanding-window
chronological backtests** — train on the past, predict the next block of
matches, exactly as the model would be used in production. Random train/test
splits are deliberately avoided because they leak future team strength into the
past and inflate accuracy.

| Model | Rolling accuracy | ROC-AUC | Log-loss |
|---|---|---|---|
| **Logistic regression (compact) — deployed** | 0.543 | **0.552** | **0.691** |
| Logistic regression (+ team one-hots) | 0.550 | 0.541 | 0.694 |
| Random forest | 0.526 | 0.547 | 0.694 |

The compact logistic regression is deployed: best calibrated log-loss (the only
candidate below the 0.693 coin-flip baseline), fully interpretable, and small
enough to evaluate client-side in JavaScript.

**A note on the numbers:** pre-match IPL outcomes are genuinely close to a coin
flip — ~54–55% accuracy with sub-baseline log-loss is the realistic ceiling
when using only information available before the first ball. Studies reporting
70–90% typically leak in-match data (powerplay scores, innings totals) or use
shuffled splits. This model reports honest probabilities, not certainties.

### Reproduce

```bash
pip install -r requirements.txt
python src/train.py          # retrains, evaluates, re-exports all artifacts
```

`train.py` writes `models/match_winner_lr.joblib`, `models/metrics.json`, and
`docs/model.json` (consumed by both the web app and `src/predict.py`).

## Deployment

`.github/workflows/deploy-pages.yml` publishes the `docs/` folder to GitHub
Pages on every push. The site is static — model inference is a six-term dot
product and a sigmoid in the browser.

## Data

[IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
by Patrick B, via Kaggle. Ties and abandoned matches are excluded from training.

## License

Apache 2.0 — see [LICENSE](LICENSE).
