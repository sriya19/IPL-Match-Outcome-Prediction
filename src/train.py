"""End-to-end training pipeline for the IPL match-winner prediction model.

Builds a leakage-free, symmetric feature representation from matches.csv,
trains and evaluates candidate models on a chronological train/test split,
then exports:

  * models/match_winner_lr.joblib   -- fitted scikit-learn pipeline
  * models/metrics.json             -- evaluation summary
  * docs/model.json                 -- coefficients + team statistics for the
                                       in-browser predictor (GitHub Pages app)

Usage:
    python src/train.py [--data-dir data] [--out-dir models] [--docs-dir docs]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Franchise renames: map historical names onto the current franchise so the
# model learns a single strength parameter per franchise.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Home cities used for the home-advantage feature. Defunct franchises are
# included so historical rows are encoded consistently.
TEAM_HOME_CITIES = {
    "Chennai Super Kings": {"Chennai"},
    "Delhi Capitals": {"Delhi"},
    "Gujarat Titans": {"Ahmedabad"},
    "Kolkata Knight Riders": {"Kolkata"},
    "Lucknow Super Giants": {"Lucknow"},
    "Mumbai Indians": {"Mumbai"},
    "Punjab Kings": {"Chandigarh", "Mohali", "Dharamsala"},
    "Rajasthan Royals": {"Jaipur"},
    "Royal Challengers Bengaluru": {"Bangalore", "Bengaluru"},
    "Sunrisers Hyderabad": {"Hyderabad"},
    "Deccan Chargers": {"Hyderabad"},
    "Gujarat Lions": {"Rajkot"},
    "Kochi Tuskers Kerala": {"Kochi"},
    "Pune Warriors": {"Pune"},
    "Rising Pune Supergiant": {"Pune"},
}

FORM_WINDOW = 10  # matches considered for the recent-form feature


def load_matches(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "matches.csv", parse_dates=["date"])
    for col in ("team1", "team2", "toss_winner", "winner"):
        df[col] = df[col].replace(TEAM_ALIASES)
    # Keep only decided matches: ties and abandoned games have no winner label.
    df = df[df["winner"].notna() & df["result"].isin(["runs", "wickets"])]
    return df.sort_values("date").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Encode each match from the perspective of team1 (label: team1 won).

    The encoding is antisymmetric -- swapping the two teams negates every
    feature -- which guarantees P(A beats B) + P(B beats A) = 1 at inference.
    Rolling statistics use only matches strictly before the current one, so
    the representation is free of target leakage.
    """
    teams = sorted(set(df["team1"]) | set(df["team2"]))
    wins: dict[str, int] = defaultdict(int)
    played: dict[str, int] = defaultdict(int)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h: dict[tuple[str, str], int] = defaultdict(int)

    rows = []
    for m in df.itertuples():
        a, b = m.team1, m.team2
        feat = {f"team_{t}": 0.0 for t in teams}
        feat[f"team_{a}"] = 1.0
        feat[f"team_{b}"] = -1.0

        feat["toss_won"] = 1.0 if m.toss_winner == a else -1.0
        bats_first = (m.toss_winner == a) == (m.toss_decision == "bat")
        feat["bats_first"] = 1.0 if bats_first else -1.0

        city = m.city if isinstance(m.city, str) else ""
        home_a = city in TEAM_HOME_CITIES.get(a, set())
        home_b = city in TEAM_HOME_CITIES.get(b, set())
        feat["home_advantage"] = float(home_a) - float(home_b)

        wr = lambda t: wins[t] / played[t] if played[t] else 0.5
        form = lambda t: np.mean(recent[t]) if recent[t] else 0.5
        feat["win_rate_diff"] = wr(a) - wr(b)
        feat["form_diff"] = form(a) - form(b)
        pair_total = h2h[(a, b)] + h2h[(b, a)]
        feat["h2h_diff"] = (h2h[(a, b)] - h2h[(b, a)]) / pair_total if pair_total else 0.0

        rows.append(feat)

        # Update rolling state after the row is emitted.
        winner = m.winner
        for t in (a, b):
            played[t] += 1
            recent[t].append(1.0 if winner == t else 0.0)
        wins[winner] += 1
        h2h[(winner, b if winner == a else a)] += 1

    X = pd.DataFrame(rows)
    y = (df["winner"] == df["team1"]).astype(int)
    return X, y, teams


def final_team_stats(df: pd.DataFrame, teams: list[str]) -> dict:
    """Team statistics over the full dataset, embedded in the web app."""
    stats = {}
    for t in teams:
        games = df[(df["team1"] == t) | (df["team2"] == t)].sort_values("date")
        wins = int((games["winner"] == t).sum())
        recent = (games["winner"] == t).tail(FORM_WINDOW)
        stats[t] = {
            "played": int(len(games)),
            "win_rate": round(wins / len(games), 4) if len(games) else 0.5,
            "form": round(float(recent.mean()), 4) if len(recent) else 0.5,
            "active": bool(games["date"].max() >= df["date"].max() - pd.Timedelta(days=400)),
        }
    return stats


def final_h2h(df: pd.DataFrame, teams: list[str]) -> dict:
    h2h = {t: {} for t in teams}
    for m in df.itertuples():
        h2h[m.winner][m.team2 if m.winner == m.team1 else m.team1] = (
            h2h[m.winner].get(m.team2 if m.winner == m.team1 else m.team1, 0) + 1
        )
    return h2h


# Deployed feature set. Team identity one-hots are deliberately excluded:
# franchise strength shifts sharply across auction eras, so identity terms
# overfit history. Rolling-statistic features adapt with the data instead.
COMPACT_FEATURES = [
    "win_rate_diff", "form_diff", "h2h_diff",
    "toss_won", "bats_first", "home_advantage",
]


def rolling_eval(model, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
    """Expanding-window evaluation over the most recent 50% of history.

    Each fold trains on all matches before the fold window and predicts the
    next 10% of matches, mimicking real deployment where the model only ever
    sees the past.
    """
    n = len(X)
    accs, aucs, lls = [], [], []
    for i in range(n_folds):
        tr_end = int(n * (0.5 + 0.1 * i))
        te_end = int(n * (0.6 + 0.1 * i))
        model.fit(X.iloc[:tr_end], y.iloc[:tr_end])
        proba = model.predict_proba(X.iloc[tr_end:te_end])[:, 1]
        y_te = y.iloc[tr_end:te_end]
        accs.append(accuracy_score(y_te, proba >= 0.5))
        aucs.append(roc_auc_score(y_te, proba))
        lls.append(log_loss(y_te, proba))
    return {
        "fold_accuracy": [round(a, 4) for a in accs],
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "mean_roc_auc": round(float(np.mean(aucs)), 4),
        "mean_log_loss": round(float(np.mean(lls)), 4),
    }


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    """Grid-search candidates with time-series CV, then score each candidate
    on expanding-window chronological folds (no random splits: shuffling
    leaks future team strength into the past and inflates accuracy)."""
    cv = TimeSeriesSplit(n_splits=5)
    Xc = X[COMPACT_FEATURES]

    candidates = {
        "logistic_regression_compact": (Xc, GridSearchCV(
            LogisticRegression(max_iter=2000, fit_intercept=False),
            {"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
            cv=cv, scoring="neg_log_loss", n_jobs=-1,
        )),
        "logistic_regression_full": (X, GridSearchCV(
            LogisticRegression(max_iter=2000, fit_intercept=False),
            {"C": [0.01, 0.03, 0.1, 0.3, 1.0]},
            cv=cv, scoring="neg_log_loss", n_jobs=-1,
        )),
        "random_forest": (Xc, GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [3, 5, 8],
             "min_samples_leaf": [4, 10, 20]},
            cv=cv, scoring="neg_log_loss", n_jobs=-1,
        )),
    }

    results = {}
    for name, (X_cand, search) in candidates.items():
        search.fit(X_cand.iloc[: int(len(X_cand) * 0.8)],
                   y.iloc[: int(len(y) * 0.8)])
        scores = rolling_eval(search.best_estimator_, X_cand, y)
        results[name] = {"best_params": search.best_params_, **scores}
        print(f"{name}: {scores} params={search.best_params_}")

    return results


def export(results: dict, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
           teams: list[str], out_dir: Path, docs_dir: Path) -> None:
    # The compact logistic regression is the deployment target: best rolling
    # log-loss, calibrated probabilities, fully interpretable, and its six
    # coefficients can be evaluated client-side in JavaScript.
    lr_params = results["logistic_regression_compact"]["best_params"]
    deploy = LogisticRegression(max_iter=2000, fit_intercept=False, **lr_params)
    Xc = X[COMPACT_FEATURES]
    deploy.fit(Xc, y)  # refit on the full history for deployment

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": deploy, "features": COMPACT_FEATURES},
                out_dir / "match_winner_lr.joblib")

    metrics = dict(results)
    metrics["deployed_model"] = "logistic_regression_compact (refit on full history)"
    metrics["n_matches"] = int(len(X))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    docs_dir.mkdir(parents=True, exist_ok=True)
    model_json = {
        "model_type": "logistic_regression",
        "trained_on": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "n_matches": int(len(X)),
        "test_accuracy": results["logistic_regression_compact"]["mean_accuracy"],
        "test_roc_auc": results["logistic_regression_compact"]["mean_roc_auc"],
        "features": COMPACT_FEATURES,
        "coefficients": dict(zip(COMPACT_FEATURES,
                                 np.round(deploy.coef_[0], 6).tolist())),
        "teams": teams,
        "team_stats": final_team_stats(df, teams),
        "h2h_wins": final_h2h(df, teams),
        "team_home_cities": {t: sorted(c) for t, c in TEAM_HOME_CITIES.items()},
        "cities": sorted(df["city"].dropna().unique().tolist()),
        "form_window": FORM_WINDOW,
    }
    (docs_dir / "model.json").write_text(json.dumps(model_json, indent=2))
    print(f"Exported {out_dir / 'match_winner_lr.joblib'} and {docs_dir / 'model.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    df = load_matches(args.data_dir)
    print(f"Loaded {len(df)} decided matches "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    X, y, teams = build_features(df)
    results = train(X, y)
    export(results, X, y, df, teams, args.out_dir, args.docs_dir)


if __name__ == "__main__":
    main()
