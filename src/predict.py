"""Command-line predictor for the trained IPL match-winner model.

Reads the exported model artifact (docs/model.json) and prints calibrated
win probabilities for a hypothetical fixture.

Usage:
    python src/predict.py --team-a "Chennai Super Kings" \
                          --team-b "Mumbai Indians" \
                          --city Chennai \
                          --toss-winner "Chennai Super Kings" \
                          --toss-decision bat
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_features(model: dict, team_a: str, team_b: str, city: str,
                   toss_winner: str, toss_decision: str) -> dict:
    stats = model["team_stats"]
    h2h = model["h2h_wins"]
    homes = model["team_home_cities"]

    a_wins = h2h.get(team_a, {}).get(team_b, 0)
    b_wins = h2h.get(team_b, {}).get(team_a, 0)
    pair_total = a_wins + b_wins

    return {
        "win_rate_diff": stats[team_a]["win_rate"] - stats[team_b]["win_rate"],
        "form_diff": stats[team_a]["form"] - stats[team_b]["form"],
        "h2h_diff": (a_wins - b_wins) / pair_total if pair_total else 0.0,
        "toss_won": 1.0 if toss_winner == team_a else -1.0,
        "bats_first": 1.0 if (toss_winner == team_a) == (toss_decision == "bat") else -1.0,
        "home_advantage": float(city in homes.get(team_a, []))
                          - float(city in homes.get(team_b, [])),
    }


def predict(model: dict, features: dict) -> float:
    z = sum(model["coefficients"][name] * value for name, value in features.items())
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("docs/model.json"))
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--city", default="")
    parser.add_argument("--toss-winner", required=True)
    parser.add_argument("--toss-decision", choices=["bat", "field"], required=True)
    args = parser.parse_args()

    model = json.loads(args.model.read_text())
    for team in (args.team_a, args.team_b):
        if team not in model["teams"]:
            parser.error(f"Unknown team {team!r}. Known teams:\n  "
                         + "\n  ".join(model["teams"]))
    if args.toss_winner not in (args.team_a, args.team_b):
        parser.error("--toss-winner must be one of the two playing teams")

    features = build_features(model, args.team_a, args.team_b, args.city,
                              args.toss_winner, args.toss_decision)
    p_a = predict(model, features)

    print(f"\n{args.team_a} vs {args.team_b}"
          f" at {args.city or 'neutral venue'}")
    print(f"Toss: {args.toss_winner} chose to {args.toss_decision}\n")
    print(f"  P({args.team_a} wins) = {p_a:.1%}")
    print(f"  P({args.team_b} wins) = {1 - p_a:.1%}\n")
    print("Feature contributions (log-odds):")
    for name, value in features.items():
        contrib = model["coefficients"][name] * value
        print(f"  {name:<16} value={value:+.3f}  contribution={contrib:+.4f}")


if __name__ == "__main__":
    main()
