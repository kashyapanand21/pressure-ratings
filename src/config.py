from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
for p in (WORK, MODELS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_YEAR = 2026
N_FOLDS = 5
HIGH = 0.75  # top 25% of balls = high pressure

# Player ratings
MIN_BALLS = 100        # high-pressure balls needed to be ranked
MIN_MATCH_BALLS = 3    # pressure balls in a match for it to count towards consistency
WEIGHTS = {"skill": 0.5, "resilience": 0.25, "consistency": 0.25}