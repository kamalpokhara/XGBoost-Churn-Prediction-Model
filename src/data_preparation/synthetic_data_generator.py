# First step
# After a lot of trial and error and other data not being of quality. I have created a synthetic dataset that 
# mimics real-world user behavior on an e-commerce platform. The dataset includes various 
# user archetypes (champions, at-risk, window shoppers, churned, and new users) 
# with realistic session patterns and event probabilities.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_USERS = 10_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

# ── USER ARCHETYPES ───────────────────────────────────────────────────────────
# Each archetype defines realistic behavioral parameters
ARCHETYPES = {
    "champion": {
        "pct": 0.10,  # 10% of users
        "session_count": (30, 80),  # sessions over the year
        "session_gap_days": (1, 7),  # days between sessions
        "events_per_session": (3, 10),
        "p_login": 0.95,  # probability of login per session
        "p_view": 0.90,
        "p_addtocart": 0.10,
        "p_purchase": 0.04,
        "churn_after_day": None,  # never churns
        "dropout_prob": 0.05,  # small chance of early dropout
    },
    "at_risk": {
        "pct": 0.15,
        "session_count": (15, 40),
        "session_gap_days": (5, 20),
        "events_per_session": (2, 6),
        "p_login": 0.80,
        "p_view": 0.85,
        "p_addtocart": 0.05,
        "p_purchase": 0.016,
        "churn_after_day": (180, 270),  # goes silent in second half of year
        "dropout_prob": 0.30,
    },
    "window_shopper": {
        "pct": 0.25,
        "session_count": (10, 30),
        "session_gap_days": (7, 25),
        "events_per_session": (2, 8),
        "p_login": 0.50,
        "p_view": 0.95,
        "p_addtocart": 0.05,
        "p_purchase": 0.002,
        "churn_after_day": None,
        "dropout_prob": 0.20,
    },
    "churned": {
        "pct": 0.30,
        "session_count": (5, 20),
        "session_gap_days": (2, 10),
        "events_per_session": (2, 5),
        "p_login": 0.70,
        "p_view": 0.80,
        "p_addtocart": 0.02,
        "p_purchase": 0.003,
        "churn_after_day": (30, 180),  # goes silent early
        "dropout_prob": 0.90,
    },
    "new_user": {
        "pct": 0.20,
        "session_count": (2, 10),
        "session_gap_days": (10, 40),
        "events_per_session": (1, 4),
        "p_login": 0.60,
        "p_view": 0.80,
        "p_addtocart": 0.03,
        "p_purchase": 0.007,
        "churn_after_day": (270, 365),  # joined late or dropped quickly
        "dropout_prob": 0.50,
    },
}


# ── GENERATOR ─────────────────────────────────────────────────────────────────
def generate_session_events(user_id, session_start, n_events, archetype):
    events = []
    current_time = session_start

    # Login always comes first if it happens
    if np.random.random() < archetype["p_login"]:
        events.append(
            {
                "user_id": user_id,
                "interaction_type": "login",
                "interaction_timestamp": current_time,
            }
        )
        current_time += timedelta(seconds=np.random.randint(5, 30))

    for _ in range(n_events):
        # Realistic within-session event flow: view → addtocart → purchase
        if np.random.random() < archetype["p_view"]:
            events.append(
                {
                    "user_id": user_id,
                    "interaction_type": "view",
                    "interaction_timestamp": current_time,
                }
            )
            current_time += timedelta(seconds=np.random.randint(10, 120))

        if np.random.random() < archetype["p_addtocart"]:
            events.append(
                {
                    "user_id": user_id,
                    "interaction_type": "addtocart",
                    "interaction_timestamp": current_time,
                }
            )
            current_time += timedelta(seconds=np.random.randint(5, 60))

        if np.random.random() < archetype["p_purchase"]:
            events.append(
                {
                    "user_id": user_id,
                    "interaction_type": "purchase",
                    "interaction_timestamp": current_time,
                }
            )
            current_time += timedelta(seconds=np.random.randint(5, 30))

    return events


def generate_user(user_id, archetype_name, archetype):
    all_events = []

    # Registration date — new_users join later in the year
    if archetype_name == "new_user":
        reg_day = np.random.randint(TOTAL_DAYS // 2, TOTAL_DAYS)
    else:
        reg_day = np.random.randint(0, TOTAL_DAYS // 4)

    reg_date = START_DATE + timedelta(days=reg_day)

    # Churn cutoff — after this day, user goes silent
    churn_cutoff = None
    if archetype["churn_after_day"] is not None:
        if np.random.random() < archetype["dropout_prob"]:
            lo, hi = archetype["churn_after_day"]
            churn_day = np.random.randint(lo, hi)
            churn_cutoff = START_DATE + timedelta(days=churn_day)

    # Generate sessions
    n_sessions = np.random.randint(*archetype["session_count"])
    current_date = reg_date

    for _ in range(n_sessions):
        # Stop if past cutoff or end of year
        if current_date > END_DATE:
            break
        if churn_cutoff and current_date > churn_cutoff:
            break

        n_events = np.random.randint(*archetype["events_per_session"])

        # Add realistic hour-of-day noise (peak shopping: 10am-10pm)
        hour = np.random.choice(range(24), p=_hour_weights())
        minute = np.random.randint(0, 60)
        session_start = current_date.replace(hour=hour, minute=minute)

        session_events = generate_session_events(
            user_id, session_start, n_events, archetype
        )
        all_events.extend(session_events)

        # Advance to next session
        gap = np.random.randint(*archetype["session_gap_days"])
        current_date += timedelta(days=gap)

    return all_events


def _hour_weights():
    # Higher probability during daytime/evening shopping hours
    weights = np.ones(24)
    weights[10:22] = 3.0  # 10am–10pm peak
    weights[22:24] = 1.5  # late night
    weights[0:6] = 0.3  # dead hours
    return weights / weights.sum()


# ── ASSIGN ARCHETYPES & GENERATE ──────────────────────────────────────────────
archetype_names = list(ARCHETYPES.keys())
archetype_probs = [ARCHETYPES[a]["pct"] for a in archetype_names]

assigned = np.random.choice(archetype_names, size=N_USERS, p=archetype_probs)

all_events = []
for user_id, archetype_name in enumerate(assigned):
    archetype = ARCHETYPES[archetype_name]
    events = generate_user(user_id, archetype_name, archetype)
    all_events.extend(events)

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
df = pd.DataFrame(all_events)
df = df.sort_values("interaction_timestamp").reset_index(drop=True)
df["interaction_timestamp"] = pd.to_datetime(df["interaction_timestamp"])

# ── SANITY CHECKS ─────────────────────────────────────────────────────────────
print("Generated shape:", df.shape)
print("\nEvent type distribution:")
print(df["interaction_type"].value_counts())
print(f"\nUnique users: {df['user_id'].nunique()}")
print(
    f"Date range: {df['interaction_timestamp'].min().date()} "
    f"→ {df['interaction_timestamp'].max().date()}"
)

# Archetype summary
archetype_df = pd.DataFrame({"user_id": range(N_USERS), "archetype": assigned})
print("\nArchetype distribution:")
print(archetype_df["archetype"].value_counts())

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("data/raw/synthetic_events.csv", index=False)
archetype_df.to_csv("data/raw/synthetic_archetypes.csv", index=False)

print("\nSaved: synthetic_events.csv")
print("Saved: synthetic_archetypes.csv ")
