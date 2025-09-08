# src/create_data.py
import pandas as pd
import numpy as np
np.random.seed(42)

N = 2000
data = {
    "sleep_hours": np.random.normal(7, 1.2, N).clip(3, 10),
    "work_hours": np.random.normal(8, 2.5, N).clip(0, 16),
    "exercise_hours": np.random.normal(1.0, 0.8, N).clip(0, 4),
    "screen_time": np.random.normal(5, 2.0, N).clip(0, 16),
    "water_intake": np.random.normal(2.5, 0.9, N).clip(0.2, 6),
    "mood_rating": np.random.randint(1, 11, N)
}

df = pd.DataFrame(data)

# Simple rule to synthesize stress label (example rule; tweak as needed)
def compute_stress(row):
    score = (row.work_hours + 0.8*row.screen_time + (10 - row.mood_rating)*0.7) \
            - (row.sleep_hours*1.2 + row.exercise_hours*1.5 + row.water_intake*0.8)
    if score < 2.5:
        return "Low"
    elif score < 7:
        return "Medium"
    else:
        return "High"

df["stress_level"] = df.apply(compute_stress, axis=1)
df.to_csv("data/stress_data.csv", index=False)
print("Saved data/stress_data.csv  shape:", df.shape)
