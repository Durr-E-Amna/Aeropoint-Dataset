import pandas as pd
import glob

INPUT_PATTERN = "gesture_data_*.csv"
OUTPUT_FILE = "master_gesture_data.csv"

files = sorted(glob.glob(INPUT_PATTERN))

if not files:
    print(f"No files found matching '{INPUT_PATTERN}'.")
    exit()

print(f"Merging {len(files)} file(s):")
for f in files:
    print(f"  - {f}")

dataframes = [pd.read_csv(f) for f in files]
combined = pd.concat(dataframes, ignore_index=True)

before = len(combined)
combined = combined.drop_duplicates()
after = len(combined)
if before != after:
    print(f"Removed {before - after} duplicate rows.")

combined.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved {OUTPUT_FILE}")
print(f"Total samples: {len(combined)}")
print(f"Total contributors: {combined['person_id'].nunique()}")
print(f"Total classes: {combined['label'].nunique()}")

print("\nSamples per person per gesture:")
print(combined.groupby(["person_id", "label"]).size().unstack(fill_value=0))

target = 7 * 400 * 10
print(f"\nTarget: ~{target:,} samples")
print(f"Current: {len(combined):,} ({len(combined)/target*100:.1f}%)")
