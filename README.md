# AeroPoint — Gesture Dataset Collection

Scripts for collecting the multi-user hand-gesture dataset (Objective 1).
Each team member runs collection sessions independently on their own
laptop, then everyone's data gets merged into one master dataset.

## One-time setup (each team member does this once)

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd aeropoint-dataset
   ```

2. Create your own virtual environment (do NOT commit this folder):
   ```bash
   python -m venv venv
   ```

3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Install the exact same package versions as everyone else:
   ```bash
   pip install -r requirements.txt
   ```

5. Confirm your webcam and hand tracking work:
   ```bash
   python 01_test_camera.py
   ```
   You should see a window with green dots tracking your hand. Press `q` to quit.

## Collecting data (repeat for each person you record)

Agreed initials for this team:
- Durr E Amna → `DA`
- Rabiya Jamil → `RJ`
- Rafey Riaz Qazi → `RQ`

Run, incrementing `--person` for each new contributor you personally record:

```bash
python 02_collect_dataset_parallel.py --collector DA --person 1
python 02_collect_dataset_parallel.py --collector DA --person 2
```

This produces a file like `gesture_data_DA_p1.csv` in your local folder.

**Before you record anyone, agree as a team on lighting variation** — don't
all three of you collect under similar lighting by accident. One person
should use a bright room, one a dim room, one near natural window light.

## Sharing your data with the team

Once a session is done:

```bash
git add gesture_data_DA_p1.csv
git commit -m "Add dataset: DA collector, person 1"
git pull
git push
```

Always `git pull` before you push, so you get everyone else's latest
files and don't create a merge conflict on unrelated CSV files.

## Merging everyone's data into one dataset

Once all sessions are pushed and everyone has pulled the latest:

```bash
git pull
python 04_merge_datasets.py
```

This creates `master_gesture_data.csv` locally (this file is intentionally
git-ignored, everyone regenerates it themselves rather than fighting over
one shared copy) and prints a breakdown of samples per person per gesture,
plus a percentage progress against the ~28,000 sample target.

## File overview

| File | Purpose |
|---|---|
| `01_test_camera.py` | Sanity check — confirms webcam + hand tracking work |
| `02_collect_dataset_parallel.py` | Main collection script, run once per person |
| `03_train_quick_check.py` | Quick accuracy gut-check on early data |
| `04_merge_datasets.py` | Combines everyone's CSVs into one master file |
| `requirements.txt` | Exact package versions, install with pip |

## Target

7 gesture classes (6 poses + Neutral) × ~400 samples per person × 9-10
contributors ≈ 28,000 labelled landmark samples, split 70/15/15 by
person for training, validation, and testing.
