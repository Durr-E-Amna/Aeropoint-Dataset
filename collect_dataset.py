import cv2
import mediapipe as mp
import pandas as pd
import argparse
import os
import re
import time

SAMPLES_PER_GESTURE = 15
GROUP_SIZE = 5
CAPTURE_COOLDOWN_SEC = 0.4  # ignores repeat key-presses within this window,
                            # so holding SPACE down doesn't count as multiple captures
MAX_CHECKPOINTS = 150       # a full session is 105 captures, so nothing gets
                            # pruned mid-session; oldest beyond this are removed

GESTURES = [
    "point",
    "pinch",
    "fist",
    "open_palm",
    "thumbs_up",
    "peace_sign",
    "neutral",
]

GROUP_LABELS = {
    0: "Group 1: normal position, centered",
    1: "Group 2: hand slightly rotated or tilted",
    2: "Group 3: slightly closer or farther from camera",
}

parser = argparse.ArgumentParser()
parser.add_argument("--collector", required=True)
parser.add_argument("--person", required=True, type=int)
args = parser.parse_args()

COLLECTOR = args.collector.upper()
PERSON_ID = f"{COLLECTOR}_p{args.person}"
OUTPUT_FILE = f"gesture_data_{COLLECTOR}_p{args.person}.csv"
CHECKPOINT_DIR = os.path.join("checkpoints", PERSON_ID)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def extract_row(hand_landmarks, label, person_id, collector):
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    row.extend([label, person_id, collector])
    return row


def column_names():
    cols = []
    for i in range(21):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])
    cols.extend(["label", "person_id", "collector"])
    return cols


def load_existing():
    """Loads any samples already saved for this person, so a session
    can resume exactly where it left off instead of starting over."""
    if not os.path.exists(OUTPUT_FILE):
        return []
    df = pd.read_csv(OUTPUT_FILE)
    return df.values.tolist()


def count_for_gesture(all_rows, gesture):
    return sum(1 for row in all_rows if row[-3] == gesture)


def delete_last_for_gesture(all_rows, gesture):
    """Removes the most recent sample of this gesture wherever it sits in the
    file, so a gesture can still be corrected after moving past it."""
    for i in range(len(all_rows) - 1, -1, -1):
        if all_rows[i][-3] == gesture:
            all_rows.pop(i)
            return True
    return False


# --- checkpoints -----------------------------------------------------------
# Every capture and every delete writes a numbered snapshot of the whole
# dataset into CHECKPOINT_DIR. Undo rewinds one snapshot at a time, and the
# numbering survives restarts because it is read back off disk.

def checkpoint_files():
    """Existing checkpoints for this person, oldest first."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    found = []
    for name in os.listdir(CHECKPOINT_DIR):
        match = re.match(r"^(\d{4})_", name)
        if match and name.endswith(".csv"):
            found.append((int(match.group(1)), os.path.join(CHECKPOINT_DIR, name)))
    return [path for _, path in sorted(found)]


def next_checkpoint_number():
    files = checkpoint_files()
    if not files:
        return 1
    return int(re.match(r"^(\d{4})_", os.path.basename(files[-1])).group(1)) + 1


def write_checkpoint(all_rows, gesture, note):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    number = next_checkpoint_number()
    path = os.path.join(CHECKPOINT_DIR, f"{number:04d}_{gesture}_{note}.csv")
    pd.DataFrame(all_rows, columns=column_names()).to_csv(path, index=False)
    prune_checkpoints()
    return path


def prune_checkpoints():
    for path in checkpoint_files()[:-MAX_CHECKPOINTS]:
        os.remove(path)


def undo_checkpoint(all_rows):
    """Drops the newest checkpoint and restores the one before it, rewriting
    the dataset in place. Returns a message describing what happened."""
    files = checkpoint_files()
    if not files:
        return "  No checkpoints to undo."

    newest = files[-1]
    previous = files[-2] if len(files) > 1 else None
    restored = pd.read_csv(previous).values.tolist() if previous else []

    os.remove(newest)
    all_rows[:] = restored
    pd.DataFrame(all_rows, columns=column_names()).to_csv(OUTPUT_FILE, index=False)

    back_to = os.path.basename(previous) if previous else "empty dataset"
    return f"  Undid {os.path.basename(newest)} -> back to {back_to} ({len(all_rows)} samples)"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    all_rows = load_existing()
    last_capture_time = 0

    if all_rows:
        print(f"Resuming session: {PERSON_ID}")
        print("Existing progress found:")
        for g in GESTURES:
            print(f"  {g}: {count_for_gesture(all_rows, g)}/{SAMPLES_PER_GESTURE}")
        print(f"Checkpoints on disk: {len(checkpoint_files())}")
    else:
        print(f"Starting new session: {PERSON_ID}")

    print("\nKeys: SPACE capture | d delete last of this gesture | u undo checkpoint")
    print("      n next gesture | b previous gesture | q save & quit")

    # Start on the first gesture that still needs samples, but keep every
    # gesture reachable with b / n so finished ones can be redone.
    idx = len(GESTURES) - 1
    for i, gesture in enumerate(GESTURES):
        if count_for_gesture(all_rows, gesture) < SAMPLES_PER_GESTURE:
            idx = i
            break
    else:
        print("\nAll gestures already complete - use b / n to revisit any of them.")

    announced = None

    while 0 <= idx < len(GESTURES):
        gesture = GESTURES[idx]
        count = count_for_gesture(all_rows, gesture)
        complete = count >= SAMPLES_PER_GESTURE

        if announced != idx:
            state = "COMPLETE" if complete else "in progress"
            print(f"\n--- {gesture.upper()} ({count}/{SAMPLES_PER_GESTURE}, {state}) ---")
            announced = idx

        if gesture == "neutral":
            group_label = "Move hand naturally, relaxed"
        else:
            group_label = GROUP_LABELS[min(count // GROUP_SIZE, len(GROUP_LABELS) - 1)]

        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        header = f"{gesture.upper()}  {count}/{SAMPLES_PER_GESTURE}"
        if complete:
            header += "  COMPLETE"

        cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if complete else (255, 255, 0), 2)
        cv2.putText(frame, group_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"gesture {idx + 1}/{len(GESTURES)}  |  {len(all_rows)} samples total",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE capture | d delete last | u undo | b back | n next | q quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("AeroPoint - Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save(all_rows, cap)
            return

        if key == ord('n'):
            if not complete:
                print(f"  Moving on with {count}/{SAMPLES_PER_GESTURE} saved for {gesture}")
            idx += 1
            continue

        if key == ord('b'):
            if idx == 0:
                print("  Already at the first gesture.")
            else:
                idx -= 1
            continue

        if key == ord('u'):
            print(undo_checkpoint(all_rows))
            announced = None  # counts may have shifted, reprint the header
            continue

        if key == ord('d'):
            if delete_last_for_gesture(all_rows, gesture):
                count -= 1
                save(all_rows, cap, keep_open=True)
                write_checkpoint(all_rows, gesture, f"del{count:02d}")
                print(f"  Deleted last sample. Now {count}/{SAMPLES_PER_GESTURE} for {gesture}")
                announced = None
            else:
                print("  Nothing to delete for this gesture yet.")
            continue

        if key == ord(' '):
            now = time.time()
            if now - last_capture_time < CAPTURE_COOLDOWN_SEC:
                pass  # ignore, this is a repeat from the same held press
            elif complete:
                last_capture_time = now
                print(f"  {gesture} already has {SAMPLES_PER_GESTURE} samples. "
                      f"Press d to delete one first, or n to move on.")
            elif results.multi_hand_landmarks:
                row = extract_row(results.multi_hand_landmarks[0], gesture, PERSON_ID, COLLECTOR)
                all_rows.append(row)
                count += 1
                last_capture_time = now
                save(all_rows, cap, keep_open=True)
                path = write_checkpoint(all_rows, gesture, f"cap{count:02d}")
                print(f"  Captured {count}/{SAMPLES_PER_GESTURE}  (checkpoint {os.path.basename(path)})")
                if count >= SAMPLES_PER_GESTURE:
                    print(f"  {gesture.upper()} complete. Press n for the next gesture.")
                    announced = None
            else:
                print("  No hand detected, sample not saved. Try again.")

    save(all_rows, cap)

    missing = [g for g in GESTURES if count_for_gesture(all_rows, g) < SAMPLES_PER_GESTURE]
    if missing:
        print(f"\nStill incomplete: {', '.join(missing)}")
        print("Re-run the same command to pick these up again.")
    else:
        print("\nAll 7 gestures complete for this person.")


def save(all_rows, cap, keep_open=False):
    if not all_rows:
        if not keep_open:
            cap.release()
            cv2.destroyAllWindows()
        return

    df = pd.DataFrame(all_rows, columns=column_names())
    df.to_csv(OUTPUT_FILE, index=False)

    if not keep_open:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSaved {len(df)} total samples to {OUTPUT_FILE}")
        print(df.groupby("label").size())


if __name__ == "__main__":
    main()
