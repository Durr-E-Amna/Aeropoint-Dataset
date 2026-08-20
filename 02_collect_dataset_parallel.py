import cv2
import mediapipe as mp
import pandas as pd
import time
import argparse

BURST_DURATION_SEC = 5
COUNTDOWN_SEC = 3
TARGET_SAMPLES_PER_GESTURE = 400

GESTURES = [
    "point",
    "pinch",
    "fist",
    "open_palm",
    "thumbs_up",
    "peace_sign",
    "neutral",
]

parser = argparse.ArgumentParser()
parser.add_argument("--collector", required=True)
parser.add_argument("--person", required=True, type=int)
args = parser.parse_args()

COLLECTOR = args.collector.upper()
PERSON_NUM = args.person
PERSON_ID = f"{COLLECTOR}_p{PERSON_NUM}"
OUTPUT_FILE = f"gesture_data_{COLLECTOR}_p{PERSON_NUM}.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def extract_landmark_row(hand_landmarks, label, person_id, collector):
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


def run_countdown(frame_getter, seconds, message):
    start = time.time()
    while time.time() - start < seconds:
        remaining = seconds - (time.time() - start)
        frame = frame_getter()
        if frame is None:
            continue
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Starting in {remaining:.1f}s...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("AeroPoint - Dataset Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    all_rows = []
    collected_counts = {g: 0 for g in GESTURES}

    def get_frame():
        success, frame = cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1)

    print(f"Session: collector={COLLECTOR}, person_id={PERSON_ID}")
    print(f"Saving to: {OUTPUT_FILE}")

    for gesture in GESTURES:
        print(f"\nGesture: {gesture.upper()}")

        while collected_counts[gesture] < TARGET_SAMPLES_PER_GESTURE:
            frame = get_frame()
            if frame is None:
                continue

            progress = f"{PERSON_ID} | {gesture.upper()}: {collected_counts[gesture]}/{TARGET_SAMPLES_PER_GESTURE}"
            cv2.putText(frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "SPACE = burst | n = next | q = quit",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("AeroPoint - Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                save_and_exit(all_rows, cap)
                return

            if key == ord('n'):
                break

            if key == ord(' '):
                if not run_countdown(get_frame, COUNTDOWN_SEC, f"Get ready: {gesture.upper()}"):
                    save_and_exit(all_rows, cap)
                    return

                burst_start = time.time()
                burst_count = 0
                while time.time() - burst_start < BURST_DURATION_SEC:
                    success, raw_frame = cap.read()
                    if not success:
                        continue
                    raw_frame = cv2.flip(raw_frame, 1)
                    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(
                            raw_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                        row = extract_landmark_row(hand_landmarks, gesture, PERSON_ID, COLLECTOR)
                        all_rows.append(row)
                        collected_counts[gesture] += 1
                        burst_count += 1

                    cv2.putText(raw_frame, f"CAPTURING: {gesture.upper()}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("AeroPoint - Dataset Collection", raw_frame)
                    cv2.waitKey(1)
                    time.sleep(1 / 15)

                print(f"+{burst_count} samples. Total for {gesture}: {collected_counts[gesture]}")

    save_and_exit(all_rows, cap)


def save_and_exit(all_rows, cap):
    cap.release()
    cv2.destroyAllWindows()

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows, columns=column_names())
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} samples to {OUTPUT_FILE}")
    print(df.groupby("label").size())


if __name__ == "__main__":
    main()
