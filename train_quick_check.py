import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "gesture_data.csv"

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes")

feature_cols = [c for c in df.columns if c not in ["label", "person_id", "collector"]]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"SVM accuracy: {svm_acc:.3f}")
print(f"Random Forest accuracy: {rf_acc:.3f}")

if rf_acc > svm_acc:
    print("\nRandom Forest performing better:")
    print(classification_report(y_test, rf_preds))
else:
    print("\nSVM performing better:")
    print(classification_report(y_test, svm_preds))
