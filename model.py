import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# df = pd.read_csv("change2017_2023.csv", nrows=50000)
df = pd.read_csv("change2022_2023.csv", nrows=50000, header=None, names=["x", "y", "deltaNDVI"])


def classify(value):
    if value < -0.1:
        return "Degraded"
    elif value > 0.1:
        return "Recovered"
    else:
        return "Stable"

df["label"] = df["deltaNDVI"].apply(classify)

X = df[["deltaNDVI"]]  # feature
y = df["label"]        # target label

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


df["prediction"] = model.predict(X)

# Map labels to numbers
label_to_num = {"Degraded": -1, "Stable": 0, "Recovered": 1}
df["prediction_num"] = df["prediction"].map(label_to_num)

# Save
df.to_csv("deltaNDVI_2022_2023_with_predictions.csv", index=False)
