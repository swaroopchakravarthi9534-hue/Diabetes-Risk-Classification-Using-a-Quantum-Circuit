
import pandas as pd
import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

data = pd.read_csv(url, names=columns)

X = data[["Glucose", "BMI"]].values
y = data["Outcome"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale using training data only
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- QUANTUM MODEL ----------------

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x * np.pi, wires=[0, 1], rotation="Y")
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

def predict_probability(x, weights):
    return (1 - circuit(x, weights)) / 2

def loss(weights):
    predictions = np.stack([
        predict_probability(x, weights) for x in X_train
    ])
    return np.mean((predictions - y_train) ** 2)

weights = np.array([0.1, 0.2], requires_grad=True)
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

print("Training quantum model...")

for step in range(30):
    weights = optimizer.step(loss, weights)

    if (step + 1) % 10 == 0:
        print("Step", step + 1, "Loss:", float(loss(weights)))

quantum_predictions = [
    int(predict_probability(x, weights) >= 0.5)
    for x in X_test
]

# ---------------- CLASSICAL MODEL ----------------

print("\nTraining classical model...")

model = LogisticRegression()
model.fit(X_train, y_train)

classical_predictions = model.predict(X_test)

# ---------------- EVALUATION ----------------

def evaluate_model(name, actual, predicted):
    print("\n", name)
    print("Accuracy :", round(accuracy_score(actual, predicted), 4))
    print("Precision:", round(precision_score(actual, predicted, zero_division=0), 4))
    print("Recall   :", round(recall_score(actual, predicted, zero_division=0), 4))
    print("F1-score :", round(f1_score(actual, predicted, zero_division=0), 4))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))

evaluate_model("QUANTUM MODEL", y_test, quantum_predictions)
evaluate_model("CLASSICAL MODEL", y_test, classical_predictions)

print("\nConfusion matrix order: [[TN, FP], [FN, TP]]")
