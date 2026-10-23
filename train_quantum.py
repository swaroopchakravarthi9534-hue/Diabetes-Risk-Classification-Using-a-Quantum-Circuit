
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

data = pd.read_csv(url, names=columns)
X = data[["Glucose", "BMI"]].values
y = data["Outcome"].values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quantum simulator
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x * np.pi, wires=[0, 1], rotation="Y")
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Convert quantum output to probability
def predict_probability(x, weights):
    return (1 - circuit(x, weights)) / 2

def loss(weights):
    predictions = np.array([
        predict_probability(x, weights) for x in X_train
    ])
    return np.mean((predictions - y_train) ** 2)

# Train the model
weights = np.array([0.1, 0.2], requires_grad=True)
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

print("Training quantum classifier...")

for step in range(30):
    weights = optimizer.step(loss, weights)

    if (step + 1) % 5 == 0:
        print("Step", step + 1, "Loss:", float(loss(weights)))

# Evaluate
predictions = [
    int(predict_probability(x, weights) >= 0.5)
    for x in X_test
]

print("\nTest Accuracy:", accuracy_score(y_test, predictions))
print("Learned weights:", weights)
