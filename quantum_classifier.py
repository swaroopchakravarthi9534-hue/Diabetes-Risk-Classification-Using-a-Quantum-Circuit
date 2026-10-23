
import pennylane as qml
from pennylane import numpy as np

# Create a simulator with 2 qubits
dev = qml.device("default.qubit", wires=2)

# Quantum circuit
@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=[0, 1], rotation="Y")
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Test with one sample
sample = np.array([0.4523, 0.4054])
weights = np.array([0.1, 0.2])

result = circuit(sample, weights)

print("Quantum circuit output:", result)
print("Output range: -1 to +1")
