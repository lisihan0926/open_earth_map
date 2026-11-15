"""RGB pixel classifier with trainable quantum gates and no trainable classical layers."""
import math
import pennylane as qml
import torch
from torch import nn


class QuantumPixel(nn.Module):
    def __init__(self, in_channels=3, n_classes=9, n_layers=2, chunk_size=256):
        super().__init__()
        if in_channels != 3 or n_classes != 9:
            raise ValueError("QuantumPixel requires RGB and classes 0..8")
        if n_layers < 1 or chunk_size < 1:
            raise ValueError("n_layers and chunk_size must be positive")
        self.chunk_size = chunk_size
        dev = qml.device("default.qubit", wires=3, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(3), rotation="Y")
            for layer in range(n_layers):
                for wire in range(3):
                    qml.Rot(*weights[layer, wire], wires=wire)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])
            return qml.probs(wires=range(3))

        self.quantum = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, 3, 3)})

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected [B,3,H,W]")
        b, _, h, w = x.shape
        # Dataset ToTensor supplies RGB in [0,1]. No learned feature encoder.
        angles = x.permute(0, 2, 3, 1).reshape(-1, 3) * math.pi
        probabilities = torch.cat([self.quantum(part) for part in angles.split(self.chunk_size)], dim=0).to(x.dtype)
        foreground = probabilities.clamp_min(1e-8).log()
        # Existing runners expect nine logits. Class 0 is unknown, not a target land-cover class.
        logits = torch.cat([torch.full_like(foreground[:, :1], -30), foreground], dim=1)
        return logits.reshape(b, h, w, 9).permute(0, 3, 1, 2).contiguous()
