import math

import pennylane as qml
import torch
from torch import nn
import torch.nn.functional as F


class Quantum(nn.Module):
    def __init__(self, in_channels: int, n_layers: int = 2):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels 必须是正整数")

        self.in_channels = in_channels
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.n_qubits = math.ceil(math.log2(in_channels)) if in_channels > 1 else 1
        self.encoded_dim = 2 ** self.n_qubits
        self.n_layers = n_layers
        self.eps = 1e-8

        # 模拟量子设备；这个示例没有加入噪声
        device = qml.device("default.qubit", wires=self.n_qubits)

        def circuit(inputs, weights):
            # 输入长度必须为 2**n_qubits
            qml.AmplitudeEmbedding(
                inputs,
                wires=range(self.n_qubits),
                normalize=True,
            )

            # 每层：每个量子比特做 RX、RY、RZ，再进行两两 CZ 纠缠
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)

                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CZ(wires=[i, j])

            # 返回 2**n_qubits 个计算基态的概率
            return qml.probs(wires=range(self.n_qubits))

        qnode = qml.QNode(
            circuit,
            device=device,
            interface="torch",
            diff_method="backprop",
        )

        weight_shapes = {
            "weights": (self.n_layers, self.n_qubits, 3)
        }
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 将量子电路输出的 encoded_dim 维概率映射回原通道数
        self.output_projection = nn.Linear(self.encoded_dim, self.in_channels)

    def get_vectors(self, x):
        """
        x: [B, C, H, W]
        沿通道轴取每个像元的 C 维向量并做 L2 归一化。
        返回形状仍然是 [B, C, H, W]。
        """
        norms = x.norm(p=2, dim=1, keepdim=True)
        normalized = x / norms.clamp_min(self.eps)

        # 如果某个像元的向量全为 0，就用 [1, 0, ..., 0] 代替，
        # 避免把全零向量送入振幅编码。
        zero_mask = norms <= self.eps
        fallback = torch.zeros_like(normalized)
        fallback[:, 0, :, :] = 1.0

        return torch.where(zero_mask, fallback, normalized)

    def forward(self, x):
        """
        输入： [B, C, H, W]
        输出： [B, C, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"期望输入形状为 [B, C, H, W]，实际是 {tuple(x.shape)}")

        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"初始化时 in_channels={self.in_channels}，"
                f"但输入通道数为 {channels}"
            )

        # [B, C, H, W] → [B, H, W, C] → [B*H*W, C]
        amplitudes = self.get_vectors(x)
        vectors = amplitudes.permute(0, 2, 3, 1).reshape(-1, channels)

        # 振幅编码要求向量长度为 2 的幂。
        # 例如 C=3 时补 1 个 0，变成长度 4；C=128 时无需补齐。
        if channels < self.encoded_dim:
            vectors = F.pad(vectors, (0, self.encoded_dim - channels))

        # 对每个像元的特征向量运行量子电路。
        # 输出：[B*H*W, encoded_dim]
        # Broadcast independent spatial vectors through the same circuit.
        quantum_outputs = self.quantum_layer(vectors).to(dtype=x.dtype)
        self.last_stats = {
            "input_shape": tuple(x.shape),
            "vectors": vectors.shape[0],
            "probability_sum_mean": float(quantum_outputs.detach().sum(-1).mean()),
        }

        # 把量子概率输出映射回 C 个通道
        outputs = self.output_projection(quantum_outputs)

        # [B*H*W, C] → [B, H, W, C] → [B, C, H, W]
        outputs = outputs.reshape(batch_size, height, width, channels)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()

        return outputs
