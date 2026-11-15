"""Small CPU check: quantum gradients, parameter update and checkpoint round trip."""
import tempfile
from pathlib import Path

import torch
from open_earth_map.networks.model_factory import build_model


if __name__ == "__main__":
    torch.manual_seed(42)
    config = {"type": "unet", "kwargs": {
        "in_channels": 3, "n_classes": 9, "depth": 2, "wf": 2,
        "quantum_enabled": True, "quantum_layers": 2,
    }}
    model = build_model(config).cpu()
    x = torch.randn(2, 3, 8, 8)
    weights = model.quantum.quantum_layer.weights
    before = weights.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model(x)
    target = torch.randint(0, 9, (2, 8, 8))
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    for label, parameter in [("circuit", weights), ("encoder", model.down_path[0].block[0].weight)]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
        print(label, "gradient norm:", parameter.grad.norm().item())
    optimizer.step()
    assert not torch.equal(before, weights.detach())
    model.eval()
    with torch.no_grad():
        expected = model(x)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.pth"
        torch.save({"model_config": config, "state_dict": model.state_dict()}, path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored = build_model(checkpoint["model_config"]).eval()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x))
    print("Output:", tuple(output.shape))
    print("Quantum stats:", model.quantum.last_stats)
    print("PASS: gradients, weight update, strict loading and predictions match.")
