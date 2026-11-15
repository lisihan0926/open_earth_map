"""Build models from the configuration stored alongside their weights."""
from .unet import UNet


def build_model(config):
    if config["type"] == "quantum_pixel":
        from .quantum_pixel import QuantumPixel
        return QuantumPixel(**config["kwargs"])
    if config["type"] != "unet":
        raise ValueError(f"Unsupported model type: {config['type']}")
    return UNet(**config["kwargs"])
