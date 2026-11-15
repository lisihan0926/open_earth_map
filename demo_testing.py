import os
import time
import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

# Use this environment's coordinate database, not PostgreSQL/QGIS paths
# inherited from the system. Configure it before importing Rasterio/GDAL.
_rasterio_dir = Path(find_spec("rasterio").origin).parent
_proj_dir = next(
    (path for path in (
        _rasterio_dir / "proj_data",
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "share" / "proj",
    ) if (path / "proj.db").is_file()),
    None,
)
if _proj_dir is not None:
    os.environ["PROJ_DATA"] = str(_proj_dir)
    os.environ["PROJ_LIB"] = str(_proj_dir)
for _gdal_dir in (
    _rasterio_dir / "gdal_data",
    Path(sys.prefix) / "Library" / "share" / "gdal",
):
    if _gdal_dir.is_dir():
        os.environ["GDAL_DATA"] = str(_gdal_dir)
        break

import rasterio
from rasterio.transform import Affine
import numpy as np
import torch
import cv2
import open_earth_map as oem
from open_earth_map.networks.model_factory import build_model
from open_earth_map.evaluation import confusion_counts, export_metrics


def print_spatial_reference(label, dataset):
    crs = dataset.crs or dataset.gcps[1]
    print(f"\n[{label}] {dataset.name}")
    print("Spatial reference:", crs if crs else "UNKNOWN")
    print("EPSG:", crs.to_epsg() if crs else None)
    print("Projected:", crs.is_projected if crs else False)
    print("Geographic:", crs.is_geographic if crs else False)
    print("Spatial reference WKT:", crs.to_wkt() if crs else "UNKNOWN")
    print("GeoTransform:", dataset.transform.to_gdal())
    print("Bounds:", dataset.bounds)
    print("Raster size:", dataset.width, "x", dataset.height, flush=True)

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "quantum", "quantum_only"], default="quantum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Number of test images; 0 evaluates all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="Evaluate test or validation list")
    args = parser.parse_args()
    print("PROJ database:", _proj_dir)

    BASE_DIR = Path(__file__).resolve().parent
    OEM_DATA_DIR = r"E:\capstone\classify_data\OpenEarthMap\OpenEarthMap_wo_xBD"
    TEST_LIST = os.path.join(OEM_DATA_DIR, f"{args.split}.txt")
    print("Evaluation split:", args.split, "List:", TEST_LIST)

    checkpoint_path = args.checkpoint or BASE_DIR / "outputs_comparison" / args.variant / f"seed_{args.seed}" / "model.pth"
    if args.checkpoint is None and not checkpoint_path.is_file():
        legacy_path = checkpoint_path.with_name("model_quantum.pth")
        if legacy_path.is_file():
            checkpoint_path = legacy_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"First run: python demo_training.py --variant {args.variant} --seed {args.seed}\n"
            "Or specify an existing configured checkpoint with --checkpoint PATH."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    input_config = checkpoint.get("input_config")
    if not model_config or not input_config:
        raise RuntimeError(
            "Checkpoint has no model/input configuration. "
            "Train with the updated demo_training.py to create a configured checkpoint. "
            "Existing weights have not been changed."
        )
    N_CLASSES = model_config["kwargs"]["n_classes"]
    network = build_model(model_config)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    quantum_keys = [k for k in checkpoint["state_dict"] if k.startswith("quantum.")]
    print("Restored quantum parameter tensors:", quantum_keys)
    print("Loaded:", checkpoint_path)
    print("Model configuration:", model_config)
    print("Input configuration:", input_config)
    print("Saved epoch:", checkpoint["epoch"] + 1)
    print("Best validation score:", checkpoint["best_score"])
    DEVICE = args.device
    PREDS_DIR = checkpoint_path.parent / "predictions"
    os.makedirs(PREDS_DIR, exist_ok=True)

    fns = sorted(f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "images" in f.parts)
    test_names = set(np.atleast_1d(np.loadtxt(TEST_LIST, dtype=str)).tolist())
    test_fns = [str(f) for f in fns if f.name in test_names]
    candidate_count = len(test_fns)
    test_fns = [fn for fn in test_fns if Path(fn.replace("images", "labels")).is_file()]
    print("Test candidates:", candidate_count)
    print("Skipped (missing labels):", candidate_count - len(test_fns))
    print("Available labeled test images:", len(test_fns))
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.limit:
        test_fns = test_fns[:args.limit]
    if not test_fns:
        raise RuntimeError("No labeled images found in selected split. If test labels are unavailable, use --split val. Evaluation requires real labels.")

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))
    print("Class IDs:", oem.utils.class_grey_oem)

    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=oem.transforms.Resize(input_config["image_size"]),
        testing=True,
    )

    network.eval().to(DEVICE)
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    inference_seconds = 0.0
    for idx in range(len(test_fns)):
        img, _, fn = test_data[idx]

        inputs = img.unsqueeze(0).to(DEVICE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        prediction_start = time.perf_counter()
        with torch.no_grad():
            logits = network(inputs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - prediction_start
        prd = logits.squeeze(0).cpu()
        # Keep the class IDs (0=unknown, 1..8=land-cover classes), not RGB values.
        prd = np.argmax(prd.numpy(), axis=0).astype(np.uint8)
        if hasattr(network.quantum, "last_stats"):
            print("Quantum inference:", network.quantum.last_stats)
        preview = oem.utils.make_rgb(prd)
        cv2.imwrite(str(PREDS_DIR / f"{Path(fn).stem}_preview.png"), cv2.cvtColor(preview.astype(np.uint8), cv2.COLOR_RGB2BGR))

        fout = os.path.join(PREDS_DIR, f"{Path(fn).stem}_classes.tif")
        with rasterio.open(fn, "r") as src:
            print_spatial_reference("SOURCE", src)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff", count=1, dtype="uint8", nodata=None,
                photometric="PALETTE", interleave="band",
                crs=src.crs, transform=src.transform,
                width=src.width, height=src.height,
            )
            gcps, gcp_crs = src.gcps
            spatial_crs = src.crs or gcp_crs
            if spatial_crs and not (spatial_crs.is_projected or spatial_crs.is_geographic):
                raise RuntimeError(
                    f"Unrecognized spatial reference: {spatial_crs}. "
                    "Check the PROJ database path printed above."
                )
            if not gcps and (src.crs is None or src.transform == Affine.identity()):
                raise RuntimeError(
                    f"Source image has no complete georeferencing: {fn}\n"
                    f"CRS={src.crs}, transform={src.transform}\n"
                    "Provide a georeferenced reference raster for this image, or "
                    "its true CRS and pixel-to-map transform. "
                    "Assigning an EPSG code alone cannot recover its location."
                )
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            with rasterio.open(fn.replace("images", "labels")) as label_src:
                truth = label_src.read(1)
            matrix += confusion_counts(truth, prd, N_CLASSES)
            with rasterio.open(fout, "w", **profile) as dst:
                if gcps:
                    dst.gcps = (gcps, gcp_crs)
                else:
                    dst.crs = src.crs
                    dst.transform = src.transform
                dst.update_tags(**src.tags())
                for namespace in ("RPC", "GEOLOCATION"):
                    tags = src.tags(ns=namespace)
                    if tags:
                        dst.update_tags(ns=namespace, **tags)
                dst.write(prd, 1)
                dst.set_band_description(1, "Land-cover class ID (0=unknown, 1-8=land cover)")
                dst.update_tags(**{
                    f"CLASS_{class_id}": name
                    for name, class_id in oem.utils.class_grey_oem.items()
                })
                # A palette changes display colors only; pixel values stay 0..8.
                dst.write_colormap(1, {
                    class_id: tuple(oem.utils.class_rgb_oem[name]) + (255,)
                    for name, class_id in oem.utils.class_grey_oem.items()
                })
            with rasterio.open(fout) as result:
                print_spatial_reference("OUTPUT (reopened from disk)", result)
                if (result.crs != src.crs or result.transform != src.transform
                        or result.shape != src.shape or result.gcps != src.gcps):
                    raise RuntimeError(f"Output georeferencing differs from source: {fout}")
                if result.count != 1 or not np.array_equal(result.read(1), prd):
                    raise RuntimeError(f"Output class IDs differ from prediction: {fout}")
                class_ids, counts = np.unique(result.read(1), return_counts=True)
                print("Class pixel counts:", dict(zip(class_ids.tolist(), counts.tolist())))
        print("Saved:", fout)

    names = [next((name for name, value in oem.utils.class_grey_oem.items() if value == i), str(i)) for i in range(N_CLASSES)]
    summary = export_metrics(matrix, PREDS_DIR, names, {
        "checkpoint": str(checkpoint_path.resolve()), "model_config": model_config,
        "input_config": input_config, "test_files": test_fns, "split": args.split,
        "evaluation_resolution": "original label resolution; predictions resized with nearest neighbor",
        "device": DEVICE, "images": len(test_fns),
        "parameters": sum(p.numel() for p in network.parameters()),
        "inference_seconds": inference_seconds,
        "seconds_per_image": inference_seconds / len(test_fns),
        "timing": "forward only, includes first-call overhead; excludes IO and preprocessing",
    })
    print("Test mIoU:", summary["mIoU"], "OA:", summary["OA"], "Macro F1:", summary["macro_F1"])
    print("Metrics saved to:", PREDS_DIR)
    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
