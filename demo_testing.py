import os
import time
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
    print("PROJ database:", _proj_dir)

    BASE_DIR = Path(__file__).resolve().parent
    OEM_DATA_DIR = r"E:\capstone\classify_data\OpenEarthMap\OpenEarthMap_wo_xBD"
    TEST_LIST = os.path.join(OEM_DATA_DIR, "test.txt")

    N_CLASSES = 9
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PREDS_DIR = BASE_DIR / "predictions_smoke_test"
    os.makedirs(PREDS_DIR, exist_ok=True)

    fns = sorted(f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "images" in f.parts)
    test_names = set(np.atleast_1d(np.loadtxt(TEST_LIST, dtype=str)).tolist())
    test_fns = [str(f) for f in fns if f.name in test_names][:2]
    if not test_fns:
        raise RuntimeError("No test images found. Check OEM_DATA_DIR and test.txt.")

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))
    print("Class IDs:", oem.utils.class_grey_oem)

    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=oem.transforms.Resize(256),
        testing=True,
    )

    network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    network = oem.utils.load_checkpoint(
        network,
        model_name="model.pth",
        model_dir=str(BASE_DIR / "outputs_smoke_test"),
    )

    network.eval().to(DEVICE)
    for idx in range(len(test_fns)):
        img, _, fn = test_data[idx]

        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        # Keep the class IDs (0=unknown, 1..8=land-cover classes), not RGB values.
        prd = np.argmax(prd.numpy(), axis=0).astype(np.uint8)

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

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
