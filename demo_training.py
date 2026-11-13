import os
import time
import warnings
import numpy as np
import torch
import open_earth_map
import torchvision
from pathlib import Path

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start = time.time()

    # OEM_DATA_DIR = "OpenEarthMap_Mini"
    OEM_DATA_DIR = r"E:\capstone\classify_data\OpenEarthMap\OpenEarthMap_wo_xBD"
    TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = 256
    # 这个指把图片裁剪成多少像素
    N_CLASSES = 9
    LR = 0.0001
    BATCH_SIZE = 2
    # 这个是指每次给模型几张图片
    NUM_EPOCHS = 50
    # 这个是指完整训练多少次介数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs_smoke_test"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("使用设备", DEVICE)

    # fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    
    fns = sorted(
    f for f in Path(OEM_DATA_DIR).rglob("*.tif")
    if "images" in f.parts
    )
    train_names = set(
    np.atleast_1d(np.loadtxt(TRAIN_LIST, dtype=str)).tolist()
    )
    val_names = set(
    np.atleast_1d(np.loadtxt(VAL_LIST, dtype=str)).tolist()
    )
    def has_label(image_path):
        label_path = Path(str(image_path).replace("images", "labels"))
        return label_path.is_file()


    # train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    # val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    # print("Total samples      :", len(fns))
    # print("Training samples   :", len(train_fns))
    # print("Validation samples :", len(val_fns))

    
    train_fns = [
    str(f) for f in fns
    if f.name in train_names and has_label(f)
    ]
    val_fns = [
    str(f) for f in fns
    if f.name in val_names and has_label(f)
    ]
    print("找到的全部影像：", len(fns))
    print("可用训练样本：", len(train_fns))
    print("可用验证样本：", len(val_fns))
    # 只取少量样本试跑
    train_fns = train_fns[:8]
    val_fns = val_fns[:4]

    if len(train_fns) < BATCH_SIZE:
        raise RuntimeError("训练样本不足，请检查影像路径、标签路径和train.txt")

    if not val_fns:
        raise RuntimeError("没有验证样本，请检查影像路径、标签路径和val.txt")

    print("本次试跑训练样本：", len(train_fns))
    print("本次试跑验证样本：", len(val_fns))
    
    train_augm = torchvision.transforms.Compose(
        [
            open_earth_map.transforms.Rotate(),
            open_earth_map.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            open_earth_map.transforms.Resize(IMG_SIZE),
        ],
    )

    train_data = open_earth_map.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = open_earth_map.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    image, mask, filename = train_data[0]

    print("首个样本：", filename)
    print("影像形状：", image.shape)
    print("标签形状：", mask.shape)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
    )

    network = open_earth_map.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = open_earth_map.losses.JaccardLoss()

    max_score = float("-inf")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")

        train_logs = open_earth_map.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = open_earth_map.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )

        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            open_earth_map.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="model.pth",
                output_dir=OUTPUT_DIR,
            )

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
