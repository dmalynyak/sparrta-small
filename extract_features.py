from pathlib import Path
import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader

DATA_DIR = Path("dataset")
OUT_DIR = Path("features")
SPLITS = ["trainval", "test"]
RELATIONS = ["left", "right", "front", "back"]
BATCH_SIZE = 50

VIT_MODELS = {
    "vit_sup": "vit_base_patch16_224.augreg_in1k",  # supervised ViT-B/16, ImageNet-1k
    "vit_dino": "vit_base_patch16_224.dino",        # self-supervised DINO ViT-B/16, ImageNet-1k
}
RESNET_MODELS = {
    "resnet50": "resnet50.tv_in1k",                 # supervised ResNet-50, ImageNet-1k
}


def load_model(name, device):
    model = timm.create_model(name, pretrained=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_loader(model, images):
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["crop_pct"] = 1.0 # without cropping
    transform = timm.data.create_transform(**cfg, is_training=False)
    return DataLoader([transform(img) for img in images], batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def extract_vit(model, loader, device):
    cls, mean = [], [] # 1000 x 12 x 768 at the end
    for x in loader:  # B x 3 x 224 x 224
        x = model.patch_embed(x.to(device))
        x = model._pos_embed(x)
        x = model.norm_pre(x)

        batch_cls, batch_mean = [], []
        for block in model.blocks:
            x = block(x) # 
            batch_cls.append(x[:, 0]) # CLS token B x 768
            batch_mean.append(x[:, 1:].mean(dim=1))  # mean of 196 patches B x 768
        
        # last MLP and normalization are skipped, because we never need classifier.

        cls.append(torch.stack(batch_cls, dim=1).cpu())    # B x 12 x 768
        mean.append(torch.stack(batch_mean, dim=1).cpu())  # B x 12 x 768
    return torch.cat(cls).numpy(), torch.cat(mean).numpy()  # 1000 x 12 x 768 each


@torch.no_grad()
def extract_resnet(model, loader, device):
    feats = {"layer1": [], "layer2": [], "layer3": [], "layer4": []}
    for x in loader: # Bx3x224x224
        x = model.conv1(x.to(device))
        x = model.bn1(x)
        x = model.act1(x)
        x = model.maxpool(x)
        for name in feats:
            x = getattr(model, name)(x)
            feats[name].append(x.mean(dim=(2, 3)).cpu())  # global average pool B x Channels
            # Bx256, Bx512, Bx1024, Bx2048
    return {name: torch.cat(v).numpy() for name, v in feats.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    paths = [p for split in SPLITS for p in sorted((DATA_DIR / split).glob("*.png"))]
    assert len(paths) == 1000, len(paths)
    images = [Image.open(p).convert("RGB") for p in paths]
    labels = np.array([RELATIONS.index(p.stem.split("_")[1]) for p in paths])  # "0213_front.png" -> 2
    splits = np.array([p.parent.name for p in paths])  # "trainval" / "test"
    OUT_DIR.mkdir(exist_ok=True)
    meta = dict(labels=labels, splits=splits, paths=np.array([str(p) for p in paths]),
                relations=np.array(RELATIONS))
    # labels: 1000
    # splits: 1000
    # paths: 1000
    # relations: 4

    for key, name in VIT_MODELS.items():
        model = load_model(name, device)
        cls, mean = extract_vit(model, make_loader(model, images), device)
        np.savez(OUT_DIR / f"{key}.npz", cls=cls, mean=mean, **meta)
        # cls: 1000 x 12 x 768
        # mean: 1000 x 12 x 768
        # labels: 1000
        # splits: 1000
        # paths: 1000
        # relations: 4
        print(f"name: {name}, shapes: cls {cls.shape}, mean {mean.shape}, labels {labels.shape}, splits {splits.shape}, paths {len(paths)}, relations {len(RELATIONS)}")

    for key, name in RESNET_MODELS.items():
        model = load_model(name, device)
        feats = extract_resnet(model, make_loader(model, images), device)
        np.savez(OUT_DIR / f"{key}.npz", **feats, **meta)
        # layer1: 1000 x 256
        # layer2: 1000 x 512
        # layer3: 1000 x 1024
        # layer4: 1000 x 2048
        # labels: 1000
        # splits: 1000
        # paths: 1000
        # relations: 4
        print(f"name: {name}, shapes: " + ", ".join(f"{s} {f.shape}" for s, f in feats.items()))


if __name__ == "__main__":
    main()
