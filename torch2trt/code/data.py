"""Real ImageNet data for calibration and top-1 evaluation.

Chapters 1-7 used `weights=None` and random inputs, which is fine for latency and
for route-vs-route numerical agreement. Quantization is different: the whole
question is "how much accuracy did I lose", and you cannot ask that of a random
network. A randomly-initialised resnet18 has no accuracy to lose, and random
noise inputs produce activation distributions nothing like real images -- which
is exactly what an INT8 calibrator measures. Calibrating on noise would give a
scale factor that is wrong for every real input.

So this module provides:
  * imagenette2-160: 10 real ImageNet classes, 3925 validation images (~99 MB).
    A genuine ImageNet subset, so torchvision's ImageNet-pretrained resnet18
    classifies it directly -- no fine-tuning, no label remapping beyond picking
    out which of the 1000 logits correspond to these 10 classes.
  * the standard ImageNet preprocessing (resize 160 -> center crop 128).
    We use 128 rather than the usual 224 because these are the 160px-shortest-side
    images; cropping 224 from them would upsample and depress accuracy for
    reasons unrelated to quantization.

Download (already done if `data/imagenette2-160` exists):

    cd /root/data/learn/torch2trt/data
    curl -sL -o imagenette2-160.tgz \
      https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar xzf imagenette2-160.tgz && rm imagenette2-160.tgz
"""
from pathlib import Path

import torch

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "imagenette2-160"

# imagenette's 10 WordNet ids, and where each sits in the 1000-class ImageNet
# output layer. The pretrained resnet18 head is untouched: we just read the
# logits at these 10 positions and argmax among them. That keeps the model
# exactly as shipped (important -- re-training the head would confound the
# quantization measurement with a training difference).
WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
         "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
LABELS = ["tench", "English springer", "cassette player", "chain saw",
          "church", "French horn", "garbage truck", "gas pump",
          "golf ball", "parachute"]

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP = 128


def available() -> bool:
    return DATA_ROOT.exists()


def _preprocess(paths, size: int = CROP):
    """Load + standard ImageNet preprocess a list of image paths -> NCHW fp32."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, 160)
        img = TF.center_crop(img, size)
        batch.append(TF.to_tensor(img))
    x = torch.stack(batch)
    return (x - _MEAN) / _STD


def _list_split(split: str):
    """Return [(path, label_0_to_9)] sorted for determinism."""
    items = []
    for label, wnid in enumerate(WNIDS):
        for p in sorted((DATA_ROOT / split / wnid).glob("*.JPEG")):
            items.append((p, label))
    return items


def loader(split: str = "val", batch: int = 8, limit: int | None = None,
           shuffle_seed: int | None = 0, size: int = CROP):
    """Yield (images_fp32_cuda, labels_cuda) batches.

    `shuffle_seed` matters for calibration: the images are grouped by class on
    disk, so an unshuffled first-N sample would only ever see tenches and
    springers, and the calibrator would compute activation ranges from 2 of 10
    classes. Shuffling with a fixed seed keeps it representative AND reproducible.
    """
    items = _list_split(split)
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(len(items), generator=g).tolist()
        items = [items[i] for i in order]
    if limit is not None:
        items = items[:limit]

    for i in range(0, len(items) - batch + 1, batch):
        chunk = items[i:i + batch]
        x = _preprocess([p for p, _ in chunk], size).cuda()
        y = torch.tensor([lab for _, lab in chunk], device="cuda")
        yield x, y


def calib_batches(n_batches: int = 16, batch: int = 8, size: int = CROP):
    """Materialise a fixed calibration set (fp32, on GPU).

    Held in memory because every quantization route needs the *same* calibration
    data -- otherwise a route could look better purely by having seen luckier
    images. 16x8 = 128 images is a normal PTQ calibration budget.
    """
    return [x for x, _ in loader("train", batch=batch, limit=n_batches * batch,
                                 size=size)]


@torch.no_grad()
def top1(runnable, batch: int = 8, limit: int | None = 800,
         dtype=torch.half, size: int = CROP) -> dict:
    """Measure real top-1 accuracy over the imagenette val split.

    Restricted to the 10 imagenette columns of the 1000-way output (see the
    IMAGENET_IDX comment). `limit=800` keeps an evaluation pass to a few seconds
    while still giving ~1.7% granularity; pass None for the full 3925 images.
    """
    idx = torch.tensor(IMAGENET_IDX, device="cuda")
    correct = total = 0
    for x, y in loader("val", batch=batch, limit=limit, size=size):
        out = runnable(x.to(dtype))
        out = out[0] if isinstance(out, (tuple, list)) else out
        pred = out.float()[:, idx].argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"top1": correct / total if total else float("nan"), "n_eval": total}


if __name__ == "__main__":
    import common
    common.device_info()
    if not available():
        raise SystemExit(f"dataset not found at {DATA_ROOT}; see module docstring")

    print(f"train images: {len(_list_split('train'))}")
    print(f"val images  : {len(_list_split('val'))}")

    import torchvision
    model = torchvision.models.resnet18(
        weights="IMAGENET1K_V1").eval().cuda().half()
    print("\nfp16 pretrained resnet18 on imagenette val:")
    print(" ", top1(model))
