"""Build a tiny COCO detection subset from the downloaded COCO 2017 zips.

The subset is for a short training smoke test, not for reporting COCO metrics.
"""

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COCO = ROOT / "mmdetection" / "data" / "coco"
OUT = ROOT / "mmdetection" / "data" / "coco_mini"
TRAIN_COUNT = 16
VAL_COUNT = 4


def read_val_annotations(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        with archive.open("annotations/instances_val2017.json") as stream:
            return json.load(stream)


def select_images(data: dict) -> tuple[list[dict], list[dict]]:
    annotation_counts = {image["id"]: 0 for image in data["images"]}
    for annotation in data["annotations"]:
        annotation_counts[annotation["image_id"]] += 1
    images = [image for image in data["images"] if annotation_counts[image["id"]] > 0]
    images.sort(key=lambda image: image["id"])
    if len(images) < TRAIN_COUNT + VAL_COUNT:
        raise RuntimeError("The COCO validation annotations contain too few images.")
    return images[:TRAIN_COUNT], images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]


def write_split(data: dict, images: list[dict], name: str) -> None:
    image_ids = {image["id"] for image in images}
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [
            annotation
            for annotation in data["annotations"]
            if annotation["image_id"] in image_ids
        ],
        "categories": data["categories"],
    }
    annotation_path = OUT / "annotations" / f"instances_{name}2017.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(subset, ensure_ascii=False), encoding="utf-8"
    )


def extract_images(images: list[dict], split: str, val_zip: Path) -> None:
    destination = OUT / f"{split}2017"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(val_zip) as archive:
        for image in images:
            member = f"val2017/{image['file_name']}"
            target = destination / image["file_name"]
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main() -> None:
    annotation_zip = COCO / "annotations_trainval2017.zip"
    val_zip = COCO / "val2017.zip"
    if not annotation_zip.exists() or not val_zip.exists():
        raise FileNotFoundError(
            "Download annotations_trainval2017.zip and val2017.zip into mmdetection/data/coco first."
        )
    if OUT.exists():
        shutil.rmtree(OUT)
    data = read_val_annotations(annotation_zip)
    train_images, val_images = select_images(data)
    extract_images(train_images, "train", val_zip)
    extract_images(val_images, "val", val_zip)
    write_split(data, train_images, "train")
    write_split(data, val_images, "val")
    print(f"Created {len(train_images)} training images and {len(val_images)} validation images.")
    print(f"Dataset: {OUT}")


if __name__ == "__main__":
    main()
