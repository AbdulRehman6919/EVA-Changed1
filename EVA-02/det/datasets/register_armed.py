"""Register the ARMED dataset with deterministic class-id mapping.

Why this file does more than `register_coco_instances`:

The previous version unconditionally set
    MetadataCatalog.get(name).thing_classes = ["Armed", "Unarmed", "Gun"]
After the COCO loader auto-remaps non-`[1, #categories]` ids, the contiguous
id <-> class-name mapping it produces may not line up with the order
assumed above. That mismatch is what produced the
    "Category ids in annotations are not in [1, #categories]! We'll apply a
     mapping for you."
warning, and it can silently mislabel per-category AP rows.

Here we read the JSON ourselves and set:
  * `thing_classes` from the JSON `categories` (sorted by id, so the order is
    deterministic and matches the contiguous indices the loader will use).
  * `thing_dataset_id_to_contiguous_id` so the loader uses our mapping
    instead of inventing one (which suppresses the warning and locks in a
    stable mapping for evaluation).
"""

import json
import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


_DATASETS = {
    "armed_train": (
        "/kaggle/working/OriginalDataset/annotations/armed_train.json",
        "/kaggle/working/OriginalDataset",
    ),
    "armed_val": (
        "/kaggle/working/OriginalDataset/annotations/armed_val.json",
        "/kaggle/working/OriginalDataset",
    ),
}


def _categories_from_json(json_path):
    """Return categories sorted by JSON id; falls back to [] if file missing."""
    if not os.path.isfile(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    return sorted(cats, key=lambda c: c["id"])


for name, (json_file, image_root) in _DATASETS.items():
    register_coco_instances(name, {}, json_file, image_root)

    cats = _categories_from_json(json_file)
    if cats:
        meta = MetadataCatalog.get(name)
        meta.thing_classes = [c["name"] for c in cats]
        meta.thing_dataset_id_to_contiguous_id = {
            c["id"]: i for i, c in enumerate(cats)
        }
    else:
        # JSON not available at registration time (e.g. running outside
        # Kaggle); keep the historical hard-coded order so downstream code
        # still has a metadata entry to inspect.
        MetadataCatalog.get(name).thing_classes = ["Armed", "Unarmed", "Gun"]
