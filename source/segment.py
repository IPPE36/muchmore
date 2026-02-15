import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from source.timer import timed

matplotlib.use("agg")


def combine_masks(masks, image_shape):
    """
    Combine individual binary masks into one uint8 numpy array.
    Each object is given a unique index starting at 1.
    """
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for idx, mask in enumerate(masks, start=1):
        segmentation = mask['segmentation']
        segmentation = segmentation.astype(bool)
        combined_mask[segmentation] = idx
    return combined_mask


@timed("Initialize SAM")
def init_sam():
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join("sam", "sam_vit_b_01ec64.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)

    # Create the automatic mask generator from SAM.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=70,
        # pred_iou_thresh=0.88,
        # stability_score_thresh=0.95,
        min_mask_region_area=25,
    )
    return mask_generator


@timed("Segment Image")
def segment_image(image_path, model) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = model.generate(image)

    # Combine all object masks into one image.
    dir_path = os.path.dirname(image_path)
    res_path = os.path.join(dir_path, "results")
    filename = os.path.basename(image_path)

    combined_mask = combine_masks(masks, image.shape)
    plt.imshow(combined_mask, cmap="jet")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, filename.replace(".PNG", "_OUT.png")), dpi=300)
    plt.close()
    plt.imshow(np.where(combined_mask > 0, 1, 0), cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, filename.replace(".PNG", "_SEG.png")), dpi=300)
    plt.close()
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, filename.replace(".PNG", "_IN.png")), dpi=300)
    plt.close()


if __name__ == "__main__":
    exit()

