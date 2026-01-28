import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm


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


def init_sam():
    # Initialize the SAM model
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join("models", "sam_checkpoint", "sam_vit_b_01ec64.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)

    # Create the automatic mask generator from SAM.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        # pred_iou_thresh=0.8,  # lower threshold for more masks
        # stability_score_thresh=0.90,  # allow slightly less stable masks
        min_mask_region_area=20,
    )
    return mask_generator


def segment_image(image_path, model) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = model.generate(image)

    # Combine all object masks into one image.
    dir_path = os.path.dirname(image_path)
    res_path = os.path.join(dir_path, "results")
    filename = os.path.basename(path)

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
    dir_path = os.path.join("data", "images", "literature", "reference_x")
    files = os.listdir(dir_path)
    filepaths = [os.path.join(dir_path, f) for f in files if f.startswith("X_")]
    progress = tqdm(total=len(filepaths))
    model = init_sam()
    for path in filepaths:
        progress.set_description(f"{path}")
        segment_image(path, model)
        progress.update()
    progress.close()
