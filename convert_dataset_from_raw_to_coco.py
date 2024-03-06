import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 933120000

# Define chunk size and overlap percentage
min_gsd = 0.15875
chunk_size = 960
overlap_percent = 0.2
# resize_to_smallest_gsd = True
# individual_image_test = True
# blured = True
# blacked = True if blured == False else False
# python tools/train.py configs/reference_configs/dino.py --work-dir work_dirs/dino_jpg_tr1
categories = [
    {"id": 1, "name": "bomb-crater", "supercategory": "crater"}
]  # , {'id': 2, 'name': 'pedestrian'}, {'id': 1, 'name': 'cyclist'}

# Initialize the COCO dataset
coco_train = {"images": [], "annotations": [], "categories": categories.copy()}
coco_val = {"images": [], "annotations": [], "categories": categories.copy()}
coco_test = {"images": [], "annotations": [], "categories": categories.copy()}

root_dir = "CHAI_RAW/"

dataset_df = pd.read_csv("dataset_information.csv", delimiter=";")


def main(args):
    """x"""
    resize_to_smallest_gsd = args.resize_to_smallest_gsd
    individual_image_test = args.individual_image_test
    blured = args.blured
    blacked = args.blacked
    strict = args.strict
    corrected = args.corrected
    out_dir = args.out_dir
    dry = args.dry
    jpg = args.jpg

    total_craters_without_additional_overlap = 0

    name = "CHAI_FULL" if not strict else "CHAI_LIGHT"
    name = name + "_JPG" if jpg else name
    size_key = "diameter(px)" if not corrected else "diameter(px-w)"

    train_image_id = 1
    train_ann_id = 1

    val_image_id = 1
    val_ann_id = 1

    test_image_id = 1
    test_ann_id = 1

    out_dir = os.path.join(out_dir, name)

    if not os.path.exists(out_dir) and not dry:
        os.makedirs(out_dir)

    out_dir_images = os.path.join(out_dir, "images")
    if not os.path.exists(out_dir_images) and not dry:
        os.makedirs(out_dir_images)

    out_dir_annotations = os.path.join(out_dir, "annotations")
    if not os.path.exists(out_dir_annotations) and not dry:
        os.makedirs(out_dir_annotations)

    train_ann_file = os.path.join(out_dir_annotations, "instances_train2017.json")
    val_ann_file = os.path.join(out_dir_annotations, "instances_val2017.json")
    test_ann_file = os.path.join(out_dir_annotations, "instances_test2017.json")

    for id, image_row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        name = image_row["Name"]
        split = image_row["Split"]
        gsd = image_row["GSD in m"]

        if individual_image_test is True and split == "Test":
            # mkdir if not exists
            if not os.path.exists(os.path.join(out_dir, name)) and not dry:
                os.makedirs(os.path.join(out_dir, name))

            ind_coco_test = {
                "images": [],
                "annotations": [],
                "categories": categories.copy(),
            }
            ind_test_ann_file = os.path.join(
                out_dir_annotations, "instances_{}.json".format(name)
            )

            ind_test_image_id = 1
            ind_test_ann_id = 1

        img_path = os.path.join(root_dir, name, name + ".png")
        img = Image.open(img_path)

        if corrected is True:
            csv_file = img_path.replace(".png", "_craters_manual_adapted.csv")
        else:
            csv_file = img_path.replace(".png", "_craters.csv")

        # Format center_x, center_y, diameter(px), diameter(m)
        annotations_df = pd.read_csv(csv_file)

        total_craters_without_additional_overlap += len(annotations_df)

        mask = Image.open(img_path.replace(".png", "_mask.png"))
        info_df = pd.read_csv(img_path.replace(".png", "_info.csv"))

        # gsd = info_df.values[0][0]

        if resize_to_smallest_gsd:
            factor = gsd / min_gsd

            (width, height) = (int(img.width * factor), int(img.height * factor))
            # print("Before:", img.size, gsd)

            # print("Resizing to smallest GSD:", min_gsd, "m/px")
            # print("Resizing to:", width, "x", height, "px")
            # print("Factor:", factor, gsd)
            # print(name)

            img = img.resize((width, height))
            mask = mask.resize((width, height))
            # print("After:", img.size, gsd / factor)

            # Transform bounding boxes to new size
            annotations_df["x"] = annotations_df["x"] * factor
            annotations_df["y"] = annotations_df["y"] * factor
            annotations_df[size_key] = annotations_df[size_key] * factor

        assert img.size == mask.size

        # Mask the image (black) #####################################################################
        if blacked == True:
            # Create a new blank image
            new_img = Image.new("RGB", img.size, color=(0, 0, 0))

            # Paste the original image onto the new image using the mask as transparency mask
            new_img.paste(img, mask=mask)

        # Mask the image (blur) ######################################################################
        elif blured == True:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=7))

            # Create a new blank image
            new_img = Image.new("RGBA", img.size, color=(255, 255, 255, 0))

            # Paste the blurred image onto the new image
            new_img.paste(blurred_img, (0, 0))

            # Paste the original image onto the new image using the mask as transparency mask
            new_img.paste(img, mask=mask)
        else:
            raise NotImplementedError
        #############################################################################################

        width, height = new_img.size

        # Calculate the amount of padding needed to make the image a multiple of the chunk size
        pad_width = math.ceil(width / chunk_size) * chunk_size - width
        pad_height = math.ceil(height / chunk_size) * chunk_size - height

        # Pad the image with borders to make it a multiple of the chunk size
        new_width = width + pad_width
        new_height = height + pad_height
        padded_img = ImageOps.expand(
            new_img, border=(0, 0, pad_width, pad_height), fill=0
        )
        padded_mask = ImageOps.expand(
            mask, border=(0, 0, pad_width, pad_height), fill=0
        )

        # Calculate the number of chunks needed in the x and y directions, taking into account the overlap percentage
        overlap_pixels = int(chunk_size * overlap_percent)
        chunk_size_with_overlap = chunk_size - overlap_pixels
        num_chunks_x = math.ceil((new_width - overlap_pixels) / chunk_size_with_overlap)
        num_chunks_y = math.ceil(
            (new_height - overlap_pixels) / chunk_size_with_overlap
        )

        # Loop over the chunks and save each one as a separate file
        for i in range(num_chunks_x):
            for j in range(num_chunks_y):
                if split == "Train":
                    image_id = train_image_id
                    train_image_id += 1
                elif split == "Val":
                    image_id = val_image_id
                    val_image_id += 1
                elif split == "Test":
                    image_id = test_image_id
                    test_image_id += 1

                if individual_image_test is True and split == "Test":
                    ind_image_id = ind_test_image_id
                    ind_test_image_id += 1

                # Calculate the top-left corner and bottom-right corner of the chunk
                x1 = i * chunk_size_with_overlap
                y1 = j * chunk_size_with_overlap

                x2 = min((i + 1) * chunk_size_with_overlap + overlap_pixels, new_width)
                y2 = min((j + 1) * chunk_size_with_overlap + overlap_pixels, new_height)

                # Crop the chunk from the padded image
                chunk = padded_img.crop((x1, y1, x2, y2))
                mask_chunk = padded_mask.crop((x1, y1, x2, y2))

                # out of region of interest
                if np.max(np.array(mask_chunk)) == 0:
                    # print("mask_chunk is empty", name, i, j)
                    continue

                # Some images were already padded by our industry partner, thus some chucks are empty
                new_img_np = np.array(chunk)[:, :, 0:3]
                if np.max(new_img_np) == np.min(new_img_np) and np.max(new_img_np) == 0:
                    # print("chunk is empty black", name, i, j)
                    continue

                # Some images were already padded by our industry partner, thus some chucks are empty
                if (
                    np.max(new_img_np) == np.min(new_img_np)
                    and np.min(new_img_np) == 255
                ):  # 1944-11-01_106G-3472_3033_3035_4033_4034
                    # print("chunk is empty white", name, i, j)
                    continue

                if chunk.size != (960, 960):
                    # print("chunk is not 960x960: " +  name + ", " + str(new_img_np.shape))
                    continue

                extension = ".jpg" if jpg else ".png"
                file_name = name + "_" + str(i) + "_" + str(j) + extension
                image_info = {
                    "id": image_id,
                    "file_name": file_name,
                    "height": chunk.size[1],
                    "width": chunk.size[0],
                }

                if individual_image_test is True and split == "Test":
                    ind_image_info = {
                        "id": ind_image_id,
                        "file_name": file_name,
                        "height": chunk.size[1],
                        "width": chunk.size[0],
                    }

                bbs_for_chunk = []

                # Calculate the bounding box for each circle in the info_df DataFrame
                for _, row in annotations_df.iterrows():
                    x, y = int(row["x"]), int(row["y"])
                    diameter_px = int(row[size_key])

                    # Calculate the bounding box coordinates
                    left = x - x1
                    top = y - y1
                    width = diameter_px
                    height = diameter_px

                    if left > 0 and left < chunk_size and top > 0 and top < chunk_size:
                        # Calculate the bounding box relative to the chunk

                        if mask_chunk.getpixel((left, top)) != 255:
                            continue

                        bbox_center_x = left - width // 2
                        bbox_center_y = top - height // 2
                        bbox_width = width
                        bbox_height = height

                        if bbox_center_x < 0:
                            bbox_center_x = 0
                        if bbox_center_y < 0:
                            bbox_center_y = 0
                        if bbox_center_x + bbox_width > chunk_size:
                            bbox_width = chunk_size - bbox_center_x
                        if bbox_center_y + bbox_height > chunk_size:
                            bbox_height = chunk_size - bbox_center_y

                        bbs_for_chunk.append(
                            [bbox_center_x, bbox_center_y, bbox_width, bbox_height]
                        )

                        if split == "Train":
                            ann_id = train_ann_id
                            train_ann_id += 1
                        elif split == "Val":
                            ann_id = val_ann_id
                            val_ann_id += 1
                        elif split == "Test":
                            ann_id = test_ann_id
                            test_ann_id += 1

                        if individual_image_test is True and split == "Test":
                            ind_ann_id = ind_test_ann_id
                            ind_test_ann_id += 1

                        annotation = {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [
                                bbox_center_x,
                                bbox_center_y,
                                bbox_width,
                                bbox_height,
                            ],
                            "area": (bbox_width * bbox_height),
                            "iscrowd": 0,
                        }

                        if split == "Train":
                            coco_train["annotations"].append(annotation)
                        elif split == "Val":
                            coco_val["annotations"].append(annotation)
                        elif split == "Test":
                            coco_test["annotations"].append(annotation)

                        if individual_image_test is True and split == "Test":
                            annotation = {
                                "id": ind_ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [
                                    bbox_center_x,
                                    bbox_center_y,
                                    bbox_width,
                                    bbox_height,
                                ],
                                "area": (bbox_width * bbox_height),
                                "iscrowd": 0,
                            }
                            ind_coco_test["annotations"].append(annotation)

                if len(bbs_for_chunk) == 0 and strict:
                    continue

                if not dry:
                    if individual_image_test is True and split == "Test":
                        chunk.save(os.path.join(out_dir, name, file_name))
                    chunk.save(os.path.join(out_dir_images, file_name))

                if split == "Train":
                    coco_train["images"].append(image_info)
                elif split == "Val":
                    coco_val["images"].append(image_info)
                elif split == "Test":
                    coco_test["images"].append(image_info)

                if individual_image_test is True and split == "Test":
                    ind_coco_test["images"].append(ind_image_info)

        if individual_image_test is True and split == "Test" and not dry:
            with open(ind_test_ann_file, "w") as f:
                json.dump(ind_coco_test, f)

    if not dry:
        with open(train_ann_file, "w") as f:
            json.dump(coco_train, f)

        with open(val_ann_file, "w") as f:
            json.dump(coco_val, f)

        with open(test_ann_file, "w") as f:
            json.dump(coco_test, f)

    splits = ["Train", "Val", "Test"]
    num_images = [
        len(coco_train["images"]),
        len(coco_val["images"]),
        len(coco_test["images"]),
    ]
    num_annotations = [
        len(coco_train["annotations"]),
        len(coco_val["annotations"]),
        len(coco_test["annotations"]),
    ]
    num_categories = [
        len(coco_train["categories"]),
        len(coco_val["categories"]),
        len(coco_test["categories"]),
    ]

    df = pd.DataFrame(
        [num_images, num_annotations, num_categories],
        columns=splits,
        index=["Images", "Annotations", "Categories"],
    )

    print(
        "Total Craters found (no duplicate):", total_craters_without_additional_overlap
    )
    print(df)

    # print(len(coco_train['images']))
    # print(len(coco_val['images']))
    # print(len(coco_test['images']))
    # print(len(coco_train['categories']))
    # print(len(coco_val['categories']))
    # print(len(coco_test['categories']))
    # print(len(coco_train['annotations']))
    # print(len(coco_val['annotations']))
    # print(len(coco_test['annotations']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing options")
    parser.add_argument(
        "--out_dir", type=str, default="", help="output dir of the dataset"
    )
    parser.add_argument(
        "--resize_to_smallest_gsd", action="store_true", help="Resize to smallest GSD"
    )
    parser.add_argument("--jpg", action="store_true", help="save as jpg instead of png")
    parser.add_argument(
        "--individual_image_test", action="store_true", help="Individual image test"
    )
    parser.add_argument("--blured", action="store_true", help="Apply blurring")
    parser.add_argument("--blacked", action="store_true", help="Apply blacking")
    parser.add_argument(
        "--strict", action="store_true", help="Only save patches with annotations"
    )
    parser.add_argument(
        "--corrected", action="store_true", help="use manually corrected annotations"
    )
    parser.add_argument("--dry", action="store_true", help="dont save anything")
    args = parser.parse_args()
    assert args.blured != args.blacked
    main(args)
