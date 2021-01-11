import contextlib
import io
import logging
import os
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import json
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]

def load_text_json(test=True):
    dataset_dict = load_coco_json("total_text_train.json", "Images/Train", extra_annotation_keys=["utf8_string", "bbox", "polygon", "orientation"])
    if test:
        dataset_dict[:300]
    else:
        dataset_dict[300:]

    for i in dataset_dict:
        check_image_size(i, plt.imread(i['file_name']))
        for j in i['annotations']:
            assert len(j['segmentation'][0]) == 20
            j['bbox_mode'] = BoxMode.XYXY_ABS
            j['bbox'] = [float(f) for f in j['bbox']]
            j['bspline'] = j['segmentation'][0]
    return dataset_dict