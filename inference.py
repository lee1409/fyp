from config.default import _C
import argparse
import os
import glob
import tqdm
from detectron2.data.detection_utils import read_image
import time
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from model import *


def setup_cfg(args):
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold

    cfg.MODEL.WEIGHTS = args.weights_file
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 for custom model")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights-file",
        metavar="FILE",
        help="Path to model weights"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print(args)

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get("__unused")
    predictor = DefaultPredictor(cfg)

    if args.input:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    for path in tqdm.tqdm(args.input, disable=not args.output):
        image = read_image(path, format="BGR")
        start_time = time.time()

        vis_output = None
        predictions = predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        visualizer = Visualizer(image)
        instances = predictions["instances"].to("cpu")
        vis_output = visualizer.draw_instance_predictions(
            predictions=instances)

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(
                    args.output, os.path.basename(path))
            else:
                assert len(
                    args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            vis_output.save(out_filename)
        else:
            assert "Please specify output dir"
