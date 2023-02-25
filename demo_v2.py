# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import pickle

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo


# Fake a video capture object OpenCV style - half width, half height of first screen using MSS


class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {"top": 0, "left": 0, "width": m0["width"] / 2, "height": m0["height"] / 2}

    def read(self):
        img = np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True

    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; " "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. " "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--seg_output",
        default="",
        help="Generate a segmentation map of the scene and save it to the output directory.",
    )
    parser.add_argument(
        "--attention_bbox_path",
        default="",
        help="BBOX for object that we are interested at.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
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


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def generate_valid_segmentation_map(masks, scores, boxes, labels, attention_bbox=None, show_bbox=False):
    if attention_bbox and len(masks) > 0:
        # select the instance that has the highest IoU with the attention bbox
        ious = [iou(attention_bbox, box) for box in boxes]
        max_iou_idx = np.argmax(ious)
        max_iou_masks = masks[max_iou_idx]
        segmentation_map = np.zeros(masks.shape[1:], dtype=np.uint8)
        segmentation_map[max_iou_masks] = 255
        if show_bbox:
            cv2.rectangle(
                segmentation_map, (int(attention_bbox[0]), int(attention_bbox[1])), (int(attention_bbox[2]), int(attention_bbox[3])), (255, 0, 0), 2
            )
        return segmentation_map
    else:
        return np.zeros(masks.shape[1:], dtype=np.uint8)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"])) if "instances" in predictions else "finished",
                    time.time() - start_time,
                )
            )
            img_idx = int(os.path.splitext(os.path.basename(path))[0])
            print("Processing image {}".format(img_idx))
            # generate predict map
            if args.seg_output:
                # create directory if not exist
                seg_dir = os.path.dirname(args.seg_output)
                if seg_dir and (not os.path.exists(seg_dir)):
                    os.makedirs(seg_dir)
                masks = predictions["instances"].pred_masks.cpu().numpy()
                scores = predictions["instances"].scores.cpu().numpy()
                boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
                labels = predictions["instances"].pred_classes.cpu().numpy()

                # get attention bbox
                if args.attention_bbox_path:
                    with open(args.attention_bbox_path, "r") as f:
                        attention_bbox_lines = f.readlines()
                    attention_bbox = attention_bbox_lines[img_idx].split("\t")
                    attention_bbox = [int(x) for x in attention_bbox]

                    # rotate bbox if height > width
                    x, y, w, h = attention_bbox
                    height, width = img.shape[:2]
                    # attention_bbox = [width - (y + h), x, width - y, (x + w)]
                    attention_bbox = [x, y, (x + w), (y + h)]

                    segmentation_map = generate_valid_segmentation_map(masks, scores, boxes, labels, attention_bbox, False)
                    # save segmentation map as image
                    if os.path.isdir(args.seg_output):
                        out_filename = os.path.join(args.seg_output, os.path.basename(path))
                        cv2.imwrite(out_filename, segmentation_map)
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.seg_output"
                        out_filename = args.seg_output
                        cv2.imwrite(out_filename, segmentation_map)

            if args.output:
                # create directory if not exist
                output_dir = os.path.dirname(args.output)
                if output_dir and (not os.path.exists(output_dir)):
                    os.makedirs(output_dir)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        if args.webcam == "screen":
            cam = ScreenGrab()
        else:
            cam = cv2.VideoCapture(int(args.webcam))
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
