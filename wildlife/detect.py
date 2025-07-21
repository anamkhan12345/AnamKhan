# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import subprocess
from ultralytics import YOLO

import cv2
import utils
import ncnn
import numpy as np


class ncnn_detect():
    def __init__(self, param_path, bin_path):
        
        # Load a YOLO11n PyTorch model
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # Set threading and optimization options for Raspberry Pi
        self.net.opt.use_vulkan_compute = False  # Disable Vulkan on RPi
        self.net.opt.use_fp16_packed = False     # Disable FP16 on RPi
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_packing_layout = False
        self.net.opt.num_threads = 1

    def detect(self, frame):
        h, w = frame.shape[:2]

        # Preprocess
        img_resized = cv2.resize(frame, (640, 640))
        
        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Create NCNN Mat from numpy array
        mat_in = ncnn.Mat(img_float)

        # Inference
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)  # Changed from "images" to "in0"
        ret, mat_out = ex.extract("out0")

        return np.array(mat_out)


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    param_path = 'model/model.ncnn.param'
    bin_path = 'model/model.ncnn.bin'
    detector = ncnn_detect(param_path, bin_path) 

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Get Detection
        detection_result = detector.detect(image)


        # Display frame
        annotated_frame = detection_result[0].plot()
 
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId',
        help='Id of camera.',
        required=False,
        type=int,
        default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
        int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
    main()
