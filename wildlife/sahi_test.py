import argparse
import sys
import time
import subprocess
from ultralytics import YOLO

import cv2
import utils
import ncnn
import numpy as np
from sahi.predict import get_sliced_prediction

def pre_process(img):
    breakpoint()
    img = cv2.flip(img, 1) # Because we are using a webcam
    image_resized = cv2.resize(img, (640,640))
    # TODO: Apply sunlight reduction based on time of day
    blur = cv.GaussianBlur(image_resized, (10,10), cv.BORDER_DEFAULT)
    rgb_image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    return rgb_image


def run_script(camera_id=0, width=640, height=640):
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
    detector = YOLO('yolo11n_ncnn_model')

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1

        # Preprocess
        clean_image = pre_process(image)
        
        # Get Detection
        #detection_result = get_sliced_prediction(

        # Display frame
        annotated_frame = detection_result[0].plot()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', annotated_frame)

    cap.release()
    cv2.destroyAllWindows()

def main():
    run_script()

if __name__ == '__main__':
    main()
