import cv2
import ncnn
import numpy as np
import time
import sys


class NCNNDetector:
    def __init__(self, param_path, bin_path):
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # Simple settings for stability
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_packing_layout = False

    def detect(self, frame):
        # Resize and preprocess
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) 
        breakpoint() 
        # NCNN inference
        mat_in = ncnn.Mat(img_float)
        mat_in0 = ncnn.Mat.from_pixels(img_float, ncnn.Mat.PixelType.PIXEL_RGB, 640,640)
        ex = self.net.create_extractor()
        
        # Try different input/output names if default fails
        ex.input("in0", mat_in0)
        ret, mat_out = ex.extract("out0")
        output_array = np.array(mat_out)
        print(f"Model output shape: {output_array.shape}")  # Debug info
        return output_array
    
    def postprocess(self, output, original_shape):
        h, w = original_shape[:2]
        
        if len(output.shape) == 3:
            output = output[0]
        
        print(f"Output shape after squeeze: {output.shape}")  # Debug
        
        # For single class model: [N, 6] = [N, 4_bbox + 1_conf + 1_class]
        if output.shape[1] == 6:  # Single class model
            # Filter by confidence (0.5 threshold)
            scores = output[:, 4]
            valid_detections = scores > 0.5
            
            if not np.any(valid_detections):
                return []
            
            valid_output = output[valid_detections]
            boxes = valid_output[:, :4]
            scores = valid_output[:, 4]
            # For single class, class_id is always 0
            class_ids = np.zeros(len(scores), dtype=int)
            
        else:  # Multi-class model
            scores = output[:, 4]
            valid_detections = scores > 0.5
            
            if not np.any(valid_detections):
                return []
            
            valid_output = output[valid_detections]
            boxes = valid_output[:, :4]
            scores = valid_output[:, 4]
            class_ids = np.argmax(valid_output[:, 5:], axis=1)
        
        # Convert to pixel coordinates and corner format
        boxes[:, 0] *= w  # x_center
        boxes[:, 1] *= h  # y_center
        boxes[:, 2] *= w  # width
        boxes[:, 3] *= h  # height
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.4)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                    'score': float(scores[i]),
                    'class_id': int(class_ids[i])
                })
        
        return results


def draw_detections(image, detections, class_name="Object"):
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        score = detection['score']
        
        # Draw box and label (since single class, just show the class name)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image


def main():
    # Initialize detector
    detector = NCNNDetector('anam_model/model.ncnn.param', 'anam_model/model.ncnn.bin')
    
    # Start camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        fps_counter += 1
        frame = cv2.flip(frame, 1)
        
        # Run detection
        output = detector.detect(frame)
        #detections = detector.postprocess(output, frame.shape)
        #annotated_frame = draw_detections(frame, detections, "YourClass")
        
        # Show FPS every 30 frames
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        # Display
        #cv2.imshow('NCNN Detection', annotated_frame)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
