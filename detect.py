import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.datasets import LoadImages, LoadWebcam

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better accuracy)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

def detect(opt):
    source, weights, imgsz, conf_thres, iou_thres = opt.source, opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres
    
    # Initialize
    device = 'cpu'
    half = False  # use FP16 half-precision inference
    
    # Load model
    print(f'Loading ONNX model from {weights}...')
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.enable_mem_pattern = True
    
    session = ort.InferenceSession(weights, sess_options, providers=['CPUExecutionProvider'])
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    
    # Dataloader
    if source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        dataset = LoadWebcam(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    
    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Preprocess
        img = letterbox(img, imgsz, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # Inference
        t1 = time.time()
        outputs = session.run(None, {input_name: img})
        t2 = time.time()
        
        # Post-process
        pred = outputs[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s.copy()
            
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {CLASSES[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{CLASSES[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=3)
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            
            # Stream results
            if dataset.mode == 'image':
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond
            else:  # 'video' or 'stream'
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                    raise StopIteration
    
    print(f'Done. ({time.time() - t0:.3f}s)')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/yolov5lite.onnx', help='model.onnx path')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt

def main(opt):
    detect(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
