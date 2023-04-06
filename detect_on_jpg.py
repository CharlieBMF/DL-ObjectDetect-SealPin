#import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

imageName=sys.argv[1]

startTime = time.time()

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Initialize the object detection model
base_options = core.BaseOptions(
    file_name='mobilenet_v2_fpnlite_202303017.tflite',
    use_coral=False,
    num_threads=4
)
print("{0} - Base options".format(time.time()-startTime))
detection_options = processor.DetectionOptions(
    max_results=10,
    score_threshold=0.2,
    category_name_allowlist=["PR"]
)
print("{0} - Detection options".format(time.time()-startTime))
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    detection_options=detection_options
)
print("{0} - Set options".format(time.time()-startTime))
detector = vision.ObjectDetector.create_from_options(options)
print("{0} -Detector".format(time.time()-startTime))
image = cv2.imread("images/{imageName}".format(imageName=imageName))
print("{0} - Load image".format(time.time()-startTime))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("{0} - RGB image".format(time.time()-startTime))
input_tensor = vision.TensorImage.create_from_array(rgb_image)
print("{0} - Tensor".format(time.time()-startTime))
detection_result = detector.detect(input_tensor)
print("{0} - Detection".format(time.time()-startTime))
detCounter=0
for det in detection_result.detections:
  for cat in det.categories:
    print("{detNr}. {cname} - {score:.2f}%".format(detNr=detCounter,cname=cat.category_name,score=cat.score*100))
  detCounter+=1
image = utils.visualize(image, detection_result)
cv2.imshow('object_detector', image)
cv2.imwrite("{imageName}_detections.{extension}".format(imageName=imageName.split('.')[0],extension=imageName.split('.')[-1]),image)
cv2.waitKey(0)
