"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor



_MARGIN = 50  # pixels
_ROW_SIZE = 30  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR_NG = (0, 0, 255)  # red
_TEXT_COLOR_OK = (0, 255, 0)


def visualize_defects(
        image: np.ndarray,
        detection_result: processor.DetectionResult,
        w: int,
        h: int,
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualized.
    w: Width of image.
    h: Height of image.

  Returns:
    Image with bounding boxes.
  """
    rotated_image = np.rot90(image, -1)
    rotated_image = np.ascontiguousarray(rotated_image, dtype=np.uint8)
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        # start_point = bbox.origin_x, bbox.origin_y
        # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        start_point = 1024 - bbox.origin_y - bbox.height, bbox.origin_x
        end_point = 1024 - bbox.origin_y, bbox.origin_x + bbox.width
        cv2.rectangle(rotated_image, start_point, end_point, _TEXT_COLOR_NG, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + 1024 - bbox.origin_y - bbox.height,
                         _MARGIN + _ROW_SIZE + bbox.origin_x)
        cv2.putText(rotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR_NG, _FONT_THICKNESS)

        # Draw border around an image
        cv2.rectangle(rotated_image, (0, 0), (w, h), _TEXT_COLOR_NG, 40)

        # Draw label for final score
        cv2.rectangle(rotated_image, (0, 0), (200, 140), _TEXT_COLOR_NG, -1)
        cv2.putText(rotated_image, "NG", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

    return rotated_image


def visualize_ok_labels(image, w, h):

    # Draw border around an image
    cv2.rectangle(image, (0, 0), (w, h), _TEXT_COLOR_OK, 40)

    # Draw label for final score
    cv2.rectangle(image, (0, 0), (200, 140), _TEXT_COLOR_OK, -1)
    cv2.putText(image, "OK", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

    return image
