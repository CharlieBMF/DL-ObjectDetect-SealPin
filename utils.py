"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
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
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR_NG, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR_NG, _FONT_THICKNESS)

        # Draw border around an image
        cv2.rectangle(image, (0, 0), (w, h), _TEXT_COLOR_NG, 40)

        # Draw label for final score
        cv2.rectangle(image, (0, 0), (200, 140), _TEXT_COLOR_NG, -1)
        cv2.putText(image, "NG", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

    return image


def visualize_ok_labels(image, w, h):

    # Draw border around an image
    cv2.rectangle(image, (0, 0), (w, h), _TEXT_COLOR_OK, 40)

    # Draw label for final score
    cv2.rectangle(image, (0, 0), (200, 140), _TEXT_COLOR_OK, -1)
    cv2.putText(image, "OK", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

    return image
