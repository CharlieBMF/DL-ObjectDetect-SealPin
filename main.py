#!/usr/bin/python

"""
Main module to initialize vision system and run loop to start functionality
"""

from VS import VisionSystem
from conf import machines_names
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


def initialize_plotting():
    """
    Configure plot parameters to show photo for operator on line
    :return: plot with configured parameters and empty numpy array
    """
    wm = plt.get_current_fig_manager()
    wm.full_screen_toggle()
    plt.axis('off')
    plt.tight_layout()
    plt.ion()
    plt.show()
    return plt.imshow(np.empty((1024, 1024, 3), dtype=np.uint8))


def show_image(displayed_plot, image_to_show):
    """
    Use to plot image
    :param displayed_plot: configured drawing where to show a new image as ion
    :param image_to_show: image processed by defect detection
    :return: None
    """
    displayed_plot.set_data(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(1)


drawing = initialize_plotting()
seal_pin = VisionSystem(id_line=machines_names['Gas_Generant']['id_line'],
                        id_machine=machines_names['Gas_Generant']['id_machine'],
                        name=machines_names['Gas_Generant']['id_machine'],
                        ip=machines_names['Gas_Generant']['ip'],
                        port=machines_names['Gas_Generant']['port'],
                        addresses=machines_names['Gas_Generant']['address'],
                        image_width=1024,
                        image_height=1024,
                        model_file_name='mobilenet_v2_fpnlite_202303017.tflite',
                        score_min_value=0.40,
                        category_names=["PR", "QA"])

while True:
    # Detect trigger value
    start_time = time.time()
    trigger_value = seal_pin.read_photo_trigger()
    if trigger_value != seal_pin.trigger_value:
        # Detected trigger value change
        seal_pin.trigger_value = trigger_value
        if seal_pin.trigger_value == 1:
            # Detected trigger value change from 0 to 1, photo has to be executed
            image = np.empty((seal_pin.image_width, seal_pin.image_height, 3), dtype=np.uint8)
            seal_pin.camera.capture(image, 'bgr')
            # Read Barcode
            seal_pin.read_barcode_value(timeout=3)
            # Capture photo
            seal_pin.save_raw_image(image)
            # Save photo as raw image
            defects = seal_pin.detect_defects(image)
            # Detect if there are any defects on image
            if defects.detections:
                # Defect detected. Draw NG frame on image. Save Image with defect.
                image_with_judgement = seal_pin.add_defects_to_raw_image(defects, image)
                seal_pin.save_image_with_defects(image_with_judgement)
            else:
                # None defect detected. Draw OK frame on image.
                image_with_judgement = seal_pin.add_ok_label_to_raw_image(image)
            # Time priority to show image for operator
            show_image(drawing, image_with_judgement)
            if defects.detections:
                detection_time = time.time() - start_time
                # Background action to send detection info to localSQL and select last 100 sql rows
                detection_json = seal_pin.create_detections_json(defects.detections, detection_time)
                seal_pin.report_detection_to_local_sql(detection_json)
                detections_json = seal_pin.select_top100_detections_from_local_sql()
                try:
                    # Report detection to api if accessible
                    seal_pin.report_detection_to_api(detections_json)
                    print('DONE REPORT TO API')
                except:
                    pass
                else:
                    # DELETE TOP 100 from local sql cause it was reported to APIxx
                    print('TRYING DELETE TOP100')
                    seal_pin.delete_top100_detections_from_local_sql()
                    print('DONE DELETE TOP100')
            try:
                # Copy images from SD Card to samba server
                seal_pin.copy_images_to_samba_server()
                seal_pin.samba_connection_available = True
            except:
                seal_pin.samba_connection_available = False
            else:
                # Delete images if pass
                seal_pin.delete_images_from_local_SDCard()
