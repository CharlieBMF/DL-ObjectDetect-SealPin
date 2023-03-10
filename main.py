import time

from VS import VisionSystem
from conf import machines_names
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2


def initialize_plotting():
    wm = plt.get_current_fig_manager()
    wm.full_screen_toggle()
    plt.axis('off')
    plt.tight_layout()
    plt.ion()
    plt.show()
    return plt.imshow(np.empty((1024, 1024, 3), dtype=np.uint8))


def show_image(displayed_plot, image_to_show):
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
                        model_file_name='mobilenet_v2_fpnlite_20230220.tflite',
                        score_min_value=0.20,
                        category_names=["PR"])

while True:
    # Detect trigger value
    trigger_value = seal_pin.read_photo_trigger()
    ######if trigger_value != seal_pin.trigger_value:
    if True:
        # Detected trigger value change
    ######    seal_pin.trigger_value = trigger_value
    ######    if seal_pin.trigger_value == 1:
        if True:
            # Detected trigger value change from 0 to 1, photo has to be executed
            image = np.empty((seal_pin.image_width, seal_pin.image_height, 3), dtype=np.uint8)
            seal_pin.camera.capture(image, 'bgr')
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
                # Background action to send detection info to SQL by API or local PostgreSQL if API not accesible
                detection_json = seal_pin.create_detections_json(defects.detections)
                try:
                    seal_pin.report_detection_to_sql_by_api(detection_json)
                    seal_pin.possible_to_api_connect = True
                except:
                    seal_pin.report_detection_to_local_sql(detection_json)
                    seal_pin.possible_to_api_connect = False

    #seal_pin.copy_localsql_to_api()

    actual_hour = datetime.now().hour
    # if actual_hour != seal_pin.actual_hour and seal_pin.possible_to_api_connect:
    #     try:
    #         seal_pin.copy_localsql_to_api()
    #     except:
    #         pass
    #
    # if actual_hour != seal_pin.actual_hour and seal_pin.possible_to_FTP_connect:
    #     pass
    #     # try:
    #     #     seal_pin.copy_local_image_to_FTP()
    seal_pin.actual_hour = actual_hour
    time.sleep(5)
