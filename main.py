from VS import VisionSystem
from conf import machines_names
import matplotlib.pyplot as plt
import numpy as np
import cv2


def initialize_plotting():
    wm = plt.get_current_fig_manager()
    wm.full_screen_toggle()
    plt.axis('off')
    plt.tight_layout()
    plt.ion()
    return plt.imshow(np.empty((1024, 1024, 3), dtype=np.uint8))


def show_image(drawing, image):
    drawing.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.2)


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
    trigger_value = seal_pin.read_photo_trigger()
    if trigger_value != seal_pin.trigger_value:
        seal_pin.trigger_value = trigger_value
        if seal_pin.trigger_value == 1:
            image = np.empty((seal_pin.image_width, seal_pin.image_height, 3), dtype=np.uint8)
            seal_pin.camera.capture(image, 'bgr')
            seal_pin.save_raw_image(image)
            defects = seal_pin.detect_defects(image)
            for det in defects.detections:
                print(det.bounding_box)
            if defects.detections:
                image_with_judgement = seal_pin.add_defects_to_raw_image(defects, image)
                seal_pin.save_image_with_defects(image_with_judgement)
            else:
                image_with_judgement = seal_pin.add_ok_label_to_raw_image(image)
            show_image(drawing, image_with_judgement)