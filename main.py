from VS import VisionSystem
from conf import machines_names


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
    seal_pin.take_photo()
