from VS import VisionSystem
from conf import machines_names
import os
import glob

seal_pin = VisionSystem(id_line=machines_names['Gas_Generant']['id_line'],
                        id_machine=machines_names['Gas_Generant']['id_machine'],
                        name=machines_names['Gas_Generant']['id_machine'],
                        ip=machines_names['Gas_Generant']['ip'],
                        port=machines_names['Gas_Generant']['port'],
                        addresses=machines_names['Gas_Generant']['address'])
seal_pin.set_camera_resolution(1024, 1024)

while True:
    seal_pin.take_photo()
