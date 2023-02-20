import pymcprotocol
from picamera import PiCamera
import cv2
import glob
import os
import time
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def time_wrapper(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Func {func.__name__} Time: {end-start}')
        return result
    return wrap


class VisionSystem:

    def __init__(self, id_line, id_machine, name, ip, port, addresses, image_width, image_height, model_file_name,
                 score_min_value, category_names, target_network=None, plc_id_in_target_network=None,):
        self.id_line = id_line
        self.id_machine = id_machine
        self.name = name
        self.ip = ip
        self.port = port
        self.target_network = target_network
        self.plc_id_in_target_network = plc_id_in_target_network
        self.machine = self.define_machine_root()
        self.trigger_address = addresses['Trigger_address']
        self.image_width = image_width
        self.image_height = image_height
        self.trigger_value = 0
        self.camera = PiCamera()
        self.set_camera_resolution()
        self.detection_model = self.initialize_model(model_file_name, score_min_value, category_names)

    def define_machine_root(self):
        pymc3e = pymcprotocol.Type3E()
        if self.target_network and self.plc_id_in_target_network:
            pymc3e.network = self.target_network
            pymc3e.pc = self.plc_id_in_target_network
        return pymc3e

    def connect(self):
        self.machine.connect(ip=self.ip, port=self.port)

    def close_connection(self):
        self.machine.close()

    def read_bits(self, head, size=1):
        return self.machine.batchread_bitunits(headdevice=head, readsize=size)

    def read_words(self, head, size=1):
        return self.machine.batchread_wordunits(headdevice=head, readsize=size)

    def read_random_words(self, word_devices, double_word_devices):
        return self.machine.randomread(word_devices=word_devices, dword_devices=double_word_devices)

    def set_camera_resolution(self):
        self.camera.resolution = (self.image_width, self.image_height)

    @time_wrapper
    def initialize_model(self, model_file_name: str, score_min_value: float, category_names: list):
        base_options = core.BaseOptions(
            file_name=model_file_name,
            use_coral=False,
            num_threads=4
        )
        detection_options = processor.DetectionOptions(
            max_results=10,
            score_threshold=score_min_value,
            category_name_allowlist=category_names
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            detection_options=detection_options
        )
        detector = vision.ObjectDetector.create_from_options(options)
        return detector

    @time_wrapper
    def take_photo(self):
        cv2.destroyAllWindows()
        print('Trigger value is..', self.trigger_value)
        self.connect()
        trigger_value = self.read_bits(head=self.trigger_address)
        self.close_connection()
        if trigger_value[0] != self.trigger_value:
            self.trigger_value = trigger_value[0]
            print('Trigger value updated to..', self.trigger_value)
            if self.trigger_value == 1:
                image = np.empty((self.image_width, self.image_height, 3), dtype=np.uint8)
                self.camera.capture(image, 'rgb')
                self.save_raw_image(image)
                defects = self.detect_defects(image)
                if defects:
                    image_with_defect = self.add_defects_to_raw_image(defects, image)
                    self.save_image_with_defects(image_with_defect)

    def save_raw_image(self, image):
        name = 'img' + self.define_photo_number() + '.jpg'
        cv2.imwrite(name, image)

    def detect_defects(self, image):
        input_tensor = vision.TensorImage.create_from_array(image)
        detection_result = self.detection_model.detect(input_tensor)
        return detection_result

    def save_image_with_defects(self, image_with_defect):
        name = 'img' + self.define_photo_number() + '_defections.jpg'
        cv2.imwrite(name, image_with_defect)

    @staticmethod
    def add_defects_to_raw_image(self, defects, image):
        for det in defects.detections:
            for cat in det.categories:
                print("{cname} - {score:.2f}%".format(cname=cat.category_name, score=cat.score * 100))
        image_with_defects = utils.visualize(image, defects)
        cv2.imshow('DEFECTS', image_with_defects)
        return image_with_defects

    @staticmethod
    def define_photo_number():
        list_of_files = glob.glob('*.jpg')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        number = str(int(latest_file.replace('img', '').replace('.jpg', '')) + 1)
        print('Capturing photo...', number)
        return number

