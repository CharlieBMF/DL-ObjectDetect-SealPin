import pymcprotocol
from picamera import PiCamera
import cv2
import glob
import os
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from datetime import datetime
import requests
import psycopg2 as pg2
import pandas as pd
import json


def time_wrapper(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if end-start > 0.1:
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
        self.first_image = True
        self.actual_hour = datetime.now().hour
        self.possible_to_api_connect = False
        self.possible_to_FTP_connect = False

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

    def read_photo_trigger(self):
        self.connect()
        trigger_value = self.read_bits(head=self.trigger_address)
        self.close_connection()
        return trigger_value[0]

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

    def save_raw_image(self, image):
        name = 'img' + self.define_photo_number() + '.jpg'
        cv2.imwrite(name, image)

    def detect_defects(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = self.detection_model.detect(input_tensor)
        return detection_result

    def save_image_with_defects(self, image_with_defect):
        name = 'img' + str(int(self.define_photo_number())-1) + '_defects.jpg'
        cv2.imwrite(name, image_with_defect)

    def add_defects_to_raw_image(self, defects, image):
        for det in defects.detections:
            for cat in det.categories:
                print("{cname} - {score:.2f}%".format(cname=cat.category_name, score=cat.score * 100))
        image_with_defects = utils.visualize_defects(image, defects, w=self.image_width, h=self.image_height)
        return image_with_defects

    def add_ok_label_to_raw_image(self, image):
        image_without_defect = utils.visualize_ok_labels(image, w=self.image_width, h=self.image_height)
        return image_without_defect

    def create_detections_json(self, detections):
        detection_json = [{
            'barcode': self.define_photo_number(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': self.define_photo_number(),
            'image_path': 'C:/',
            'detections': [
                {
                    'bounding_box':
                        {
                            'origin_x': detect.bounding_box.origin_x,
                            'origin_y': detect.bounding_box.origin_y,
                            'width': detect.bounding_box.width,
                            'height': detect.bounding_box.height
                        },
                    'categories':
                        [
                            {
                                'index': detect.categories[0].index,
                                'score': detect.categories[0].score,
                                'display_name': detect.categories[0].display_name,
                                'category_name': detect.categories[0].category_name
                            }
                        ]
                }
                for detect in detections
            ]
        }]
        return detection_json

    def copy_localsql_to_api(self):
        conn = pg2.connect(database='pi', user='pi', password='pi')
        select_df = pd.read_sql('SELECT * from test', con=conn)
        print('Full df:', select_df)
        json_to_api = [json.loads(row) for row in select_df['name']]
        print('selected json from local to api:', json_to_api)
        self.report_detection_to_sql_by_api(json_to_api)

    @staticmethod
    def report_detection_to_sql_by_api(json_obj):
        # print('JSON OBJ BEFORE CONVER:', json_obj)
        # json_quotes_converted = json.loads(str(json_obj).replace("'", '"'))
        print('FINAL JSON TO API:', json_obj)
        response = requests.post('http://hamster.dsse.local', json=json_obj)
        print('API response:', response.status_code)
        if response.status_code != 200:
            raise Exception('Sorry response from API is not succesfull')

    @staticmethod
    def report_detection_to_local_sql(json_obj):
        detection_quotes_converted = str(json_obj).replace("'", '"')
        query = f"INSERT INTO test(name) VALUES ('{detection_quotes_converted}')"
        print('LocalSQL INSERT:', query)
        conn = pg2.connect(database='pi', user='pi', password='pi')
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        conn.close()

    @staticmethod
    def define_photo_number():
        list_of_files = glob.glob('*.jpg')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        number = str(
            int
            (latest_file.replace('img', '').replace('.jpg', '').replace('_defects', '')) + 1)
        print('Capturing photo...', number)
        return number
