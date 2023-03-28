from picamera import PiCamera
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from datetime import datetime
import utils
import cv2
import glob
import os
import time
import shutil
import pymcprotocol
import requests
import psycopg2 as pg2
import pandas as pd
import json
import smbclient


def time_wrapper(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if end - start > 0.1:
            print(f'Func {func.__name__} Time: {end - start}')
        return result

    return wrap


class VisionSystem:

    def __init__(self, id_line: int, id_machine: int, name: str, ip: str, port: int, addresses: dict, image_width: int,
                 image_height: int, model_file_name: str, score_min_value: float, category_names: list,
                 target_network=None, plc_id_in_target_network=None, ):
        self.id_line = id_line
        self.id_machine = id_machine
        self.name = name
        self.ip = ip
        self.port = port
        self.target_network = target_network
        self.plc_id_in_target_network = plc_id_in_target_network
        self.machine = self.define_machine_root()
        self.trigger_address = addresses['Trigger_address']
        self.barcode_address = addresses['Barcode_address']
        self.image_width = image_width
        self.image_height = image_height
        self.trigger_value = 0
        self.barcode_value = ''
        self.barcode_value_OK_read = False
        self.camera = PiCamera()
        self.set_camera_resolution()
        self.detection_model = self.initialize_model(model_file_name, score_min_value, category_names)
        self.first_image = True
        self.local_image_directory = '/home/pi/Scripts/' + datetime.now().strftime('%Y-%m-%d')
        if not os.path.isdir(self.local_image_directory):
            os.mkdir(self.local_image_directory)
        smbclient.register_session("192.168.200.101", username="pythones3", password="Daicel@DSSE2023")
        self.samba_image_directory = '\\\\192.168.200.101\\vp_es3_ai\SEALPIN\\' + datetime.now().strftime("%Y-%m-%d")
        if not smbclient.path.isdir(self.samba_image_directory):
            smbclient.mkdir(self.samba_image_directory)
        self.samba_connection_available = self.check_samba_connection_availability()

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

    def define_directories(self):
        if not self.local_image_directory.endswith(datetime.now().strftime('%Y-%m-%d')):
            new_local_directory = self.local_image_directory[:-10] + datetime.now().strftime('%Y-%m-%d')
            if not os.path.isdir(new_local_directory):
                os.mkdir(new_local_directory)
            self.local_image_directory = new_local_directory
            new_samba_directory = self.samba_image_directory[:-10] + datetime.now().strftime('%Y-%m-%d')
            if not smbclient.path.isdir(new_samba_directory):
                smbclient.mkdir(new_samba_directory)
            self.samba_image_directory = new_samba_directory

    def read_photo_trigger(self):
        self.connect()
        trigger_value = self.read_bits(head=self.trigger_address)
        self.close_connection()
        return trigger_value[0]

    def read_barcode_value(self):
        try:
            self.connect()
            barcode_decimal_list = self.read_words(head=self.barcode_address, size=6)
            self.close_connection()
            print('Barcode:', barcode_decimal_list)
            barcode_binary = [bin(i).replace('b', '') for i in barcode_decimal_list]
            barcode_binary[0] = barcode_binary[0][1:]
            barcode_binary[1] = barcode_binary[1][1:]
            print('Barcode:', barcode_binary)
            barcode_ASCII_separated = list(map(
                lambda x: str(chr(int(x[0:7], 2)) + chr(int(x[7:16], 2))), barcode_binary[:-1]
            ))
            barcode_ASCII_swapped = list(map(lambda x: str(x[1] + x[0]), barcode_ASCII_separated))
            barcode_ASCII_swapped += chr(int(barcode_binary[-1], 2))
            print(barcode_ASCII_swapped)
            self.barcode_value = ''.join(barcode_ASCII_swapped)
            if self.barcode_value.startswith('ER'):
                raise Exception('Error read in barcode number')
        except:
            self.barcode_value = self.define_photo_number()
            self.barcode_value_OK_read = False
        else:
            self.barcode_value_OK_read = True
        print('Final barcode:', self.barcode_value)

    def define_raw_photo_name(self):
        print('self.barcode_value_ok_read:', self.barcode_value_OK_read)
        if self.barcode_value_OK_read:
            '''This means in variable barcode_value is value like: SC4T3319258'''
            jpg_name = self.barcode_value + '.jpg'
            if os.path.isfile(jpg_name):
                for i in range(1, 1001):
                    print('i:', i)
                    temp_jpg_name = self.barcode_value + '_' + str(i) + '.jpg'
                    if os.path.isfile(temp_jpg_name):
                        continue
                    else:
                        jpg_name = temp_jpg_name
                        break
        else:
            '''This means in variable barcode_value is raw number like: 1521 '''
            jpg_name = self.barcode_value + 'img.jpg'
        print('RAW JPG Name:', jpg_name)
        return jpg_name

    def define_photo_number(self):
        if self.samba_connection_available:
            list_of_files = smbclient.listdir(self.samba_image_directory)
            list_of_images = [image for image in list_of_files if image.endswith('.jpg')]
            print('SELF DIRECTORY:', self.samba_image_directory)
            print('FILES IN DIRECTORY:', list_of_images)
            if list_of_images:
                list_of_int = [int(x.replace('img.jpg', '').replace('img_defects.jpg', '')) for x in list_of_images]
                max_number = max(list_of_int)
                number = str(max_number + 1)
            else:
                number = str(1)
        else:
            list_of_files = glob.glob(self.local_image_directory + '/*img.jpg')
            latest_file = max(list_of_files, key=os.path.getctime)
            number = str(
                int
                (latest_file.replace('img.jpg', '')) + 1)
        print('Actual number...', number)
        return number

    def save_raw_image(self, image):
        image_name = self.define_raw_photo_name()
        self.define_directories()
        image_path = self.local_image_directory + '/' + image_name
        cv2.imwrite(image_path, image)

    def detect_defects(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = self.detection_model.detect(input_tensor)
        return detection_result

    def add_ok_label_to_raw_image(self, image):
        image_without_defect = utils.visualize_ok_labels(image, w=self.image_width, h=self.image_height)
        return image_without_defect

    def define_defects_photo_name(self):
        if self.barcode_value_OK_read:
            jpg_name = self.barcode_value + '.jpg'
            if os.path.isfile(jpg_name):
                for i in range(1, 1001):
                    temp_jpg_name = self.barcode_value + '_' + str(i) + '.jpg'
                    if os.path.isfile(temp_jpg_name):
                        continue
                    else:
                        jpg_name = self.barcode_value + '_' + str(i - 1) + '_defects.jpg'
        else:
            jpg_name = self.barcode_value + 'img_defects.jpg'
        return jpg_name

    def add_defects_to_raw_image(self, defects, image):
        for det in defects.detections:
            for cat in det.categories:
                print("{cname} - {score:.2f}%".format(cname=cat.category_name, score=cat.score * 100))
        image_with_defects = utils.visualize_defects(image, defects, w=self.image_width, h=self.image_height)
        return image_with_defects

    def save_image_with_defects(self, image_with_defect):
        image_name = self.define_defects_photo_name()
        image_path = self.local_image_directory + '/' + image_name
        cv2.imwrite(image_path, image_with_defect)

    def create_detections_json(self, detections, detection_time):
        detection_json = {
            'barcode': self.barcode_value,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': str(detection_time),
            'image_path': self.samba_image_directory + '\\' + self.barcode_value + '.jpg',
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
        }
        return detection_json

    def report_detection_to_local_sql(self, json_obj):
        detection_quotes_converted = str(json_obj).replace("'", '"')
        query = f"INSERT INTO detections(detecion_json, time_stamp) VALUES" \
                f" ('{detection_quotes_converted}','{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')"
        print('LocalSQL INSERT:', query)
        self.commit_query_to_local_sql(query)

    def delete_top100_detections_from_local_sql(self):
        query = "DELETE FROM detections WHERE ctid IN (SELECT ctid FROM detections ORDER BY time_stamp LIMIT 100)"
        self.commit_query_to_local_sql(query)

    def copy_images_to_samba_server(self):
        list_of_images_path = glob.glob(self.local_image_directory + '/*.jpg')
        list_of_images = [os.path.basename(x) for x in list_of_images_path]
        list_of_images_tuple = [(list_of_images_path[i], list_of_images[i]) for i in range(0, len(list_of_images_path))]
        for path, name in list_of_images_tuple:
            print(path, name)
            with open(path, 'rb') as local:
                with smbclient.open_file(self.samba_image_directory + '\\' + name, "wb") as remote:
                    shutil.copyfileobj(local, remote)

    def delete_images_from_local_SDCard(self):
        list_of_images_path = glob.glob(self.local_image_directory + '/*.jpg')
        for image in list_of_images_path:
            os.remove(image)

    @staticmethod
    def check_samba_connection_availability():
        if smbclient.path.isdir('\\\\192.168.200.101\\vp_es3_ai\SEALPIN'):
            return True
        else:
            return False

    @staticmethod
    def select_top100_detections_from_local_sql():
        query = "SELECT * FROM detections LIMIT 100;"
        conn = pg2.connect(database='pi', user='pi', password='pi')
        select_df = pd.read_sql(query, con=conn)
        print('SELECTED DF:', select_df)
        detections_json = [json.loads(row) for row in select_df['detecion_json']]
        print('SELECTED FROM LOCALSQL\n', detections_json)
        return detections_json

    @staticmethod
    def commit_query_to_local_sql(query):
        conn = pg2.connect(database='pi', user='pi', password='pi')
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        conn.close()

    @staticmethod
    def report_detection_to_api(json_obj):
        print('FINAL JSON TO API:', json_obj)
        response = requests.post('http://hamster.dsse.local/Vision/SendData', json=json_obj)
        print('Response text:', response.text)
        print('Response succes', json.loads(response.text)["sucess"])
        if not json.loads(response.text)["sucess"]:
            raise Exception('Sorry response from API is not succesfull')
