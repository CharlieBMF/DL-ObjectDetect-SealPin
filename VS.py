
"""
Vision system module. Contains all the methods necessary for the vision system to function
"""

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
    """
    Wrapper for showing function name and time. Debbuging & controlling
    :param func: any function
    :return: wrapped function
    """
    def wrap(*args, **kwargs):
        print('Starting... ', func.__name__)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('Finished.. ', func.__name__)
        print(f'{func.__name__} TIME (sec) {end-start}\n')
        return result
    return wrap


class VisionSystem:
    """Class representing vision system"""

    def __init__(self, id_line: int, id_machine: int, name: str, ip: str, port: int, addresses: dict, image_width: int,
                 image_height: int, model_file_name: str, score_min_value: float, category_names: list,
                 target_network=None, plc_id_in_target_network=None, ):
        """
        Init for class
        :param id_line: id of the line on which the RaspberryPi used as a vision system is mounted
        :param id_machine: id of the machine on which the RaspberryPi used as a vision system is mounted
        :param name: name of the machine on which the RaspberryPi used as a vision system is mounted
        :param ip: ip the address at which the machine (PLC) to which the RaspberryPi is connected as a vision system
         is defined
        :param port: open port for communication between the RaspberryPi and the machine
        :param addresses: addresses used for communication between the RaspberryPi and the machine on which it is
         installed (PLC)
        :param image_width: width of the image taken by the vision system
        :param image_height: height of the image taken by the vision system
        :param model_file_name: name of the tflight file that is loaded as a model
        :param score_min_value: the minimum value for which the detection is detected
        :param category_names: list of category names to be detected by the model
        :param target_network: it is possible to communicate using other communication protocols to other machines
         on the line. If the vision system is connected to PLC, it is possible to perform routing on the PLC
          for a different network number and communicate with another PLC. This number determines to which network
           number routing should be performed
        :param plc_id_in_target_network:
        """

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
        self.barcode_read_finished = addresses['Barcode_read_finished']
        self.image_width = image_width
        self.image_height = image_height
        self.trigger_value = 0
        self.barcode_value = ''
        self.barcode_value_OK_read = False
        self.camera = PiCamera()
        self.set_camera_resolution()
        self.detection_model = self.initialize_model(model_file_name, score_min_value, category_names)
        self.first_image = True
        self.local_image_directory = os.path.join('/home/pi/Scripts', datetime.now().strftime('%Y-%m-%d'))
        if not os.path.isdir(self.local_image_directory):
            os.mkdir(self.local_image_directory)
        smbclient.ClientConfig(username="pythones3", password="Daicel@DSSE2023")
        smbclient.register_session("192.168.200.101", username="pythones3", password="Daicel@DSSE2023")
        self.samba_image_directory = os.path.join(
            '\\\\192.168.200.101\\vp_es3_ai\SEALPIN', datetime.now().strftime("%Y-%m-%d")
        )
        if not smbclient.path.isdir(self.samba_image_directory):
            smbclient.mkdir(self.samba_image_directory)
        self.samba_connection_available = self.check_samba_connection_availability()

    def define_machine_root(self):
        """
        PLC controller connection object definition
        :return: connection pymcprotocol object
        """
        pymc3e = pymcprotocol.Type3E()
        if self.target_network and self.plc_id_in_target_network:
            pymc3e.network = self.target_network
            pymc3e.pc = self.plc_id_in_target_network
        return pymc3e

    def connect(self):
        """
        Establish connection with PLC
        :return:
        """
        self.machine.connect(ip=self.ip, port=self.port)

    def close_connection(self):
        """
        Closing connection with PLC
        :return:
        """
        self.machine.close()

    def read_bits(self, head, size=1):
        """
        Reading bits from PLC in order
        :param head: initial bit address
        :param size: number of consecutive bits to read
        :return: list with bit values
        """
        return self.machine.batchread_bitunits(headdevice=head, readsize=size)

    def read_words(self, head, size=1):
        """
        Reading words from PLC in order
        :param head: initial word address
        :param size: number of consecutive words to read
        :return: list with words values
        """
        return self.machine.batchread_wordunits(headdevice=head, readsize=size)

    def read_random_words(self, word_devices, double_word_devices):
        """
        Reading words from PLC not in order
        :param word_devices: list of words to read
        :param double_word_devices: list of dwords to read
        :return: list of words/dwords values
        """
        return self.machine.randomread(word_devices=word_devices, dword_devices=double_word_devices)

    def set_camera_resolution(self):
        """
        Setting the height and width of the photo
        :return:
        """
        self.camera.resolution = (self.image_width, self.image_height)

    @time_wrapper
    def initialize_model(self, model_file_name: str, score_min_value: float, category_names: list):
        """
        Initialization of the trained model with the relevant options
        :param model_file_name: the name of the tflight file with the model
        :param score_min_value: minimum detection value used for defect detection
        :param category_names: categories detected in defect detection
        :return: model initialized with parameters
        """
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
    def define_directories(self):
        """
        Defined currently used access paths both on the SD card and on the samba server. In case of a new day,
         another folder is created in the format YYYY-MM-DD
        :return:
        """
        if not self.local_image_directory.endswith(datetime.now().strftime('%Y-%m-%d')):
            new_local_directory = os.path.join(self.local_image_directory[:-10], datetime.now().strftime('%Y-%m-%d'))
            if not os.path.isdir(new_local_directory):
                os.mkdir(new_local_directory)
            self.local_image_directory = new_local_directory
            new_samba_directory = os.path.join(self.samba_image_directory[:-10], datetime.now().strftime('%Y-%m-%d'))
            if not smbclient.path.isdir(new_samba_directory):
                smbclient.mkdir(new_samba_directory)
            self.samba_image_directory = new_samba_directory

    def read_photo_trigger(self):
        """
        Reading the trigger to take the picture
        :return: value of trigger to take a picture
        """
        self.connect()
        trigger_value = self.read_bits(head=self.trigger_address)
        self.close_connection()
        return trigger_value[0]

    def read_2d_reader_finish_work(self):
        """
        Reading the bit indicating that the serial number has been read
        :return: The value of the bit indicating that the serial number has been read
        """
        self.connect()
        barcode_read_finished_status = self.read_bits(head=self.barcode_read_finished)
        self.close_connection()
        print('barcode_read_finished_status', barcode_read_finished_status[0])
        return barcode_read_finished_status[0]

    @time_wrapper
    def read_barcode_value(self, timeout):
        """
        Reading the serial number value
        :param timeout: time in seconds in which the bit indicating that the serial number has been correctly read
        by the machine reader is expected to appear. The serial number value is converted to char.
        If the value starts with ER, which means ERROR when read, the serial number value is considered as another
        int value for the folder image
        :return:
        """
        barcode_send_OK_signal = False
        max_wait_time = time.time() + timeout
        while time.time() < max_wait_time:
            if self.read_2d_reader_finish_work() == 1:
                barcode_send_OK_signal = True
                time.sleep(0.1)
                break
        if barcode_send_OK_signal:
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
        else:
            self.barcode_value = self.define_photo_number()
            self.barcode_value_OK_read = False
        print('Final barcode:', self.barcode_value)

    @time_wrapper
    def define_raw_photo_name(self):
        """
        Defining the name of the raw photo to be saved. There are two cases.
        1. correct reading of the serial number by the serial number reader and by the script directly from the PLC.
        In this case, the name of the photo reflects the serial number. In the case where such a photo already exists,
         a suffix is added, which is an int number representing the next time a photo of the same piece is taken.
         2. Incorrect reading of serial number from plc or incorrect reading of serial number by serial number reader.
          In this case, the picture name is defined as another free int number.
        :return: raw image name
        """
        print('self.barcode_value_ok_read:', self.barcode_value_OK_read)
        if self.barcode_value_OK_read:
            '''This means in variable barcode_value is value like: SC4T3319258'''
            jpg_name = self.barcode_value + '.jpg'
            if smbclient.path.isfile(os.path.join(self.samba_image_directory, jpg_name)):
                for i in range(1, 1001):
                    print('i:', i)
                    temp_jpg_name = self.barcode_value + '_' + str(i) + '.jpg'
                    print('temp_jpg_name', temp_jpg_name)
                    if smbclient.path.isfile(os.path.join(self.samba_image_directory, temp_jpg_name)):
                        continue
                    else:
                        jpg_name = temp_jpg_name
                        break
            else:
                jpg_name = self.barcode_value + '.jpg'
        else:
            '''This means in variable barcode_value is raw number like: 1521 '''
            jpg_name = self.barcode_value + 'img.jpg'
        print('RAW JPG Name:', jpg_name)
        return jpg_name

    @time_wrapper
    def define_photo_number(self):
        """
        Defining the next free int number for the photo in case of incorrect reading of serial number
        :return: string number that should be given to the photo
        """
        if self.samba_connection_available:
            list_of_files = smbclient.listdir(self.samba_image_directory)
            list_of_images = [image for image in list_of_files if image.endswith('img.jpg')]
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

    @time_wrapper
    def save_raw_image(self, image):
        """
        Save the raw image
        :param image: image
        :return:
        """
        image_name = self.define_raw_photo_name()
        self.define_directories()
        image_path = os.path.join(self.local_image_directory, image_name)
        cv2.imwrite(image_path, image)

    @time_wrapper
    def detect_defects(self, image):
        """
        Passing the image through the model and detection of defects
        :param image: image
        :return: detection result presence of defects
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = self.detection_model.detect(input_tensor)
        return detection_result

    @time_wrapper
    def add_ok_label_to_raw_image(self, image):
        """
        Calling the photo processing function according to the ok pattern.
        Drawing a green frame around the photo and the OK text
        :param image: image to process
        :return: image with OK frame
        """
        image_without_defect = utils.visualize_ok_labels(image, w=self.image_width, h=self.image_height)
        return image_without_defect

    @time_wrapper
    def define_defects_photo_name(self):
        """
        Defining the name of the photo witg defects to be saved. There are two cases.
        1. correct reading of the serial number by the serial number reader and by the script directly from the PLC.
        In this case, the name of the photo reflects the serial number with '_defects'.
        In the case where such a photo already exists,a suffix is added, which is an int number representing the
        next time a photo of the same piece is taken.
         2. Incorrect reading of serial number from plc or incorrect reading of serial number by serial number reader.
          In this case, the picture name is defined as another free int number with '_defects'.
        :return: defect image name
        """
        print('self.barcode value ok read', self.barcode_value_OK_read)
        if self.barcode_value_OK_read:
            jpg_name = self.barcode_value + '_defects.jpg'
            print('jpg defect name:', jpg_name)
            print('jpg defect path', os.path.join(self.samba_image_directory, jpg_name))
            print(smbclient.path.isfile(os.path.join(self.samba_image_directory, jpg_name)))
            if smbclient.path.isfile(os.path.join(self.samba_image_directory, jpg_name)):
                for i in range(1, 1001):
                    temp_jpg_name = self.barcode_value + '_' + str(i) + '_defects.jpg'
                    if smbclient.path.isfile(os.path.join(self.samba_image_directory, temp_jpg_name)):
                        continue
                    else:
                        jpg_name = temp_jpg_name
                        break
            else:
                jpg_name = self.barcode_value + '_defects.jpg'
        else:
            jpg_name = self.barcode_value + 'img_defects.jpg'
        return jpg_name

    @time_wrapper
    def add_defects_to_raw_image(self, defects, image):
        """
        Calling the photo processing function according to the NG pattern with defects.
        Drawing a red frame around the photo and the NG text.
        Drawing bounding boxes with detection categories and score value
        :param defects: defects to be drawn
        :param image: image to process
        :return: image with NG label and bounding boxes
        """
        for det in defects.detections:
            for cat in det.categories:
                print("{cname} - {score:.2f}%".format(cname=cat.category_name, score=cat.score * 100))
        image_with_defects = utils.visualize_defects(image, defects, w=self.image_width, h=self.image_height)
        return image_with_defects

    @time_wrapper
    def save_image_with_defects(self, image_with_defect):
        """
        Save the defects image
        :param image_with_defect: image with drawn defects
        :return:
        """
        image_name = self.define_defects_photo_name()
        image_path = os.path.join(self.local_image_directory, image_name)
        print('defect image name, path:', image_name, image_path)
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

    @time_wrapper
    def report_detection_to_local_sql(self, json_obj):
        detection_quotes_converted = str(json_obj).replace("'", '"')
        query = f"INSERT INTO detections(detecion_json, time_stamp) VALUES" \
                f" ('{detection_quotes_converted}','{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')"
        print('LocalSQL INSERT:', query)
        self.commit_query_to_local_sql(query)

    def delete_top100_detections_from_local_sql(self):
        query = "DELETE FROM detections WHERE ctid IN (SELECT ctid FROM detections ORDER BY time_stamp LIMIT 100)"
        self.commit_query_to_local_sql(query)

    @time_wrapper
    def copy_images_to_samba_server(self):
        list_of_images_path = glob.glob(self.local_image_directory + '/*.jpg')
        list_of_images = [os.path.basename(x) for x in list_of_images_path]
        list_of_images_tuple = [(list_of_images_path[i], list_of_images[i]) for i in range(0, len(list_of_images_path))]
        for path, name in list_of_images_tuple:
            print(path, name)
            with open(path, 'rb') as local:
                with smbclient.open_file(self.samba_image_directory + '\\' + name, "wb") as remote:
                    shutil.copyfileobj(local, remote)

    @time_wrapper
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
