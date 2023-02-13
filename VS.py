import pymcprotocol
from picamera import PiCamera
import glob
import os
import time


def time_wrapper(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Func {func.__name__} Time: {end-start}')
        return result
    return wrap


class VisionSystem:

    def __init__(self, id_line, id_machine, name, ip, port, addresses, target_network=None,
                 plc_id_in_target_network=None,):
        self.id_line = id_line
        self.id_machine = id_machine
        self.name = name
        self.ip = ip
        self.port = port
        self.target_network = target_network
        self.plc_id_in_target_network = plc_id_in_target_network
        self.machine = self.define_machine_root()
        self.trigger_address = addresses['Trigger_address']
        self.trigger_value = 0
        self.camera = PiCamera()

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

    @time_wrapper
    def take_photo(self):
        print('Trigger value is..', self.trigger_value)
        self.connect()
        trigger_value = self.read_bits(head=self.trigger_address)
        self.close_connection()
        if trigger_value[0] != self.trigger_value:
            self.trigger_value = trigger_value[0]
            print('Trigger value updated to..', self.trigger_value)
            if self.trigger_value == 1:
                name = 'img' + self.define_photo_number() + '.jpg'
                self.camera.capture(name)

    def set_camera_resolution(self, w, h):
        self.camera.resolution = (w, h)

    @staticmethod
    def define_photo_number():
        list_of_files = glob.glob('*.jpg')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        number = str(int(latest_file.replace('img', '').replace('.jpg', '')) + 1)
        print('Capturing photo...', number)
        return number

