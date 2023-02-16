import os

files = os.listdir(r'C:\Scripts\RPI-ES3-VS-SealPin\Tensorflow\workspace\training_workspace\images\images_1024x1024')

for file in files:
    if file.endswith('.jpg'):
        name = file.replace('.jpg', '')
        xml = name + '.xml'
        if xml in files:
            print(xml, '... EXIST!')
        else:
            os.remove(r'C:\Scripts\RPI-ES3-VS-SealPin\Tensorflow\workspace\training_workspace\images\images_1024x1024'+'\\'+file)
