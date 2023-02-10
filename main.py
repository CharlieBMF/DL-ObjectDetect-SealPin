from picamera import PiCamera


w, h = 320, 320
camera = PiCamera()
camera.resolution = (w, h)

for i in range(1,10):
    print('IN')
    camera.capture('img'+str(i)+'.jpg')
    print(i)


