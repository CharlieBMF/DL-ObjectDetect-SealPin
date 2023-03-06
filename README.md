# DL-ObjectDetect-SealPin
Project to assess the emergence of defects on a product using DL methods, Object detecting, tensorflow, labelimg, raspberry pi, real time conection with machines at line, cv2.

<h1> Problem description </h3>
The product in simple terms comes in the form of a roller with a hole in the side surface. A pin is placed in this hole and then welded. As a result of the welding process, weld beads are sometimes formed at the location of the pin. These burrs are sharp, as a result of which the operator is sometimes injured - getting stuck in the finger. When such a defect occurs, it should be necessary to remove it with cutters. <br>
The goal of the project is to install on the production line a vision system built from a Raspberry Pi minicomputer with a connected camera and lens from Keyence. This system takes pictures and evaluates, makes detection if there is a burr in the image that could cause an accident using DL methods. A monitor connected directly to the RPI displays the photo along with where the defect was found and a red box indicating that the product is Not Good (NG), or a photo along with a green box indicating that the product is OK (OK). 

![image](https://user-images.githubusercontent.com/109242797/223128406-75077e16-5c15-4a1f-afee-4bb92c008b1f.png)

<h1> Sequence of work </h1> 
The resulting system must interact with the operator according to the following sequence of operation: <br>
<ol>
  <li> The operator places the item in the declining field in the machine. </li>
  <li>The machine uses a detection sensor to detect that a new workpiece is placed in the loading area, and also checks that the hands have been removed beyond the safety curtain. As a result, a signal - a marker - is triggered in the machine. </li>
  <li> A python script implemented on a Raspberry Pi microcomputer communicates with the machine in real time and checks the status of the marker. If it is lit it is a signal to take a picture. </li>
  <li> The lit marker triggers the taking of a picture using the PiCamera, a camera connected to the RPI. The photo is saved in an array in python script </li>
