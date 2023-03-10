# DL-ObjectDetect-SealPin
Project to assess the emergence of defects on a product using DL methods, Object detecting, tensorflow, labelimg, raspberry pi, real time conection with machines at line, cv2.

<h1> Problem description </h1>
The product in simple terms comes in the form of a roller with a hole in the side surface. A pin is placed in this hole and then welded. As a result of the welding process, weld beads are sometimes formed at the location of the pin. These burrs are sharp, as a result of which the operator is sometimes injured - getting stuck in the finger. When such a defect occurs, it should be necessary to remove it with cutters. <br>
The goal of the project is to install on the production line a vision system built from a Raspberry Pi minicomputer with a connected camera and lens from Keyence. This system takes pictures and evaluates, makes detection if there is a burr in the image that could cause an accident using DL methods. A monitor connected directly to the RPI displays the photo along with where the defect was found and a red box indicating that the product is Not Good (NG), or a photo along with a green box indicating that the product is OK (OK). 

![image](https://user-images.githubusercontent.com/109242797/223128406-75077e16-5c15-4a1f-afee-4bb92c008b1f.png)

<h1> Sequence of work </h1> 
The resulting system must interact with the operator according to the following sequence of operation: <br>
<ol>
  <li> The operator places the item in the declining field in the machine. </li>
  <li>The machine uses a detection sensor to detect that a new workpiece is placed in the loading area, and also checks that the hands have been removed beyond the safety curtain. As a result, a signal - a marker - is triggered in the machine. </li>
  <li> A python script implemented on a Raspberry Pi microcomputer communicates with the machine in real time and checks the status of the marker. If it is lit it is a signal to take a picture. </li>
  <li> The marker triggers the taking of a picture using the PiCamera, a camera connected to the RPI. The photo is saved in an array in python script </li>
  <li> An attempt is made to save the raw photo to the FTP server. If the attempt is successful the next step is performed. If there is an error in saving the photo to the FTP server, the photo is saved locally to the SD card in the RPI microcomputer. The name of the photo, which was saved locally due to a comminication error with the FTP server, is added to the corresponding table in the local SQL database. </li>
  <li> A photo previously stored in a numpy array is passed to be processed by a pre-trained model. The model checks for the presence of burrs in the photo. </li>
  <li> If no burrs are present in the photo, it is passed to the script responsible for drawing a green box around it with the word OK </li>
  <li> When burrs are detected, they are passed as an argument to the function that processes the image. A red frame with the word NG is drawn around the photo. In addition, the burr is marked along with the % grade in the photo. </li>
  <li> The processed photo (OK or NG) is displayed on a display hooked up to the RPI visible to the operator. This allows him to see the need to trim the burrs if they are present. </li>
  <li> If there are no burrs, an OK signal is sent to the machine. This signal causes the corresponding marker in the PLC to light up. </li>
  <li> For burr detection, the process is extended. What follows is an attempt to save the photo with the burrs marked on the FTP server, with the photo name including the suffix '_defects'. If there is no communication, saving to the local SD card takes place, analogous to the process with the raw photo. 
In addition, the burr information (detection window coordinates, category, % grade) is converted to JSON. What follows is an attempt to pass the above information to an external SQL server via the API. If the information is not written to the external SQL table, a write to the local sql database on the RPI is performed. </li>
  <li> On subsequent program rounds, entries in the local SQL database are checked. This applies to both the table associated with the FTP image storage and the table associated with the NG piece detection site. If communication with an external FTP or SQL server is already possible, the local entries are transferred to the outside and deleted from the local database. This guarantees the reliability of the system's operation even if connections to external services are broken. In this case, there is a transfer of information to the local base, and from there it is passed to the outside when communication is stabilized. </li>
  <li> If burrs occur, the operator must remove the piece from the machine, clean the burrs and repeat the process until the piece is determined to be OK </li>
 </ol>
 
 ![image](https://user-images.githubusercontent.com/109242797/223391530-2d4cd6d1-efa2-49ef-8bae-87f3282d4b67.png)


<h1> Labeling </h1>
<p>LabelImg was used to mark the burrs to train the model. In the first stage, 3,000 images were collected, where about 650 burrs were found. These photos were labeled according to two categories - PR - this is a category indicating a large burr, and QA - a category indicating a smaller burr, not necessarily required to be removed, but the operator should pay attention to it. </p>
<p> After a positive first evaluation, the marking of the photos was made using only one category. The final model was learned using one burr category on a sample of 2k photos of specially selected burrs </p>

![image](https://user-images.githubusercontent.com/109242797/223373334-4914c0e7-91b2-4f19-a5ff-659d02fdeb0e.png)


<h1> Model </h1> 
<p>Because of the need for an edge solution, it was decided to use the MobileNet V2 FPNLite 640x640 SSD network. This is a good application for mobile systems, or RPI microcomputers. The processing time of the image in this case is very optimal, and sufficient accuracy is guaranteed. The system was also compared in the application of SSD ResNet50 V1 FPN 640x640 (RetinaNet50), however, due to the limited hardware for training the model, as well as its use, it turned out to be too extensive. In addition, in this case, model processing time is crucial. The operator, in cooperation with the script, must perform all the steps in a time not exceeding 7 seconds. The model was trained achieving results as follows </p>

![image](https://user-images.githubusercontent.com/109242797/223375454-0d3da04a-0f6f-4427-8a4d-52fae3f1b9c4.png)

<p> In the training process, a suitable value of hyperparameters was sought in order to achieve the best possible burr evaluation result. By determining the appropriate hyperparameters, the model was learned. The model thus created had to be converted to a tflite graph. The resulting conversion was again converted to tflite format. After completing the conversion with metadata describing the model, the final tflite model was obtained, which could be fired in an edge application. This model was directly placed on the RPI microcomputer, from where it is loaded once at each reboot. During the program's circulation, the model is used to predict burrs on new images. A diagram of the model formation is illustrated below. </p>

![image](https://user-images.githubusercontent.com/109242797/223389130-d4967d8a-3bf7-415a-a43b-5507885bc2d9.png)


