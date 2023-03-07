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
  
