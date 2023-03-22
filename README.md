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
<p>Because of the need for an edge solution, it was decided to use the MobileNet V2 FPNLite 640x640 SSD network. This is a good application for mobile systems, or RPI microcomputers. The processing time of the image in this case is very optimal, and sufficient accuracy is guaranteed. The system was also compared in the application of SSD ResNet50 V1 FPN 640x640 (RetinaNet50), however, due to the limited hardware for training the model, as well as its use, it turned out to be too extensive. In addition, in this case, model processing time is crucial. The operator, in cooperation with the script, must perform all the steps in a time not exceeding 7 seconds.  </p>
<h3> 120k steps </h3>

<p>In the first step, the model was trained on the basis of 120,000 runs with the most important pipeline settings as below.</p>

```python
train_config {
  batch_size: 2
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.07999999821186066
          total_steps: 120000
          warmup_learning_rate: 0.026666000485420227
          warmup_steps: 1000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 120000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}
```

<p>The model with 120k steps achieving results as follows.</p>

![image](https://user-images.githubusercontent.com/109242797/223375454-0d3da04a-0f6f-4427-8a4d-52fae3f1b9c4.png)

<h2> Second 240k steps </h2>

```python
train_config {
  batch_size: 2
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.07999999821186066
          total_steps: 240000
          warmup_learning_rate: 0.026666000485420227
          warmup_steps: 1000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 240000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}
```

![image](https://user-images.githubusercontent.com/109242797/226922735-3f09f61a-a864-4de1-bbd8-38f93df1aee1.png)

<h2> 921k steps </h3>
<p>Also for testing purposes, a model was created for more than 921,000 steps. With current equipment resources, the model was trained for about 52 hours. The results as below</p>

```python
train_config {
  batch_size: 2
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.08
          total_steps: 921600
          warmup_learning_rate: 0.026666
          warmup_steps: 1000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 921600
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}
```

![image](https://user-images.githubusercontent.com/109242797/226924222-580e9987-993f-48bc-a5f6-a5b50fb7de3a.png)

<p> The model thus created had to be converted to a tflite graph. The resulting conversion was again converted to tflite format. After completing the conversion with metadata describing the model, the final tflite model was obtained, which could be fired in an edge application. This model was directly placed on the RPI microcomputer, from where it is loaded once at each reboot. During the program's circulation, the model is used to predict burrs on new images. </p>
<p> In order to apply all the above conversions, a number of commands were applied using the appropriate python scripts as follows </p>

```python
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_mobilenet_v2_fpnlite_640x640_240000s\pipeline.config --trained_checkpoint_dir .\models\my_mobilenet_v2_fpnlite_640x640_240000s\ --output_directory .\exported-models\my_mobilenet_v2_fpnlite_640x640_240000steps
```

```python
python export_tflite_graph_tf2.py --pipeline_config_path ./exported-models/my_mobilenet_v2_fpnlite_640x640_240000steps/pipeline.config --trained_checkpoint_dir ./exported-models/my_mobilenet_v2_fpnlite_640x640_240000steps/checkpoint/ --output_directory ./exported-models/my_mobilenet_v2_fpnlite_640x640_240000s_tfgraph
```

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('exported-models/my_mobilenet_v2_fpnlite_640x640_240000s_tfgraph/saved_model/') # path to the SavedModel directory
# enable TensorFlow ops along with enable TensorFlow Lite ops
#converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
tflite_model = converter.convert()

# Save the model.
with open('mobilenet_v2_fpnlite_202303017.tflite', 'wb') as f:
  f.write(tflite_model)
```

```python
python metadata_writer_for_object_detection.py --model_file= mobilenet_v2_fpnlite_20230317.tflite --label_file= RPI_label --export_directory= ./exported-tflite
```

<p>A diagram of the model formation is illustrated below. </p>

![image](https://user-images.githubusercontent.com/109242797/223389130-d4967d8a-3bf7-415a-a43b-5507885bc2d9.png)



