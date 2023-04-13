# DL-ObjectDetect-SealPin
Project to assess the emergence of defects on a product using DL methods, Object detecting, tensorflow, labelimg, raspberry pi, real time conection with machines at line, cv2.

<h1> Problem description </h1>
The product in simple terms comes in the form of a roller with a hole in the side surface. A pin is placed in this hole and then welded. As a result of the welding process, weld beads are sometimes formed at the location of the pin. These burrs are sharp, as a result of which the operator is sometimes injured - getting stuck in the finger. When such a defect occurs, it should be necessary to remove it with cutters. <br>
The goal of the project is to install on the production line a vision system built from a Raspberry Pi minicomputer with a connected camera and lens from Keyence. This system takes pictures and evaluates, makes detection if there is a burr in the image that could cause an accident using DL methods. A monitor connected directly to the RPI displays the photo along with where the defect was found and a red box indicating that the product is Not Good (NG), or a photo along with a green box indicating that the product is OK (OK). 

![image](https://user-images.githubusercontent.com/109242797/223128406-75077e16-5c15-4a1f-afee-4bb92c008b1f.png)

<h1> Workstation design </h1>

![image](https://user-images.githubusercontent.com/109242797/230353967-f3dbf645-a8bd-4a17-9a1a-ba8ce5dbcc7d.png)

<h1> Sequence of work </h1> 
The resulting system must interact with the operator according to the following sequence of operation: <br>
<ol>
  <li> The operator places the item in the declining field in the machine. </li>
  <li>The machine uses a detection sensor to detect that a new workpiece is placed in the loading area, and also checks that the hands have been removed beyond the safety curtain. As a result, a signal - a marker - is triggered in the machine. </li>
  <li> A python script implemented on a Raspberry Pi microcomputer communicates with the machine in real time and checks the status of the trigger marker. If it is lit it is a signal to take a picture. </li>
  <li> The marker triggers the taking of a picture using the PiCamera, a camera connected to the RPI. The photo is saved in an array in python script </li>
  <li> Barcode reader connected to machine reads S/N of piece. </li>
  <li> A python script attempts to read 2d code directly from a PLC placed on the machine. The timeout for this operation is 3 seconds. If the s/n is read correctly, it is then used in the name of the photo stored on the SAMBA server. If the code is not read, the photo number is given in the structure (int)img.jpg by specifying the last photo number on the SAMBA server. </li>
  <li> The raw photo without processing is saved locally on the SD card in a folder represented as YYYY-MM-DD.  </li>
  <li> A photo previously stored in a numpy array is passed to be processed by a pre-trained model. The model checks for the presence of burrs in the photo. </li>
  <li> If no burrs are present in the photo, it is passed to the script responsible for drawing a green box around it with the word OK. Such a picture is displayed to the operator, it is allowed to put the part in the machine </li>
  <li> Only in the case of burr detection, the photo is saved with detection to the SD card. In the case of OK photo, it is not saved after processing. </li>
  <li> When a burr is detected the matter becomes more complicated. The priority becomes the timing of the image display for the operator. First, the raw photo along with the detection parameters is passed to a script that applies visual effects to the photo. The raw photo is processed, the detection area is superimposed on it, along with a red frame around it and the word NG. </li>
  
  ![image](https://user-images.githubusercontent.com/109242797/230356525-f874dd47-015b-4ab4-8ab1-f627e40580f5.png)

  <li> The processed photo is displayed on a display hooked up to the RPI visible to the operator. This allows him to see the need to trim the burrs if they are present. Besides, in the background, the reporting of the art is started by the following actions. </li>
  <li> A json object is created, which will be used for the detection report to the API. The API is configured as follows.
  
  ```python
  [
  {
    "barcode": "string",
    "created_at": "2023-04-06T10:58:12.862Z",
    "description": "string",
    "image_path": "string",
    "detections": [
      {
        "bounding_box": {
          "origin_x": 0,
          "origin_y": 0,
          "width": 0,
          "height": 0
        },
        "categories": [
          {
            "index": 0,
            "score": 0,
            "display_name": "string",
            "category_name": "string"
          }
        ]
      }
    ]
  }
]
  ```
  <li> First, the resulting json object is saved as a string in the local sql created on the RPI. This is done to secure the results when the connection to the external API breaks. 

![image](https://user-images.githubusercontent.com/109242797/230358551-5eeb73bd-4983-4f82-9893-413f806d4c82.png)

</li>
<li> At this point, both the photo and the detection json are stored locally. What follows is an attempt to flip these items to an external API and SAMBA server </li>
<li> The last 100 records are downloaded to the local sql. If the external API works correctly, only one (last) record will be in the local sql. If the connection to the API is broken, more records will be added to the local sql. Limiting the number of records to 100 prevents too many records from being reported to the API at the same time. Records will be dosed in packets of maximum 100. </li>
<li> In case of success response from api, 100 previously selected records from local sql are deleted. In case of no success nothing happens </li>
<li> What follows is an attempt to copy all photos from the SD card to an external SAMBA server to a folder compatible with the YYYY-MM-DD format.  </li>
<li> If writing to SAMBA is successful, the photos are deleted from the SD card. Otherwise, they are retained until they are correctly copied to the server. </li>
<li> Program ends its circuit and waits for the next signal to trigger the photo. </li>
 </ol>


<h1> Labeling </h1>
<p>LabelImg was used to mark the burrs to train the model. In the first stage, 3,000 images were collected, where about 650 burrs were found. These photos were labeled according to two categories - PR - this is a category indicating a large burr, and QA - a category indicating a smaller burr, not necessarily required to be removed, but the operator should pay attention to it. </p>
<p> After a positive first evaluation, the marking of the photos was made using only one category. The final model was learned using one burr category on a sample of 2k photos of specially selected burrs </p>

![image](https://user-images.githubusercontent.com/109242797/223373334-4914c0e7-91b2-4f19-a5ff-659d02fdeb0e.png)


<h1> Model </h1> 
<p>Because of the need for an edge solution, it was decided to use the MobileNet V2 FPNLite 640x640 SSD network. This is a good application for mobile systems, or RPI microcomputers. The processing time of the image in this case is very optimal, and sufficient accuracy is guaranteed. The system was also compared in the application of SSD ResNet50 V1 FPN 640x640 (RetinaNet50), however, due to the limited hardware for training the model, as well as its use, it turned out to be too extensive. In addition, in this case, model processing time is crucial. The operator, in cooperation with the script, must perform all the steps in a time not exceeding 7 seconds.  </p>

<h2> 120k steps </h2>

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

![image](https://user-images.githubusercontent.com/109242797/226926664-0070f28e-11b5-4f67-bcac-5290a02be50b.png)

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

<h2> 921k steps </h2>
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
python metadata_writer_for_object_detection.py --model_file mobilenet_v2_fpnlite_20230404.tflite --label_file RPI_label.txt --export_directory ./exported-tflite
```

<p>A diagram of the model formation is illustrated below. </p>

![image](https://user-images.githubusercontent.com/109242797/223389130-d4967d8a-3bf7-415a-a43b-5507885bc2d9.png)

<h1> Conf.py </h1>
<p> conf.py is the main configuration file for the script. It defines the way to access the machine on which the vision system is installed, the addresses used and, most importantly, the PLC addresses used for triggering the image, reading the barcode number and also reading the barcode itself. </p>

```python
machines_names = {
    'Gas_Generant':
        {
            'id_line': 33,
            'id_machine': 240,
            'name': 'GG',
            'ip': '192.168.10.161',
            'port': 40021,
            'address':
                {
                    'Trigger_address': 'M766',
                    'Barcode_read_finished': 'M767',
                    'Barcode_address': 'D654',

                },
            'target_network': None,
            'plc_id_in_target_network': None
        },
}
```

<h1> API </h1>

<p> The api was configured in the initial phase as one get and one post only. Post is used to record detections, while with get it is possible to obtain data of any historical element as follows </p>

![image](https://user-images.githubusercontent.com/109242797/231136103-0caf20a1-8d59-4adc-87ba-e98dc9a37c19.png)

![image](https://user-images.githubusercontent.com/109242797/231136244-1f8dfc1a-86d2-4f45-ac44-33b7ac161d8d.png)

<h1> SQL config </h1>

![image](https://user-images.githubusercontent.com/109242797/231774482-1e235f28-51e8-4678-95ab-a4f7df203298.png)


<h1> Results </h1> 

