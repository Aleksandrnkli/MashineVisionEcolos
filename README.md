MTCNN IN THE PIPELINE
-------------------------------------------------------------------------------------------------
In order to prepare data for MTCNN located in E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class 
should be converted. 
The convertation process consists of the following steps: 
 - run script gen_Pnet_train_data.py to generate dataset for PNet
 - (after training PNet) run script gen_Onet_train_data.py to generate dataset for ONet

In order to train MTCNN dataset:
 - run script Train_Pnet.py to train PNet
 - run script Train_Onet.py to train ONet
Net are trained with the default Adam's parameters which are lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
during 16 epochs. Both networks achieved ~0.9 accuracy (calculated percent of detected LPs out of total number taken from ground true)
Weights are in E:\Models\LPR\MTCNN_1cls, files onet_Weights and pnet_Weights

MTCNN integrated (although some refactoring and tweaks are still required) into the Pipeline. Script 
pipeline_test_dataset.py Classification RU LPs vs KZ LPs is implemented.
Weights for 1 class (Russian's LPs only) is located in E:\Models\LPR\MTCNN_1cls
Weights for 2 classes (RU LPs vs KZ LPs are told apart via MTCNN classification) is located in E:\Models\LPR\MTCNN_2cls


DATASET NAMING CONVENTION
-------------------------------------------------------------------------------------------------
During annotation/training NN iterations we merged (many times) different datasets from Web and 
our own and there was too many problems with training cause of different naming of files and its
extension (time consuming filtration). So we empirically decided to use the following plain naming:
Images extension - .JPG
Name itself - 9-digit number with leading zeros (eg. 000000001.jpg)
Script for rename - dataset_rename_by_convention.py (./nn/Scripts)



PLATES DETECTION (RetinaNet)
-------------------------------------------------------------------------------------------------
Weights path - E:/Models/LPR/LPDetection_test20022021/pascal_custom_weights
Dataset path - E:/DataSets/LicensePlate/Russian_KZ_LPR_boundaries_include
Samples - 10400
Learning rate - 1e-6
Batch size - 10
Optimizer - Adam

Since we work with the LabelImg utility and there is no special xml attribute for plates number,
we add a plate number as a class name in annotations. Then we need to run the replace_anno_path.py 
(./nn/Scripts) script to convert a class name to 'plate number' and save the plate number itself 
in the custom attribute 'plate' in xml




PLATES RECONGITION (STN & LPRNet(Beam Search inside))
-------------------------------------------------------------------------------------------------
Weights path - E:/Models/LPR/LPDetection_test20022021 (prefix for each net)
Dataset path - D:/PyCharmProjects/LPR_Pipeline/lprnet/Cropped
Samples - more than 50000 in summ
Loss fn - CTC
Batch size - 128
Optimizer - Adam

To crop full images with the plates number (Detection dataset) you need to run the 
crop_lpr_for_recognizer.py (./nn/license_plate_detection_recognition/retinanet) 





OLD WEIGHTS
-------------------------------------------------------------------------------------------------
Plates detection (prev iter) - E:/Models/LPR/LPDetection29.01(7300)
Plates detection (prev-prev iter) - E:/Models/LPR/LPDetection27.01(4500)
Plates detection (huge bounding boxes) - E:/Models/LPR/LPDetection24.01.hugebox





PIPELINE TESTING
-------------------------------------------------------------------------------------------------
Pipeline - python class included scripts for testing all trained NNs on real data
(Detector - Retina(or MTCNN) --> Transformation network - STN --> Reconizer - LPR)  

Testing pipeline on dataset - ./nn/license_plate_detection_recognition/pipeline_demo_video.py  
to start from cmd: python pipeline_demo_video.py --stnetru_checkpoint_path STNETRU_CHECKPOINT_PATH --stnetkz_checkpoint_path STNETKZ_CHECKPOINT_PATH --lprnetru_checkpoint_path LPRNETRU_CHECKPOINT_PATH --lprnetkz_checkpoint_path LPRNETKZ_CHECKPOINT_PATH --onet_checkpoint_path ONET_CHECKPOINT_PATH --pnet_checkpoint_path PNET_CHECKPOINT_PATH --file_full_name FILE_FULL_NAME  

Testing pipeline on video - ./nn/license_plate_detection_recognition/pipeline_test_dataset.py   
to start from cmd: python pipeline_test_dataset.py --stnetru_checkpoint_path STNETRU_CHECKPOINT_PATH --stnetkz_checkpoint_path STNETKZ_CHECKPOINT_PATH --lprnetru_checkpoint_path LPRNETRU_CHECKPOINT_PATH --lprnetkz_checkpoint_path LPRNETKZ_CHECKPOINT_PATH --onet_checkpoint_path ONET_CHECKPOINT_PATH --pnet_checkpoint_path PNET_CHECKPOINT_PATH --file_full_name FILE_FULL_NAME  

Start pipeline with web-interface and saving detections - ./nn/license_plate_detection_recognition/pipeline_post_detections.py  
to start from cmd: python pipeline_post_detections.py --stnetru_checkpoint_path STNETRU_CHECKPOINT_PATH --stnetkz_checkpoint_path STNETKZ_CHECKPOINT_PATH --lprnetru_checkpoint_path LPRNETRU_CHECKPOINT_PATH --lprnetkz_checkpoint_path LPRNETKZ_CHECKPOINT_PATH --onet_checkpoint_path ONET_CHECKPOINT_PATH --pnet_checkpoint_path PNET_CHECKPOINT_PATH --dataset_path DATASET_PATH





TESTING STN ON DATASET
-------------------------------------------------------------------------------------------------
Script for visualizing cropped LP immages spatial transoformation
Ecolos/nn/license_plate_detection_recognition/lprnet/lpr_view_on_dataset.py





LABELIMG - the installation process and other significant points are described below
-------------------------------------------------------------------------------------------------
https://github.com/tzutalin/labelImg
Prerequisites - Python (o rly?), PyQt (I suggest pyqt5, so - pip install pyqt5)
and lxml (pip install lxml)

In the labelImg folder in CMD:
For pyqt4 - pyrcc4 -o lib/resources.py resources.qrc
For pyqt5 - pyrcc5 -o libs/resources.py resources.qrc

If at this point you see 'PyQt5 is not recognized as an internal command' message, try this:
1) Check whether Python/Scripts folder is present in PATH OS env variable
2) Install PyQt5-tools via pip install PyQt5-tools

Start - python labelImg.py

There was also a Windows-related bug that caused the app to crash. It happened due to the fact 
that one of the app methods tried to split the path string (annotations dir) through slashes (/).
But we all know that Windows uses backslashes (\) for the path. So when the method tried to add 
chunks of the path to a Python list and then access it's index 3 (or near it), the app crashed 
due to the unhandled IndexError exception 'cause there was no such index. Unfortunately, I could not 
reproduce this bug just before writing the README. But if it will persists, you just need to look 
at the name of the method in the stack trace, find the line through debug and slightly change it.
I will update this README if will catch the bug again

----------------------------------------------------------------------------------------------------



NUMPY (and Tensorflow since it implicitly install Numpy) - brief remark
-----------------------------------------------------------------------------------------------------
At the moment of writing this README, there is one known issue with a Numpy installation on Windows 
(here we go again). The most recent version of the library (1.19.4 at the moment) fails to import itself 
and raise some exceptions. Plain workaround - install more older version (1.19.3 for instance)

------------------------------------------------------------------------------------------------------



ECOLOS RTSP CAMERA FOR AUTO 
------------------------------------------------------------------------------------------------------
Link: rtsp://admin:admin@10.1.2.224:554
Web-interface available only in Internet Explorer 'cause its use deprecated plugins

Recommendation on camera FPS and resolution: usage in RetinaNet detector showed that with 25 FPS the script
does not have time to process all frames and crashes, so settings selected empirically as follows: 
7 FPS, 1080p resolution



AUXILIARY SCRIPTS
---------------------------------------------------------------------------------------------------------
In the process of development, we wrote auxiliary scripts to help solve some problems of processing and forming a dataset.

Path to the folder with scripts: EcoSmart/Desktop/pipeline/ecosmart/Ecolos/nn/Scripts

List of scripts:  

1)add_plate_tag.py - Helps to transfer plate tag with license plate into annotation if this information is lost.  
 to start from cmd: python add_plate_tag.py --path_without_tag="path to annotations withot tag" --path_with_tag="path to annotations with tag"   
3)filter_numbers_and_change_class.py - Derived from the previous one, helps to change or set class you need, also removing annotations where is no plate tag.    
 to start from cmd: python filter_numbers_and_change_class.py --path="path to annotations" --class_name="RU or KZ"   
4)create_list_of_annotations.py - Creates txt file files.txt - list of the files located in folder indicated on.    
 to start from cmd: python create_list_of_annotations.py --path="path to annotations"  
5)dataset_auxiliary_funcs.py - consist of 4 methods: converting from xml to csv, replacing xml annotation path, deleting images without annotations and changing annotation class.    
6)dataset_rename_by_convention.py - Renames all samples according to the desired layout.  
 to start from cmd: python dataset_rename_by_convention.py --path="path to images"  
7)filter_cropped_LPR_by_name.py - Filteres cropped photos for LPR, checking for valid characters.  
 to start from cmd: python filter_cropped_LPR_by_name.py --path="path to images"  --class_name="RU or KZ"  
8)remove_without_tag.py - Checks annotations for empty objects or empty plate tag, removes if they're empty.  
  to start from cmd: python remove_without_tag.py --path="path to annotations"   
9)filter_photos_by_resolution.py - Checks resolurion of samples, removes if resolution is less than 640X480.  
  to start from cmd: python filter_photos_by_resolution.py --path="path to images"  
10)remove_without_photo.py - Removes annotation without image/ Removes image withot annotation.  
  to start from cmd: python remove_without_photo.py --anno_path="path to annotations" --images_path="path to images"  
11)rotate_and_resize.py - rotates and resizes photo if needed    
12)replace_anno_class.py - Change name tag to "RU or KZ", put number into plate tag  
  to start from cmd: python replace_anno_class.py --anno_path="path to annotations" --anno_class="RU or KZ"  




