# Sky segmentation

The objective is to create a model that automatically find the sky in a picture.

The sun database is used for sky segmentation. SUN database is a bank of image categorized by scene. Most of the images are segmented for object identification. For our purpose we download only the images with sky and cloud

## 1/ prepare data

To download the image : 

```
from src.data_source.sun import find_all_cloud_sky_images, download_images

find_all_cloud_sky_images()

download_image()
```

## 2/ create model 

The model creation process can be found in  `notebook/sky_segmentation.ipynb`

## 3/ results

The results can be seen in `notebook/sky_segmentation_results.ipynb`

## 4/ models

### Description

| model name | data selection                                              | features selection                                                                                             | model                                                             | Correction            |
|------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------|
| ss_model1    | - all SUN image if sky coverage > 5% <br> - resize at 100px | - 5 time bootsraping with 200 images <br> - 20 best (from bootsrap average <br> - using DecisionTreeClassifier | - 5 DecisionTreeClassifier(max_depth=10) <br> - VotingClassifier  | - opening with size 3 |
|            |                                                             |                                                                                                                |                                                                   |                       |

### Score


| model name                 | Accuracy | Recall | Precision |
|----------------------------|----------|--------|-----------|
| ss_model1 without correction | 0.898    | 0.776  | 0.863     |
| ss_model1 with correction    | 0.90     | 0.722  | 0.920     |


