# cloudFinder


## Data source

| name | url                                      | information                           | size                                                                             |
|------|------------------------------------------|---------------------------------------|--------------------------------------------------------------------------------------------|
| SUN  | https://groups.csail.mit.edu/vision/SUN/ | - images segmentation <br> - scene recognition | - 131067 Images <br> - 908 Scene categories <br> - 313884 Segmented objects <br> - 4479 Object categories |

## sky segmentation

The objective is to create a model that automatically find the sky in a picture.

The sun database is used for cloud segmentation. SUN database is a bank of image categorized by scene. Most of the images are segmented for object identification. For our purpose we download only the images with sky and cloud

#### prepare data

To download the image : 

```
from src.data_source.sun import find_all_cloud_sky_images, download_images

find_all_cloud_sky_images()

download_image()
```

#### create model 

The model creation process can be found in  `notebook/sky_segmentation.ipynb`

#### results

The results can be seen in `notebook/sky_segmentation_results.ipynb`

