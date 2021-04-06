"""Download all images with sky on sun database
"""
import os, sys

project_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_path)
os.chdir(project_path)

from src.data_source.sun import find_all_cloud_sky_images, download_images


path_meta = "data/img_metadata/"
path_image = "data/img/sky_segmentation/"


def main():

    df_meta = find_all_cloud_sky_images(path_meta, save_intermediate=True)

    df_meta.to_csv(os.path.join(path_meta, "sun.csv"))

    download_images(df_img_meta=df_meta, out=path_image)


if __name__ == "__main__":

    main()
