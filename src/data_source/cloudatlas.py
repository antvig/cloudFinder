from bs4 import BeautifulSoup
import tqdm
import requests
import json
import wget
import pandas as pd
import time
from selenium import webdriver
import uuid
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

URL = "https://cloudatlas.wmo.int/"


def find_cloud_images(out, save=True):
    url_path_img = "search-image-gallery.html"

    driver = webdriver.Chrome()
    driver.get(URL + url_path_img)
    time.sleep(1)
    res = []

    for page_nbr in tqdm.tqdm(range(0, 42), desc='Page'):
        driver.execute_script("return go_to_page({})".format(page_nbr))
        page_html = driver.execute_script("return document.documentElement.outerHTML")
        soup = BeautifulSoup(page_html)
        img_block = soup.find_all('div', attrs={'class': 'image_block'})

        for ib in tqdm.tqdm(img_block, desc='Image'):
            url_path_img_description = ib.contents[0]['href'].split('/')[-1]

            description_html = requests.get(URL + url_path_img_description)
            soup_description = BeautifulSoup(description_html.text)
            main_img = soup_description.find_all('li', attrs={"class": "pnri thumb_show active"})[0]

            id = 'ca_' + datetime.now().date().isoformat().replace('-','') +'_'+ str(uuid.uuid4()).split('-')[0]
            loc = main_img.find_all('div', attrs={"class": "img_desc", "id": "loc1"})[0].text
            lat = main_img.find_all('div', attrs={"class": "img_desc", "id": "lat1"})[0].text
            lng = main_img.find_all('div', attrs={"class": "img_desc", "id": "lng1"})[0].text
            date = main_img.find_all('div', attrs={"class": "img_desc", "id": "photodate1"})[0].text
            cam_dir = main_img.find_all('div', attrs={"class": "img_desc", "id": "dir1"})[0].text
            img_path = main_img.find_all('img', attrs={"id": "img1"})[0]['src'].replace('./', '')
            title = main_img.find_all('h1', attrs={"class": "img_title"})[0].text
            desc_html = main_img.find_all('p')
            desc = ' '.join([d.text for d in desc_html])

            script = soup_description.find_all('script')
            annotation = [eval(a.replace(';', '').replace(' ', '').split('=')[-1]) for a in script[-1].text.split('\n')
                          if
                          a.find("annotation_ary[2][") != -1]
            annotation = json.dumps(annotation)

            res.append([id, loc, lat, lng, date, cam_dir, img_path, title, desc, annotation])

            time.sleep(0.01)

    driver.quit()

    df = pd.DataFrame(res, columns=["id",
                                    "loc",
                                    "lat",
                                    "lng",
                                    "date",
                                    "cam_dir",
                                    "img_path",
                                    "title",
                                    "desc",
                                    "annotation"])

    if save:
        df.to_csv(out + 'cloud_atlas.csv', index=False)

    return df


def download_images(in_img_meta, out):
    df_img_meta = pd.read_csv(in_img_meta + 'cloud_atlas.csv')

    for _, img_meta in tqdm.tqdm(df_img_meta.iterrows(), desc='Download images'):
        id = img_meta['id']
        img_path = img_meta['img_path']

        wget.download(URL + img_path, out + "{}.png".format(id))


if __name__ == "__main__":
    out_metadata = "data/img_metadata/"
    out_images = "data/img/"

    # find_cloud_images(out_metadata)

    download_images(out_metadata, out_images)
