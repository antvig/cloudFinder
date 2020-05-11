from bs4 import BeautifulSoup
import requests
import wget
import uuid
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm

URL = 'https://www.clouds-online.com/cloud_atlas/'


def find_all_cloud_images(out, save=True):
    res = []
    list_pages = ["Cirrus", "Cirrostratus", "Cirrocumulus", "Altocumulus", "Altostratus",
                  "Stratocumulus", "Stratus", "Nimbostratus", "Cumulus", "Cumulonimbus"]
    for cloud_page in tqdm(list_pages):

        go_next = 1
        page_nbr = 1

        while go_next == 1:

            if page_nbr == 1:
                current_url = URL + "{}/{}.htm".format(cloud_page.lower(), cloud_page.lower())
            else:
                current_url = URL + "{}/{}_{}.htm".format(cloud_page.lower(), cloud_page.lower(), page_nbr)

            page = requests.get(current_url)
            data = page.text

            if data.find('oops') != -1:
                go_next = 0
                print('INVALID URL ({})'.format(current_url))
            else:
                print(current_url)
                page_nbr += 1
                soup = BeautifulSoup(data)
                img_of_page = soup.find_all('img')

                for i in img_of_page:
                    l = i["src"]
                    if l.find(cloud_page.lower()) != -1:
                        id = 'co_' + datetime.now().date().isoformat().replace('-', '') + '_' + \
                             str(uuid.uuid4()).split('-')[0]
                        l_large_image = l.replace('/th/', '/').replace('_th.', '.')
                        res.append([id, l_large_image, cloud_page])
                        time.sleep(0.01)

    df = pd.DataFrame(res, columns=["id",
                                    "img_path",
                                    "cloud_type"
                                    ])

    if save:
        df.to_csv(out + 'cloud_online.csv', index=False)

    return df


def download_images(in_img_meta, out):
    df_img_meta = pd.read_csv(in_img_meta + 'cloud_online.csv')

    for _, img_meta in tqdm(df_img_meta.iterrows(), desc='Download images'):
        id = img_meta['id']
        img_path = img_meta['img_path']

        wget.download(img_path, out + "{}.jpg".format(id))


if __name__ == '__main__':
    out_metadata = 'data/img_metadata/'
    out_img = 'data/img/'

    all_img = find_all_cloud_images(out_metadata)

    download_images(out_metadata, out_img)
