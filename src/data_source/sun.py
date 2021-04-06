from bs4 import BeautifulSoup
import requests
import warnings
import xml.etree.ElementTree as ET
import pandas as pd
import os

warnings.filterwarnings("ignore")
from tqdm import tqdm
import wget
import xml


# DISABLE Zscaler
#


def find_all_cloud_sky_images(out, save_intermediate=False):

    URL_annotation = (
        "http://labelme.csail.mit.edu/Annotations/users/antonio/static_sun_database/"
    )

    page = requests.get(URL_annotation)
    data = page.text

    soup = BeautifulSoup(data)

    all_letters = [
        a
        for a in soup.find_all("a")
        if a.has_attr("href")
        and (a.text != "Parent Directory")
        and (a.text != "tmp.tmp")
        and (a.text.find("/") != -1)
    ]

    all_res = []

    for letter in all_letters:
        res = []

        if "sun_{}.csv".format(letter.get("href").split("/")[0]) in os.listdir(out):
            continue

        print("-- letter {}".format(letter.get("href")))
        URL_letter = URL_annotation + letter.get("href")
        page = requests.get(URL_letter)
        data = page.text
        soup = BeautifulSoup(data)

        all_objects = [
            a
            for a in soup.find_all("a")
            if a.has_attr("href")
            and (a.text != "Parent Directory")
            and (a.text != "tmp.tmp")
            and (a.text.find("/") != -1)
        ]

        for object in all_objects:
            c = 0
            print("    -- object {}".format(object.get("href")))
            URL_object = URL_letter + object.get("href")
            page = requests.get(URL_object)
            data = page.text
            soup = BeautifulSoup(data)

            all_xml = [
                a
                for a in soup.find_all("a")
                if a.has_attr("href") and (a.text.find(".xml") != -1)
            ]

            for xml in all_xml:
                URL_xml = URL_object + xml.get("href")
                page = requests.get(URL_xml)
                data = page.text

                # if (data.find('cloud') == -1) and (data.find('sky') == -1) and (data.find('clouds') == -1):
                #     continue

                soup = BeautifulSoup(data)
                img_name = soup.find("filename").text
                img_name_path = URL_object.replace("Annotations", "Images")

                all_img_objects = soup.find_all("object")
                for img_object in all_img_objects:
                    name = img_object.find("name").text.lower()
                    if name in ["cloud", "sky", "clouds"]:
                        res.append([name, img_name, img_name_path, str(img_object)])
                        c += 1
            print("      --> {} sky or cloud founds !".format(c))

        df = pd.DataFrame(
            res, columns=["name", "img_name", "img_name_path", "polygone"]
        )
        all_res.append(df)

        if save_intermediate:
            df.to_csv(
                os.path.join(
                    out, "sun_{}.csv".format(letter.get("href").split("/")[0])
                ),
                index=False,
            )

    if not save_intermediate:
        all_res = pd.concat(all_res)

        return all_res

    else:
        all_res = aggregate_results(out)
        return all_res


def aggregate_results(out):

    res = []
    for file in os.listdir(out):
        if file[:4] == "sun_":
            res.append(pd.read_csv(os.path.join(out, file)))

    res = pd.concat(res)

    return res


def download_images(df_img_meta, out):

    df_img_meta = df_img_meta.drop_duplicates("img_name")

    for _, img_meta in tqdm(df_img_meta.iterrows(), desc="Download images"):

        img_path = img_meta["img_name_path"] + img_meta["img_name"]
        local_img_path = os.path.join(out, img_meta["img_name"])
        if not os.path.isfile(local_img_path):
            wget.download(img_path, local_img_path)


def parse_xml_polygone(xml_polygon):

    a = xml.etree.ElementTree.fromstring(xml_polygon)
    poly = a.find("polygon")

    if poly is None:
        polygone_coordinate = []
    else:
        try:
            pts = poly.findall("pt")
            polygone_coordinate = [
                [int(pt.find("y").text), int(pt.find("x").text)] for pt in pts
            ]
        except:
            polygone_coordinate = []

    return polygone_coordinate


if __name__ == "__main__":
    out_metadata = "data/img_metadata/"
    out_img = "data/img/"

    aggregate_results(out_metadata)
