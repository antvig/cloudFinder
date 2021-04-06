import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
os.chdir(project_path)

import dash
import dash_core_components as dcc
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output, State
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dataset_meta = pd.read_csv("data/process/sky_segmentation/dataset_meta.csv")
sun_meta = pd.read_csv("data/img_metadata/sun.csv")
metadata = pd.merge(dataset_meta, sun_meta, left_on="image", right_on="img_name")
metadata = metadata[metadata['idx'] != -1].copy()
metadata = metadata[["image", "name", "sky_coverage"]].copy()

X = np.load("data/process/sky_segmentation/X.npy")
y = np.load("data/process/sky_segmentation/y.npy")

y_pred = np.load("data/process/sky_segmentation/y_pred.npy")[:, :, :, 0]

images = list(set(metadata.image))

app.layout = html.Div(
        id="app-container",
        children=[
                # Banner
                html.Div(
                        id="banner",
                        className="row",
                        children=[html.H1("Sky Segmentation")],
                        ),
                # Left column
                html.Div(
                        className="row",
                        children=[
                                html.Div(
                                        id="left-column",
                                        className="four columns",
                                        children=[
                                                html.B("Table"),
                                                dash_table.DataTable(
                                                        id='datatable-interactivity',
                                                        columns=[
                                                                {"name": i, "id": i}
                                                                for i in metadata.columns
                                                                ],
                                                        data=metadata.to_dict('records'),
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        page_action="native",
                                                        page_current=0,
                                                        page_size=10,
                                                        ),
                                                html.B("Image"),
                                                html.Br(),
                                                dcc.Dropdown(
                                                        id="image-select",
                                                        options=[{"label": i, "value": i} for i in images],
                                                        value=images[0],
                                                        ),
                                                html.Br(),
                                                html.B("Threshold"),
                                                html.Br(),
                                                dcc.Slider(
                                                        id='threshold-select',
                                                        min=0,
                                                        max=1,
                                                        step=0.05,
                                                        value=0.5,
                                                        ),
                                                html.Br(),
                                                ],
                                        ),
                                # Right column
                                html.Div(
                                        id="right-column",
                                        className="eight columns",
                                        children=[
                                                html.B("Image"),
                                                html.Img(src="", id="img_plot")
                                                ],
                                        ),
                                ],
                        ),
                ],
        )


@app.callback(
        Output("img_plot", "src"), [Input("image-select", "value"), Input("threshold-select", "value")]
        )
def update_image(image, threshold):
    image_idx = dataset_meta[dataset_meta.image == image].idx.iloc[0]

    sky = y[image_idx]
    sky_contours = measure.find_contours(sky, 0)

    sky_predicted = y_pred[image_idx] >= threshold
    sky_predicted_contours = measure.find_contours(sky_predicted, 0)

    img = X[image_idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for n, c in enumerate(sky_contours):
        ax.plot(c[:, 1], c[:, 0], linewidth=3, color="r")
    for n, c in enumerate(sky_predicted_contours):
        ax.plot(c[:, 1], c[:, 0], linewidth=3, color="y")
    figfile = BytesIO()
    plt.savefig(figfile, format="png")
    figdata_png = base64.b64encode(figfile.getvalue()).decode()

    src = "data:image/png;base64,{}".format(figdata_png)

    return src


if __name__ == "__main__":
    app.run_server(debug=True)
