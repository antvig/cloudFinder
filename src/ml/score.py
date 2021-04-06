from sklearn.metrics import jaccard_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import pandas as pd

score_fct = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "iou": jaccard_score,
}


def compute_segmentation_score(
    y, y_pred, metrics=["accuracy", "recall", "precision", "iou"], group=False
):
    """
    Compute metrics for segmentation tasks.
    y has shape (nbr_img, size_height, size_width)

    Parameters
    ----------
    y
    y_pred
    metrics
    group

    Returns
    -------

    """

    out = []
    if group is False:
        for idx in tqdm(range(len(y))):
            tmp_y = y[idx]
            tmp_y_pred = y_pred[idx]
            score = [idx]
            for metric in metrics:
                score.append(score_fct[metric](tmp_y.ravel(), tmp_y_pred.ravel()))
            out.append(score)

        return pd.DataFrame(out, columns=["idx"] + metrics)
    else:
        score = []
        for metric in metrics:
            score.append(score_fct[metric](y.ravel(), y_pred.ravel()))

        out.append(score)

        return pd.DataFrame(out, columns=metrics)
