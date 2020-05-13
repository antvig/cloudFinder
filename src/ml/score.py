from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
)

import pandas as pd


score_fct = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
}


def compute_classification_score(
    df, target, pred_label, score=["accuracy", "recall", "precision"], gp_by=None
):

    if type(pred_label) == str:
        pred_label = [pred_label]

    if type(score) == str:
        score = [score]

    if gp_by is None:
        df['gp'] = 0
        _tmp = df.groupby('gp')
    else:
        _tmp = df.groupby(gp_by)
    score = _tmp.apply(
        lambda x: pd.Series(
            {(v, s): score_fct[s](x[target], x[v]) for v in pred_label for s in score}
        )
    )
    
    if gp_by is None:
        del df['gp']
        score.reset_index(drop=True, inplace=True)
    
    return score