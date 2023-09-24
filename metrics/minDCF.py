import numpy as np

def compuet_fnr_and_fpr(predictions, y_array):
    fv = predictions - y_array
    false_positives = len(fv[fv == 1])
    false_negatives = len(fv[fv == -1])
    fv = predictions + y_array
    true_positives = len(fv[fv == 2])
    true_negatives = len(fv[fv == 0])
    false_negatives_rate = false_negatives / (true_positives + false_negatives)
    false_positives_rate = false_positives / (true_negatives + false_positives)
    return false_negatives_rate, false_positives_rate

def actDCF(y_array, score_array, pi, c_fn, c_fp, th=None):
    if th == None:
        th = -np.log((pi*c_fn)/((1-pi)*c_fp))
    predictions = np.where(score_array > th,1,0)
    false_negatives_rate, false_positives_rate = compuet_fnr_and_fpr(predictions, y_array)
    DCFu = pi * c_fn *  false_negatives_rate + (1 - pi) * c_fp * false_positives_rate
    act_dcf = DCFu / min(pi * c_fn, (1 - pi) * c_fp)
    return act_dcf


def minDCF(y_array, score_array, pi, c_fn, c_fp):
    DCF_list = []
    for th in score_array:
        act_dcf = actDCF(y_array, score_array, pi, c_fn, c_fp, th=th)
        DCF_list.append(act_dcf)
    return round(min(DCF_list),3)