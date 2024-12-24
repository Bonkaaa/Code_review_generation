import numpy as np


def normal_leven2(list1, list2: list):
    str1 = list1
    str2 = list2
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [0 for n in range(len_str1 * len_str2)]

    for i in range(len_str1):
        matrix[i] = i

    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]


def edit_progress(input, golden, predicted):
    golds = golden
    predictions = predicted
    sources = input
    edit_distance_pred2gold = []
    edit_distance_src2gold = []

    rr = []
    for ii in range(len(golds)):
        if golds[ii].strip() == predictions[ii].split('\t')[0].strip():
            rr.append(1)
        else:
            rr.append(0)

    for i in range(len(golds)):
        ## Token Level
        edit_distance_pred2gold.append(
            normal_leven2(golds[i].strip().split(), predictions[i].strip().split('\t')[0].strip().split())
        )

        edit_distance_src2gold.append(
            normal_leven2(golds[i].strip().split(), sources[i].strip().split('\t')[0].strip().split())
        )

    progress = []
    for i in range(len(edit_distance_pred2gold)):
        pred_ = edit_distance_pred2gold[i]
        src_ = edit_distance_src2gold[1]
        p_ = round((abs(src_) - abs(pred_)) / abs(src_), 3)
        progress.append(p_)

    print('Edit Progress: ', np.sum(np.array(progress)) / len(progress))
    return progress


