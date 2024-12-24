def exact_match(preds, goldens):
    correct_ = 0
    for i in range(len(preds)):
        prediction = preds[i].split('\t')[0].strip()  ## only the top 1 recommendation
        if prediction.strip() == goldens[i].strip():
            correct_ += 1
    print('Exact Match: ', correct_ / len(preds))