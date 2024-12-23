from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(outputs, actuals, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculating bleu_socre
    :param outputs: the output of the model
    :param targets: the target that we want to output
    :param weights: weights for n-grams

    :return: BLEU score
    """
    outputs_list = [outputs]  # Convert to list
    smoothing = SmoothingFunction().method1  # Avoid zeros score for missing n-grams

    bleu_score = sentence_bleu(outputs_list, actuals, weights=weights, smoothing_function=smoothing)

    return bleu_score