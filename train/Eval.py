import torch
import numpy as np
from GPU_usage import *
from metrics.BLEU_score import *

def evaluate(tokenizer, model, val_set_loader):
    model.eval()

    # Initialize
    total_loss = 0
    all_ep = 0
    all_em = 0
    all_bleu_score = []

    with torch.no_grad():
        for i, data in enumerate(val_set_loader):
            inputs_ids = data['source_ids'].to(device)
            input_masks = data['source_mask'].to(device)
            target_ids = data['target_ids'].to(device)

            # calculate loss
            outputs = model(
                input_ids=inputs_ids,
                attention_mask=input_masks,
                labels=target_ids,
            )
            total_loss += outputs.loss.item()

            # generate comments
            generate_ids = model.generate(
                input_ids=inputs_ids,
                attention_mask=input_masks,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )


            # decode generated and actual comments
            generated_comments = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            actual_comments = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            # calculating BLEU score
            bleu_score = calculate_bleu(generated_comments, actual_comments)

            all_bleu_score.append(bleu_score)

    # Calculate avg metrics
    avg_loss = total_loss / len(val_set_loader)
    avg_bleu_score = np.mean(all_bleu_score) if all_bleu_score else 0

    return avg_loss, avg_bleu_score