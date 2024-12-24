import torch
import numpy as np
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from evaluate import *


def train_per_iter(model, train_set_loader, optimizer):
    # Initialize variables
    total_loss = 0
    final_loss = 0
    for i, data in enumerate(train_set_loader):
        # Extracting data
        in_ids = data['source_ids'].to(DEVICE)
        in_mask = data['source_mask'].to(DEVICE)
        target_ids = data['target_ids'].to(DEVICE)
        target_ids_new = target_ids[:-1, :]  # Shifting by one to avoid <technical_language>

        # Clear cache
        optimizer.zero_grad()

        # forward
        outputs = model(
            input_ids=in_ids,
            attention_mask=in_mask,
            decoder_input_ids=target_ids,
        )

        loss = outputs.loss

        # update loss
        total_loss += loss.item()

        # backward
        loss.backward()

        # Update parameters
        optimizer.step()

    final_loss = total_loss / len(train_set_loader)
    return final_loss


def T5_trainer(model_params, train_path, val_path):
    """
    T5 trainer
    :param model_params: list of parameters
    """
    # Initialize
    val_avg_loss = 0
    val_avg_bleu_score = 0

    # set random seeds
    torch.manual_seed(model_params['SEED'])
    np.random.seed(model_params['SEED'])

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_params['MODEL'])

    # model T5
    model = T5ForConditionalGeneration.from_pretrained(model_params['MODEL'])
    model = model.to(DEVICE)

    # creating
    training_set = CustomDataset(
        path=train_path,
        tokenizer=T5Tokenizer.from_pretrained(model_params['MODEL']),
        code_lines_len=['MAX_SOURCE_TEXT_LENGTH'],
        comments_len=model_params['MAX_TARGET_TEXT_LENGTH'],
    )
    val_set = CustomDataset(
        path=val_path,
        tokenizer=T5Tokenizer.from_pretrained(model_params['MODEL']),
        code_lines_len=['MAX_SOURCE_TEXT_LENGTH'],
        comments_len=model_params['MAX_TARGET_TEXT_LENGTH'],
    )

    # Dataloader
    training_loader = DataLoader(training_set, batch_size=model_params['TRAIN_BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=model_params['VAL_BATCH_SIZE'], shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['LEARNING_RATE'])

    # Training loop
    for epoch in range(model_params['TRAIN_EPOCHS']):
        # Gradient tracking is on
        model.train(True)

        # Train
        train_per_iter(model=model, train_set_loader=training_loader, optimizer=optimizer)

        model.train(False)

    # Save model
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, "model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Evaluating model
    for epoch in range(model_params['VAL_EPOCHS']):
        val_avg_loss, val_avg_bleu_score, val_avg_ep, val_avg_em = evaluate(tokenizer, model, val_set_loader=val_loader)

        if epoch % 1 == 0:
            print(f"Loss = {val_avg_loss:.4f}")
            print(f"BLEU Score = {val_avg_bleu_score:.4f}")