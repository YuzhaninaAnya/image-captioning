import os
import time
import math
import json
import torch
import torchvision.transforms as T
import gensim.downloader as api
from torch import nn
from torch.utils.data import DataLoader
from .models import *
from .config import get_config
from .logger import get_logger, close_logger
from .dataset import FlickrDataset
from .utils import split_data, collate_fn, init_random_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.path.dirname(os.path.realpath(__file__))
path_to_config = os.path.join(path, 'config.yaml')


def main():
    """
    Training and validation
    """
    config = get_config(path_to_config)
    logger = get_logger(config["data"]["path_to_log_file"])
    config["data"]["path_to_checkpoint"] = os.path.join(config["data"]["path_to_output_folder"],
                                                        config["data"]["model_file_name"])
    config["data"]["path_to_best_model"] = os.path.join(config["data"]["path_to_output_folder"],
                                                        "best-val-model.pt")
    checkpoint = os.path.isfile(config["data"]["path_to_checkpoint"])

    init_random_seed()

    # defining the transform to be applied
    transforms = T.Compose([
        T.Resize((288, 288)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if not checkpoint:
        logger.info("Loading data")
    data = FlickrDataset(
        config["data"]["path_to_images"],
        config["data"]["path_to_caption_file"],
        transforms
    )

    # Save word map to a JSON
    with open(os.path.join(config["data"]["path_to_output_folder"], "word2idx.json"), "w") as file:
        json.dump(data.vocab.word2idx, file)

    train_dataset, test_dataset = split_data(data)

    if not checkpoint:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        shuffle=False,
        collate_fn=collate_fn
    )

    if not checkpoint:
        logger.info(f"""Download embeddings: {config["gensim_model_name"]}""")
    glove = api.load(config["gensim_model_name"])

    embed_dim = config["model"]["embedding_dimension"]
    decoder_dim = config["model"]["decoder_hidden_dimension"]
    word2idx = data.vocab.word2idx

    encoder = Encoder()
    encoder.fine_tune(fine_tune=config["model"]["fine_tune_encoder"])
    decoder = DecoderWithAttention(embed_dim, word2idx, decoder_dim)
    # decoder = DecoderWithSimpleAttention(embed_dim, word2idx, decoder_dim)
    decoder.load_pretrained_embeddings(glove, True)
    model = Seq2Seq(encoder, decoder).to(device)

    if not checkpoint:
        logger.info(f"Embedding dimension: {embed_dim}")
        logger.info(f"Decoder hidden dimension: {decoder_dim}")
        logger.info(f"""Batch size: {config["batch_size"]}""")

    pad_idx = data.vocab.word2idx["<pad>"]
    learning_rate = config["model"]["learning_rate"]
    # momentum = config["model"]["momentum"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if not checkpoint:
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Start train")

    if os.path.isfile(config["data"]["path_to_checkpoint"]):
        checkpoint = torch.load(config["data"]["path_to_checkpoint"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_history = checkpoint["train_history"]
        valid_history = checkpoint["valid_history"]
        best_valid_loss = checkpoint["best_valid_loss"]
    else:
        start_epoch = -1
        train_history = []
        valid_history = []
        best_valid_loss = float('inf')

    for epoch in range(start_epoch+1, config["model"]["n_epochs"]):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, config["model"]["clip"])
        valid_loss = evaluate(model, test_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config["data"]["path_to_best_model"])

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history": train_history,
            "valid_history": valid_history,
            "best_valid_loss": best_valid_loss
        }, config["data"]["path_to_checkpoint"])

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        logger.info(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        logger.info(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        logger.info(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")

    logger.info("End")
    close_logger(logger)

    return 0


def train(model, iterator, optimizer, criterion, clip):
    """
    Performs one epoch's training

    :param model: model to train
    :param iterator: dataLoader for training data
    :param optimizer: optimizer to update weights
    # :param scheduler: learning rate scheduling
    :param criterion: loss layer
    :param clip: max norm of the gradients
    :return: average loss per epoch for training data
    """
    vocab_size = model.decoder.vocab_size
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):

        imgs = batch[0].to(device)
        caps = batch[1].to(device)  # (batch_size, sequence_length)

        optimizer.zero_grad()

        outputs = model(imgs, caps)  # (sequence_length, batch_size, vocab_size)
        outputs = outputs.permute((1, 0, 2))  # (batch_size, sequence_length, vocab_size)

        # Calculate the batch loss
        targets = caps[:, 1:]  # exclude token <start>
        loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), targets.reshape(-1))

        loss.backward()

        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    # scheduler.step()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Performs one epoch's validation

    :param model: model to evaluate
    :param iterator: dataLoader for validation data
    :param criterion: loss layer
    :return: average loss per epoch for validation data
    """
    vocab_size = model.decoder.vocab_size
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            imgs = batch[0].to(device)
            caps = batch[1].to(device)

            # turn off teacher forcing
            outputs = model(imgs, caps, teacher_forcing_ratio=0)  # (sequence_length, batch_size, vocab_size)
            outputs = outputs.permute((1, 0, 2))  # (batch_size, sequence_length, vocab_size)

            # Calculate the batch loss
            targets = caps[:, 1:]  # exclude token <start>
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), targets.reshape(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    """
    Measure time between events

    :param start_time: start time of event
    :param end_time: end time of event
    :return: total minutes and seconds between the two datetime objects
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    main()
