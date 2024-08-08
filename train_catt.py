import argparse

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from catt.data.tashkeel_dataset import PrePaddingDataLoader, TashkeelDataset
from catt.data.tashkeel_tokenizer import TashkeelTokenizer
from catt.models.encoder_decoder import EncoderDecoderTashkeelModel
from catt.models.encoder_only import EncoderOnlyTashkeelModel
from catt.models.model_types import ModelType


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    model_type = ModelType.from_string(config["model_type"])

    # Model's Configs
    dl_num_workers = config["dl_num_workers"]
    batch_size = config["batch_size"]
    max_seq_len = config["max_seq_len"]
    threshold = config["threshold"]
    device = config["device"]
    # Pretrained Char-Based BERT
    pretrained_mlm_pt = config[
        "pretrained_mlm_pt"
    ]  # Use None if you want to initialize weights randomly OR the path to the char-based BERT

    train_txt_folder_path = "dataset/train"
    val_txt_folder_path = "dataset/val"
    test_txt_folder_path = "dataset/test"

    tokenizer = TashkeelTokenizer()

    print("Creating Train Dataset...")
    train_dataset = TashkeelDataset(
        train_txt_folder_path,
        tokenizer,
        max_seq_len,
        tashkeel_to_text_ratio_threshold=threshold,
    )
    print("Creating Train Dataloader...")
    train_dataloader = PrePaddingDataLoader(
        tokenizer,
        train_dataset,
        batch_size=batch_size,
        num_workers=dl_num_workers,
        shuffle=True,
    )

    print("Creating Validation Dataset...")
    val_dataset = TashkeelDataset(
        val_txt_folder_path,
        tokenizer,
        max_seq_len,
        tashkeel_to_text_ratio_threshold=threshold,
    )
    print("Creating Validation Dataloader...")
    val_dataloader = PrePaddingDataLoader(
        tokenizer,
        val_dataset,
        batch_size=batch_size,
        num_workers=dl_num_workers,
        shuffle=False,
    )

    print("Creating Test Dataset...")
    test_dataset = TashkeelDataset(
        test_txt_folder_path,
        tokenizer,
        max_seq_len,
        tashkeel_to_text_ratio_threshold=threshold,
    )
    print("Creating Test Dataloader...")
    test_dataloader = PrePaddingDataLoader(
        tokenizer,
        test_dataset,
        batch_size=batch_size,
        num_workers=dl_num_workers,
        shuffle=False,
    )

    print("Creating Model...")
    if model_type == ModelType.ENCODER_ONLY:
        model_class = EncoderOnlyTashkeelModel
    elif model_type == ModelType.ENCODER_DECODER:
        model_class = EncoderDecoderTashkeelModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_class(
        tokenizer,
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        drop_prob=config["drop_prob"],
        learnable_pos_emb=config["learnable_pos_emb"],
    )

    print(f"Model type: {model_type}, Number of layers: {config['n_layers']}")

    # Use the pretrained weights of the char-based BERT model to initialize the model
    if pretrained_mlm_pt is not None:
        missing = model.transformer.load_state_dict(
            torch.load(pretrained_mlm_pt), strict=False
        )
        print(f"Missing layers: {missing}")

    # This is to freeze the encoder weights
    # freeze(model.transformer.encoder)

    dirpath = f"models/training/catt_{model_type}_model_v1/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        save_top_k=10,
        save_last=True,
        monitor="val_der",
        filename=f"catt_{model_type}_model"
        + "-{epoch:02d}-{val_loss:.5f}-{val_der:.5f}",
    )

    print("Creating Trainer...")

    logs_path = f"{dirpath}/logs"

    print("#" * 100)
    print(model)
    print("#" * 100)

    trainer = Trainer(
        accelerator=device,
        devices=-1,
        max_epochs=config["max_epochs"],
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
        logger=CSVLogger(save_dir=logs_path),
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tashkeel model with specified configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config_path)
