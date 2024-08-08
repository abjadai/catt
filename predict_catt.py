import argparse

import torch
import yaml

from catt.data.tashkeel_tokenizer import TashkeelTokenizer
from catt.models.encoder_decoder import EncoderDecoderTashkeelModel
from catt.models.encoder_only import EncoderOnlyTashkeelModel
from catt.models.model_types import ModelType
from catt.utils import remove_non_arabic


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model(config, tokenizer: TashkeelTokenizer, device: str):
    model_type = ModelType.from_string(config["model_type"])

    if model_type == ModelType.ENCODER_ONLY:
        model_class = EncoderOnlyTashkeelModel
    elif model_type == ModelType.ENCODER_DECODER:
        model_class = EncoderDecoderTashkeelModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    ckpt_path = config["model_path"]
    model = model_class(
        tokenizer,
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        drop_prob=config["drop_prob"],
        learnable_pos_emb=config["learnable_pos_emb"],
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.eval().to(device)


def main(config_path: str):
    config = load_config(config_path)
    tokenizer = TashkeelTokenizer()
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = load_model(config, tokenizer, device)

    x = [
        "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدةوقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة,وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة"
    ]

    x = [remove_non_arabic(i) for i in x]
    batch_size = config["batch_size"]
    verbose = True
    x_tashkeel = model.do_tashkeel_batch(x, batch_size, verbose)

    print("Input text:")
    print(x)
    print("-" * 85)
    print("Diacritized text:")
    print(x_tashkeel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Tashkeel inference with specified configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config_path)
