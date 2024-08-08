import torch
import yaml

from catt.data.tashkeel_tokenizer import TashkeelTokenizer
from catt.models.encoder_decoder import EncoderDecoderTashkeelModel
from catt.models.encoder_only import EncoderOnlyTashkeelModel
from catt.models.model_types import ModelType
from catt.utils import remove_non_arabic

# Load config
config_path = "configs/EncoderOnly_config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize tokenizer and device
tokenizer = TashkeelTokenizer()
device = config['device'] if torch.cuda.is_available() else "cpu"


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


# Load the model once
model = load_model(config, tokenizer, device)


def tashkeel_text(text: str) -> str:
    input_text = remove_non_arabic(text)
    diacritized_text = model.do_tashkeel_batch(
        [input_text], batch_size=1, verbose=False
    )[0]
    return diacritized_text
