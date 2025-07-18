# CATT: Character-based Arabic Tashkeel Transformer
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/MohamedRashad/arabic-auto-tashkeel)

This is the official implementation of the paper [CATT: Character-based Arabic Tashkeel Transformer](https://arxiv.org/abs/2407.03236).

## How to Run?
You need first to download models. You can find them in the `Releases` section of this repo.\
The best checkpoint for Encoder-Decoder (ED) model is `best_ed_mlm_ns_epoch_178.pt`.\
For the Encoder-Only (EO) model, the best checkpoint is `best_eo_mlm_ns_epoch_193.pt`.\
use the following bash script to download models:
```bash
mkdir models/
wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt
wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_eo_mlm_ns_epoch_193.pt
```
You can use the inference code examples: `predict_ed.py` for ED models and `predict_eo.py` for EO models.\
Both examples are provided with batch inference support. Read the source code to gain a better understanding.
```bash
python predict_ed.py
python predict_eo.py
```
EO models are recommended for faster inference.\
ED models are recommended for better accuracy of the predicted diacritics.

## Converting Models to ONNX

### Export PyTorch Models

To convert your trained PyTorch models to ONNX format, use the export script:

```bash
python export_to_onnx.py
```

This script will:
- Load your trained PyTorch model checkpoints
- Export separate ONNX models for encoder and decoder components
- Validate the exported models for correctness
- Save the ONNX models in the `onnx_models/` directory

**Output files:**
- `encoder.onnx` - The encoder component
- `decoder.onnx` - The decoder component (or linear layer for encoder-only models)

### Running ONNX Models

To test and run inference with the exported ONNX models:

```bash
python run_onnx.py
```

This script will:
- Load the exported ONNX models
- Run inference using ONNX Runtime

For more details on the export process, check the `export_to_onnx.py` script configuration.


## How to Train?
To start trainnig, you need to download the dataset from the `Releases` section of this repo.
```bash
wget https://github.com/abjadai/catt/releases/download/v2/dataset.zip
unzip dataset.zip
```
Then, edit the script `train_catt.py` and adjest the default values:
```python
# Model's Configs
model_type = 'ed' # 'eo' for Encoder-Only OR 'ed' for Encoder-Decoder
dl_num_workers = 32
batch_size = 32
max_seq_len = 1024
threshold = 0.6

# Pretrained Char-Based BERT
pretrained_mlm_pt = None # Use None if you want to initialize weights randomly OR the path to the char-based BERT
#pretrained_mlm_pt = 'char_bert_model_pretrained.pt'
```
Finally, run the training script.
```bash
python train_catt.py
```

## Resources
- This code is mainly adapted from [this repo](https://github.com/hyunwoongko/transformer).
- An [older version](https://github.com/MTG/ArabicTransliterator/blob/master/qalsadi/libqutrub/arabic_const.py) of some Arabic scripts that are available in [pyarabic](https://github.com/linuxscout/pyarabic/blob/master/pyarabic/araby_const.py) library were used as well.

### ToDo
- [x] inference script
- [x] upload our pretrained models
- [x] upload CATT dataset
- [x] upload DER scripts
- [x] training script

## License Change: CC-BY-NC to Apache 2.0
This repository has updated its license from Creative Commons Attribution-NonCommercial (CC-BY-NC) to Apache 2.0 License.
This change removes commercial use restrictions and adopts an industry-standard open-source license, enabling broader adoption and collaboration.
This transition supports the project's growth while maintaining our commitment to open-source principles.
