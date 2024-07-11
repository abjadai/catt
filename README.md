# CATT: Character-based Arabic Tashkeel Transformer
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] <a href='https://arxiv.org/abs/2407.03236'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

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

## Resources
- This code is mainly adapted from [this repo](https://github.com/hyunwoongko/transformer).
- An [older version](https://github.com/MTG/ArabicTransliterator/blob/master/qalsadi/libqutrub/arabic_const.py) of some Arabic scripts that are available in [pyarabic](https://github.com/linuxscout/pyarabic/blob/master/pyarabic/araby_const.py) library were used as well.

### ToDo
- [x] inference script
- [x] upload our pretrained models
- [x] upload CATT dataset
- [x] upload DER scripts
- [x] training script


This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

