# CATT: Character-based Arabic Tashkeel Transformer
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] <a href='https://arxiv.org/abs/2407.03236'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This is the official implementation of the paper [CATT: Character-based Arabic Tashkeel Transformer](https://arxiv.org/abs/2407.03236).

## Table of Contents
- [CATT: Character-based Arabic Tashkeel Transformer](#catt-character-based-arabic-tashkeel-transformer)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [How to Run?](#how-to-run)
    - [Using the API](#using-the-api)
    - [Using the Prediction Script](#using-the-prediction-script)
  - [How to Train?](#how-to-train)
  - [Resources](#resources)
  - [ToDo](#todo)
  - [License](#license)

## Project Structure
```
├── api/                  # API-related files
├── benchmarking/         # Benchmarking scripts and data
├── catt/                 # Core CATT package
│   ├── data/             # Data handling modules
│   ├── models/           # Model architectures
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── dataset/              # Dataset files
├── docs/                 # Documentation
├── models/               # Pre-trained model checkpoints
├── scripts/              # Utility scripts
├── tests/                # Test files
├── compute_der.py        # Diacritization Error Rate computation
├── predict_catt.py       # Prediction script
├── train_catt.py         # Training script
├── pyproject.toml        # Project dependencies and metadata
└── README.md             # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abjadai/catt.git
   cd catt
   ```

2. Install the required dependencies:
   ```bash
   pip install poetry
   poetry install
   ```
3. Download the pre-trained models:
   ```bash
   mkdir models/
   wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt
   wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_eo_mlm_ns_epoch_193.pt
   ```

## How to Run?
   
### Using the API

1. Start the FastAPI server:
   ```bash
   python -m api.main
   ```

2. Send a POST request to `http://localhost:8000/tashkeel` with a JSON body:
   ```json
   {
     "text": "العلم نور والجهل ظلام."
   }
   ```

### Using the Prediction Script

1. Run the prediction script:
   ```bash
   python predict_catt.py ./configs/EncoderDecoder_config.yaml
   # or
   python predict_catt.py ./configs/EncoderOnly_config.yaml
   ```

Note:
- **Encoder-Only (EO)**: model is recommended for faster inference.
- **Encoder-Decoder (ED)**: model is recommended for better accuracy of the predicted diacritics.

## How to Train?

1. Download the dataset:
   ```bash
   wget https://github.com/abjadai/catt/releases/download/v2/dataset.zip
   unzip dataset.zip
   ```

2. Edit the `configs/Sample_config.yaml` file to adjust the training parameters.
    ```yaml
    model_type: encoder-only # or encoder-decoder
    max_seq_len: 1024
    d_model: 512
    n_layers: 6
    n_heads: 16
    drop_prob: 0.1
    learnable_pos_emb: false
    batch_size: 32
    dl_num_workers: 32
    threshold: 0.6
    max_epochs: 300
    model_path: 
    pretrained_mlm_pt: # Use None if you want to initialize weights randomly OR the path to the char-based BERT
    device: 'cuda'
    ```

3. Run the training script:
   ```bash
   python train_catt.py ./configs/Sample_config.yaml
   ```

## Resources
- This code is mainly adapted from [this repo](https://github.com/hyunwoongko/transformer).
- An [older version](https://github.com/MTG/ArabicTransliterator/blob/master/qalsadi/libqutrub/arabic_const.py) of some Arabic scripts that are available in [pyarabic](https://github.com/linuxscout/pyarabic/blob/master/pyarabic/araby_const.py) library were used as well.

## ToDo
- [x] Inference script
- [x] Upload pre-trained models
- [x] Upload CATT dataset
- [x] Upload DER scripts
- [x] Training script


## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg