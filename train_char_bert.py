
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from char_bert_tokenizer import CharBERTTokenizer
from char_bert_dataset import CharBERTDataset, PrePaddingDataLoader
from char_bert_pl import CharBERTpl
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.nn import functional as F


# IMPORTANT NOTE: keep the following list in the same order since we will used IN SHA ALLAH in Tashkeel model with the same order
letters = [' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى', 'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر', 'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ']

max_seq_len = 1024
tokenizer = CharBERTTokenizer(letters, masking_prob=0.25, max_seq_len=max_seq_len)

train_text_file = 'train_text.txt'
val_text_file = 'val_text.txt'

train_dataset = CharBERTDataset(train_text_file, tokenizer)
train_dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, train_dataset, batch_size=32)

val_dataset = CharBERTDataset(val_text_file, tokenizer)
val_dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, val_dataset, batch_size=32)

model = CharBERTpl(tokenizer, max_seq_len=max_seq_len, d_model=512, n_layers=6, n_heads=16, drop_prob=0.1)

dirpath = 'char_bert_model_v1/'
checkpoint_callback = ModelCheckpoint(dirpath=dirpath, save_top_k=3, save_last=True,
                                      monitor='val_loss',
                                      filename='char_bert_model-{epoch:02d}-{val_loss:.5f}')


print('Creating Trainer...')

logs_path = f'{dirpath}/logs'

print('#'*100)
print(model)
print('#'*100)

trainer = Trainer(
    accelerator="cuda",
    devices=-1,
    max_epochs=300,
    callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
    logger=CSVLogger(save_dir=logs_path),
    strategy="ddp_find_unused_parameters_false"
    )

trainer.fit(model, train_dataloader, val_dataloader)


