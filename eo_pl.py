
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from eo import Transformer
from tqdm import tqdm
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
# sequences is a list of tensors of shape TxH where T is the seqlen and H is the feats dim
def pad_seq(sequences, batch_first=True, padding_value=0.0, prepadding=True):
    lens = [i.shape[0]for i in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value) # NxTxH
    if prepadding:
        for i in range(len(lens)):
            padded_sequences[i] = padded_sequences[i].roll(-lens[i])
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1) # TxNxH
    return padded_sequences



def get_batches(X, batch_size=16):
    num_batches = math.ceil(len(X) / batch_size)
    for i in range(num_batches):
        x = X[i*batch_size : (i+1)*batch_size]
        yield x


class TashkeelModel(pl.LightningModule):
    def __init__(self, tokenizer, max_seq_len, d_model=512, n_layers=3, n_heads=16, drop_prob=0.1, learnable_pos_emb=True):

        super(TashkeelModel, self).__init__()

        ffn_hidden = 4 * d_model
        src_pad_idx = tokenizer.letters_map['<PAD>']
        trg_pad_idx = tokenizer.tashkeel_map['<PAD>']
        enc_voc_size = len(tokenizer.letters_map) # 37 + 3
        dec_voc_size = len(tokenizer.tashkeel_map) # 15 + 3
        self.transformer = Transformer(src_pad_idx=src_pad_idx,
                            trg_pad_idx=trg_pad_idx,
                            d_model=d_model,
                            enc_voc_size=enc_voc_size,
                            dec_voc_size=dec_voc_size,
                            max_len=max_seq_len,
                            ffn_hidden=ffn_hidden,
                            n_head=n_heads,
                            n_layers=n_layers,
                            drop_prob=drop_prob,
                            learnable_pos_emb=learnable_pos_emb
                            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tashkeel_map['<PAD>'])
        self.tokenizer = tokenizer


    def forward(self, x):
        y_pred = self.transformer(x)
        return y_pred


    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, 1:-1]
        y_out = target_ids[:, 1:-1]
        y_pred = self(input_ids)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        self.log('train_loss', loss, prog_bar=True)
#        sch = self.lr_schedulers()
#        sch.step()
#        self.log('lr', sch.get_last_lr()[0], prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, 1:-1]
        y_out = target_ids[:, 1:-1]
        y_pred = self(input_ids)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        pred_text_with_tashkeels = self.tokenizer.decode(input_ids, y_pred.argmax(2).squeeze())
        true_text_with_tashkeels = self.tokenizer.decode(input_ids, y_out)
        total_val_der_distance = 0
        total_val_der_ref_length = 0
        for i in range(len(true_text_with_tashkeels)):
            pred_text_with_tashkeel = pred_text_with_tashkeels[i]
            true_text_with_tashkeel = true_text_with_tashkeels[i]
            val_der = self.tokenizer.compute_der(true_text_with_tashkeel, pred_text_with_tashkeel)
            total_val_der_distance += val_der['distance']
            total_val_der_ref_length += val_der['ref_length']

        total_der_error = total_val_der_distance / total_val_der_ref_length
        self.log('val_loss', loss)
        self.log('val_der', torch.FloatTensor([total_der_error]))
        self.log('val_der_distance', torch.FloatTensor([total_val_der_distance]))
        self.log('val_der_ref_length', torch.FloatTensor([total_val_der_ref_length]))


    def test_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        y_pred = self(input_ids, None)
        loss = self.criterion(y_pred.transpose(1, 2), target_ids)
        self.log('test_loss', loss)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
        #max_iters = 10000
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=3e-6)
        gamma = 1 / 1.000001
        #gamma = 1 / 1.0001
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        opts = {"optimizer": optimizer} #,  "lr_scheduler": lr_scheduler}
        return opts


    @torch.no_grad()
    def do_tashkeel_batch(self, texts, batch_size=16, verbose=True):
        self.eval()
        device = next(self.parameters()).device
        text_with_tashkeel = []
        data_iter = get_batches(texts, batch_size)
        if verbose:
            num_batches = math.ceil(len(texts) / batch_size)
            data_iter = tqdm(data_iter, total=num_batches)
        for texts_mini in data_iter:
            input_ids_list = []
            for text in texts_mini:
                input_ids, _ = self.tokenizer.encode(text, test_match=False)
                input_ids_list.append(input_ids)
            batch_input_ids = pad_seq(input_ids_list, batch_first=True, padding_value=self.tokenizer.letters_map['<PAD>'], prepadding=False)
            batch_input_ids = batch_input_ids[:, 1:-1].to(device)
            y_pred = self(batch_input_ids)
            y_pred = y_pred.argmax(-1)
            # IMPORTANT NOTE: the following code snippet is to FORCE the prediction of the input space char to output no_tashkeel tag '<NT>'
            y_pred[self.tokenizer.letters_map[' '] == batch_input_ids] = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
            text_with_tashkeel_mini = self.tokenizer.decode(batch_input_ids, y_pred)
            text_with_tashkeel += text_with_tashkeel_mini

        return text_with_tashkeel


    @torch.no_grad()
    def do_tashkeel(self, text):
        return self.do_tashkeel_batch([text])[0]
