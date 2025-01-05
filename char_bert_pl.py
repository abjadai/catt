from transformer import Encoder, make_pad_mask
import torch.nn as nn
import torch
import random
import pytorch_lightning as pl
from torch.nn import functional as F
import xer

class CharBERTpl(pl.LightningModule):

    def __init__(self, tokenizer, max_seq_len=2048, d_model=512, n_layers=6, n_heads=16, drop_prob=0.1, learnable_pos_emb=True):
        super(CharBERTpl, self).__init__()
        self.tokenizer = tokenizer
        ffn_hidden = 4 * d_model
        enc_voc_size = len(self.tokenizer.letters) # 36 Arabic letter + 1 space char + 4 special chars
        self.pad_token_id = tokenizer.pad_token_id

        self.encoder = Encoder(d_model=d_model,
                               n_head=n_heads,
                               max_len=max_seq_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=self.pad_token_id,
                               learnable_pos_emb=learnable_pos_emb)

        self.linear = nn.Linear(d_model, enc_voc_size, bias=False)
        self.linear.weight.data = self.encoder.emb.tok_emb.weight.data # weight tying
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)


    def forward(self, x):
        x_mask = make_pad_mask(x, self.pad_token_id)
        y_pred = self.encoder(x, x_mask)
        y_pred = self.linear(y_pred)
        return y_pred


    def training_step(self, batch, batch_idx):
        orig_token_ids, input_ids, target_ids = batch
        outputs = self(input_ids)
        loss = self.criterion(outputs.permute(0,2,1), target_ids)
        self.log('train_loss', loss, prog_bar=True)
        sch = self.lr_schedulers()
        if not sch is None:
            sch.step()
            self.log('lr', sch.get_last_lr()[0], prog_bar=True)
        return loss


    def replace_with_mask_randomly(self, text, n_times, mask_token, is_completely_random=False):
        if not is_completely_random:
        # this will make the examples have determinstic random mask, i.e., it will generate the same masks for the same example for each time
            seed = 0
        random.seed(seed)
        all_idxs = list(range(len(text)))
        rand_idxs = []
        while len(rand_idxs) < n_times:
            rand_idx = random.choice(all_idxs)
            if not rand_idx in rand_idxs:
                rand_idxs.append(rand_idx)
        char_list = list(text)
        for i in rand_idxs:
            char_list[i] = mask_token
        masked_text = ''.join(char_list)
        return masked_text


    def validation_step(self, batch, batch_idx):
        orig_token_ids, input_ids, target_ids = batch
        outputs = self(input_ids)
        loss = self.criterion(outputs.permute(0,2,1), target_ids)

        mask_token = '*'
        total_val_cer_distance = 0
        total_val_cer_ref_length = 0
        total_val_wer_distance = 0
        total_val_wer_ref_length = 0
        for i in range(len(orig_token_ids)):
            orig_text = self.tokenizer.detokenize(orig_token_ids[i])
            masked_text = self.replace_with_mask_randomly(orig_text, int(0.15*len(orig_text)), mask_token, is_completely_random=False)
            predicted_text = self.predict_mask(masked_text, mask_token)
            ref, hyp = orig_text, predicted_text
            sample_wer = xer.wer(ref, hyp)
            sample_cer = xer.cer(ref, hyp)
            total_val_wer_distance += sample_wer['distance']
            total_val_wer_ref_length += sample_wer['ref_length']
            total_val_cer_distance += sample_cer['distance']
            total_val_cer_ref_length += sample_cer['ref_length']

        total_wer_error = total_val_wer_distance / total_val_wer_ref_length
        total_cer_error = total_val_cer_distance / total_val_cer_ref_length
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_wer', torch.FloatTensor([total_wer_error]).to(self.device), sync_dist=True)
        self.log('val_wer_distance', torch.FloatTensor([total_val_wer_distance]).to(self.device), sync_dist=True)
        self.log('val_wer_ref_length', torch.FloatTensor([total_val_wer_ref_length]).to(self.device), sync_dist=True)
        self.log('val_cer', torch.FloatTensor([total_cer_error]).to(self.device), sync_dist=True)
        self.log('val_cer_distance', torch.FloatTensor([total_val_cer_distance]).to(self.device), sync_dist=True)
        self.log('val_cer_ref_length', torch.FloatTensor([total_val_cer_ref_length]).to(self.device), sync_dist=True)

        return loss


    @torch.no_grad()
    def predict_mask(self, text, mask_token=None):
        orig_mask_token = self.tokenizer.mask_token
        if not mask_token is None:
            self.tokenizer.set_mask_token(mask_token)
        token_ids = self.tokenizer.tokenize(text)
        mask_idxs = []
        for i, token_id in enumerate(token_ids):
            if token_id == self.tokenizer.mask_token_id:
                mask_idxs.append(i)
        mask_idxs = torch.tensor(mask_idxs).long().to(self.device)
        token_ids_tensor = torch.tensor(token_ids).long().unsqueeze(0).to(self.device)
        out = self(token_ids_tensor)
        out = out[:, mask_idxs, :].argmax(-1).squeeze(0) # select output of the mask ONLY

        filled_masks_token_ids = list(token_ids)
        for i, idx in enumerate(mask_idxs.tolist()):
            filled_masks_token_ids[idx] = int(out[i])
        filled_mask_text = self.tokenizer.detokenize(filled_masks_token_ids)
        self.tokenizer.set_mask_token(orig_mask_token)
        return filled_mask_text


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        gamma = 1 / 1.000001
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        opts = {"optimizer": optimizer,  "lr_scheduler": lr_scheduler}
        return opts


