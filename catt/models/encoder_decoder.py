import torch.nn as nn

from catt.models.transformer import Decoder, Encoder, math, pl, torch
from catt.utils import get_batches, pad_seq, tqdm


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        learnable_pos_emb=True,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            padding_idx=src_pad_idx,
            learnable_pos_emb=learnable_pos_emb,
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            padding_idx=trg_pad_idx,
            learnable_pos_emb=learnable_pos_emb,
        )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, src, trg):
        device = self.get_device()
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx).to(
            device
        )
        src_trg_mask = self.make_pad_mask(
            trg, src, self.trg_pad_idx, self.src_pad_idx
        ).to(device)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx).to(
            device
        ) * self.make_no_peak_mask(trg, trg).to(device)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)
        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)
        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)
        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)
        return mask


class EncoderDecoderTashkeelModel(pl.LightningModule):

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        d_model=512,
        n_layers=3,
        n_heads=16,
        drop_prob=0.1,
        learnable_pos_emb=True,
    ):
        super(EncoderDecoderTashkeelModel, self).__init__()

        ffn_hidden = 4 * d_model
        src_pad_idx = tokenizer.letters_map["<PAD>"]
        trg_pad_idx = tokenizer.tashkeel_map["<PAD>"]
        enc_voc_size = len(tokenizer.letters_map)  # 37 + 3
        dec_voc_size = len(tokenizer.tashkeel_map)  # 15 + 3
        self.transformer = EncoderDecoderTransformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            d_model=d_model,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_seq_len,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layers=n_layers,
            drop_prob=drop_prob,
            learnable_pos_emb=learnable_pos_emb,
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tashkeel_map["<PAD>"]
        )
        self.tokenizer = tokenizer

    def forward(self, x, y=None):
        y_pred = self.transformer(x, y)
        return y_pred

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, :-1]
        y_in = target_ids[:, :-1]
        y_out = target_ids[:, 1:]
        y_pred = self(input_ids, y_in)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        self.log("train_loss", loss, prog_bar=True)
        sch = self.lr_schedulers()
        sch.step()
        self.log("lr", sch.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        input_ids = input_ids[:, :-1]
        y_in = target_ids[:, :-1]
        y_out = target_ids[:, 1:]
        y_pred = self(input_ids, y_in)
        loss = self.criterion(y_pred.transpose(1, 2), y_out)

        pred_text_with_tashkeels = self.tokenizer.decode(
            input_ids, y_pred.argmax(2).squeeze()
        )
        true_text_with_tashkeels = self.tokenizer.decode(input_ids, y_out)
        total_val_der_distance = 0
        total_val_der_ref_length = 0
        for i in range(len(true_text_with_tashkeels)):
            pred_text_with_tashkeel = pred_text_with_tashkeels[i]
            true_text_with_tashkeel = true_text_with_tashkeels[i]
            val_der = self.tokenizer.compute_der(
                true_text_with_tashkeel, pred_text_with_tashkeel
            )
            total_val_der_distance += val_der["distance"]
            total_val_der_ref_length += val_der["ref_length"]

        total_der_error = total_val_der_distance / total_val_der_ref_length
        self.log("val_loss", loss)
        self.log("val_der", torch.FloatTensor([total_der_error]))
        self.log("val_der_distance", torch.FloatTensor([total_val_der_distance]))
        self.log("val_der_ref_length", torch.FloatTensor([total_val_der_ref_length]))

    def test_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        y_pred = self(input_ids, None)
        loss = self.criterion(y_pred.transpose(1, 2), target_ids)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        # max_iters = 10000
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=3e-6)
        gamma = 1 / 1.000001
        # gamma = 1 / 1.0001
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        opts = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
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
            batch_input_ids = pad_seq(
                input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.letters_map["<PAD>"],
                prepadding=False,
            )
            target_ids = torch.LongTensor(
                [[self.tokenizer.tashkeel_map["<BOS>"]]] * len(texts_mini)
            ).to(device)
            src = batch_input_ids.to(device)

            src_mask = self.transformer.make_pad_mask(
                src, src, self.transformer.src_pad_idx, self.transformer.src_pad_idx
            ).to(device)
            enc_src = self.transformer.encoder(src, src_mask)

            for i in range(src.shape[1] - 1):
                trg = target_ids
                src_trg_mask = self.transformer.make_pad_mask(
                    trg, src, self.transformer.trg_pad_idx, self.transformer.src_pad_idx
                ).to(device)
                trg_mask = self.transformer.make_pad_mask(
                    trg, trg, self.transformer.trg_pad_idx, self.transformer.trg_pad_idx
                ).to(device) * self.transformer.make_no_peak_mask(trg, trg).to(device)

                preds = self.transformer.decoder(trg, enc_src, trg_mask, src_trg_mask)
                # IMPORTANT NOTE: the following code snippet is to FORCE the prediction of the input space char to output no_tashkeel tag '<NT>'
                target_ids = torch.cat(
                    [target_ids, preds[:, -1].argmax(1).unsqueeze(1)], axis=1
                )
                target_ids[
                    self.tokenizer.letters_map[" "] == src[:, : target_ids.shape[1]]
                ] = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
            #                target_ids = torch.cat([target_ids, preds[:, -1].argmax(1).unsqueeze(1)], axis=1)
            text_with_tashkeel_mini = self.tokenizer.decode(src, target_ids)
            text_with_tashkeel += text_with_tashkeel_mini
        return text_with_tashkeel

    @torch.no_grad()
    def do_tashkeel(self, text):
        return self.do_tashkeel_batch([text])[0]
