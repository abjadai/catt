import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

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


class PrePaddingDataLoader(DataLoader):
    def __init__(self, pad_token_id, *args, **kwargs):
        super(PrePaddingDataLoader, self).__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        orig_token_ids_list = []
        input_ids_list = []
        target_ids_list = []
        for orig_token_ids, input_ids, target_ids in batch:
            orig_token_ids_list.append(orig_token_ids)
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
        batch_orig_token_ids = pad_seq(orig_token_ids_list, batch_first=True, padding_value=self.pad_token_id, prepadding=False)
        batch_input_ids = pad_seq(input_ids_list, batch_first=True, padding_value=self.pad_token_id, prepadding=False)
        batch_target_ids = pad_seq(target_ids_list, batch_first=True, padding_value=self.pad_token_id, prepadding=False)
        return batch_orig_token_ids, batch_input_ids, batch_target_ids


class CharBERTDataset(Dataset):

    def __init__(self, txt_file_path, tokenizer):
        self.tokenizer = tokenizer
        self.txt_file_path = txt_file_path
        with open(self.txt_file_path) as f1:
            self.lines = [line.strip() for line in f1]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        sample = self.lines[index]
        if len(sample) > (self.tokenizer.max_seq_len - 4):
            sample = sample[:(self.tokenizer.max_seq_len - 4)]
            # let the sequence stop at the last space to maintain full context of words
            # In other words, we don't want to split in the middle of a word
            sample = sample[:sample.rfind(' ')]

        orig_token_ids, input_ids, target_ids = self.tokenizer.mask_n_random_tokens(sample)
        orig_token_ids, input_ids, target_ids = torch.tensor(orig_token_ids).long(), torch.tensor(input_ids).long(), torch.tensor(target_ids).long()
        return orig_token_ids, input_ids, target_ids


if __name__ == '__main__':
    from char_bert_tokenizer import CharBERTTokenizer
    # IMPORTANT NOTE: keep the following list in the same order since we will used IN SHA ALLAH in Tashkeel model with the same order
    letters = [' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى', 'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر', 'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ']

    tokenizer = CharBERTTokenizer(letters, is_completely_random=False)
    text_file = 'sample_text.txt'
    dataset = CharBERTDataset(text_file, tokenizer)
    print(dataset[0])
    dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, dataset, batch_size=4)
    a = next(iter(dataloader))
    print(a)

