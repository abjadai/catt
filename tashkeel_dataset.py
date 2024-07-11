
import os
import glob
import torch
import bw2ar
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def get_files(mypath, extension='*.txt'):
    return [y for x in os.walk(mypath, followlinks=True) for y in glob.glob(os.path.join(x[0], extension), recursive=True)]

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


class TashkeelDataset(Dataset):

    def __init__(self, txt_folder_path, tokenizer, max_seq_len=2048, tashkeel_to_text_ratio_threshold=0.8):
        self.tokenizer = tokenizer
        prepared_lines = []
        for filepath in tqdm(get_files(txt_folder_path, '*.txt')):
            with open(filepath) as f1:
                for line in f1:
                    # filter out all chars that ARE NOT arabic chars or tashkeel
                    clean_line = self.tokenizer.clean_text(line)
                    if clean_line != '':
                        prepared_lines.append(clean_line)

        total_lines_before_filteration = len(prepared_lines)
        # filter out small text (less than 5 chars + tashkeel combined) as well as long segments
        # more than the specified max_seq_len
        self.prepared_lines = list(filter(lambda x: len(x) > 5 and len(x) <= max_seq_len, prepared_lines))

        # filter out texts that have less than "tashkeel_to_text_ratio_threshold"
        tmp = []
        for line in tqdm(self.prepared_lines, desc=f'Filtering out texts that have tashkeel to text ratio less than {tashkeel_to_text_ratio_threshold}'):
            letter_n_tashkeel_pairs = self.tokenizer.split_tashkeel_from_text(line, test_match=False)
            text, tashkeel = zip(*letter_n_tashkeel_pairs)
            if (len(tashkeel) - tashkeel.count(self.tokenizer.no_tashkeel_tag)) / len(tashkeel) >= tashkeel_to_text_ratio_threshold:
                tmp.append(line)

        self.prepared_lines = tmp
        print('Filteration process successfully completed!!')
        print(f'Remaining useable examples: {len(tmp)} / {total_lines_before_filteration}')
        self.prepared_lines = sorted(self.prepared_lines, key=lambda x: len(x))

    def __len__(self):
        return len(self.prepared_lines)

    def __getitem__(self, index):
        sample = self.prepared_lines[index]
        input_ids, target_ids = self.tokenizer.encode(sample, test_match=False)
        return input_ids, target_ids


class PrePaddingDataLoader(DataLoader):
    def __init__(self, tokenizer, *args, **kwargs):
        super(PrePaddingDataLoader, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        input_ids_list = []
        target_ids_list = []
        for input_ids, target_ids in batch:
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
        batch_input_ids = pad_seq(input_ids_list, batch_first=True, padding_value=self.tokenizer.letters_map['<PAD>'], prepadding=False)
        batch_target_ids = pad_seq(target_ids_list, batch_first=True, padding_value=self.tokenizer.tashkeel_map['<PAD>'], prepadding=False)
        return batch_input_ids, batch_target_ids


if __name__ == '__main__':
    from tashkeel_tokenizer import TashkeelTokenizer
    tokenizer = TashkeelTokenizer()

    dl_num_workers = 2
    batch_size = 32
    max_seq_len = 1024

    train_txt_folder_path = 'dataset/train'
    val_txt_folder_path = 'dataset/val'
    test_txt_folder_path = 'dataset/test'

    threshold = 0.6
    print('Creating Train Dataset...')
    train_dataset = TashkeelDataset(train_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Train Dataloader...')
    train_dataloader = PrePaddingDataLoader(tokenizer, train_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=True)

    print('Creating Val Dataset...')
    val_dataset = TashkeelDataset(val_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Val Dataloader...')
    val_dataloader = PrePaddingDataLoader(tokenizer, val_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False)

    print('Creating Test Dataset...')
    test_dataset = TashkeelDataset(test_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Test Dataloader...')
    test_dataloader = PrePaddingDataLoader(tokenizer, test_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False)


