
from torch.utils.data import DataLoader, Dataset

from catt.utils import get_files, pad_seq, tqdm


class TashkeelDataset(Dataset):

    def __init__(
        self,
        txt_folder_path,
        tokenizer,
        max_seq_len=2048,
        tashkeel_to_text_ratio_threshold=0.8,
    ):
        self.tokenizer = tokenizer
        prepared_lines = []
        for filepath in tqdm(get_files(txt_folder_path, "*.txt")):
            with open(filepath) as f1:
                for line in f1:
                    # filter out all chars that ARE NOT arabic chars or tashkeel
                    clean_line = self.tokenizer.clean_text(line)
                    if clean_line != "":
                        prepared_lines.append(clean_line)

        total_lines_before_filteration = len(prepared_lines)
        # filter out small text (less than 5 chars + tashkeel combined) as well as long segments
        # more than the specified max_seq_len
        self.prepared_lines = list(
            filter(lambda x: len(x) > 5 and len(x) <= max_seq_len, prepared_lines)
        )

        # filter out texts that have less than "tashkeel_to_text_ratio_threshold"
        tmp = []
        for line in tqdm(
            self.prepared_lines,
            desc=f"Filtering out texts that have tashkeel to text ratio less than {tashkeel_to_text_ratio_threshold}",
        ):
            letter_n_tashkeel_pairs = self.tokenizer.split_tashkeel_from_text(
                line, test_match=False
            )
            text, tashkeel = zip(*letter_n_tashkeel_pairs)
            if (len(tashkeel) - tashkeel.count(self.tokenizer.no_tashkeel_tag)) / len(
                tashkeel
            ) >= tashkeel_to_text_ratio_threshold:
                tmp.append(line)

        self.prepared_lines = tmp
        print("Filteration process successfully completed!!")
        print(
            f"Remaining useable examples: {len(tmp)} / {total_lines_before_filteration}"
        )
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
        batch_input_ids = pad_seq(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.letters_map["<PAD>"],
            prepadding=False,
        )
        batch_target_ids = pad_seq(
            target_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.tashkeel_map["<PAD>"],
            prepadding=False,
        )
        return batch_input_ids, batch_target_ids
