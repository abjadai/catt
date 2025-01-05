import random

class CharBERTTokenizer:

    def __init__(self, letters, masking_prob=0.15, max_seq_len=2048,
                                mask_token='<MASK>', pad_token='<PAD>',
                                bos_token='<BOS>', eos_token='<EOS>', is_completely_random=True):
        self.masking_prob = masking_prob
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.letters = [self.pad_token, self.bos_token, self.eos_token] + list(letters) + [self.mask_token]
        self.letters_map = {c:i for i,c in enumerate(self.letters)}
        self.mask_token_id = self.letters_map[self.mask_token]
        self.pad_token_id = self.letters_map[self.pad_token]
        self.bos_token_id = self.letters_map[self.bos_token]
        self.eos_token_id = self.letters_map[self.eos_token]
        self.special_tokens_ids = [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id]
        self.is_completely_random = is_completely_random


    def set_mask_token(self, new_mask_token):
        self.mask_token = new_mask_token
        letters = list(self.letters[3:-1])
        self.letters = [self.pad_token, self.bos_token, self.eos_token] + list(letters) + [self.mask_token]
        self.letters_map = {c:i for i,c in enumerate(self.letters)}


    def tokenize(self, text, bos=True, eos=True):
        token_ids =  [self.letters_map[c] for c in text.strip()]
        if bos:
            token_ids = [self.bos_token_id] + token_ids
        if eos:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids


    def detokenize(self, token_ids):
        text = ''.join([self.letters[i] for i in token_ids])
        text = text.replace(self.bos_token, ' ').replace(self.eos_token, ' ').replace(self.pad_token, ' ')
        text = ' '.join(text.strip().split())
        return text


    def mask_n_random_tokens(self, text):
        token_ids = self.tokenize(text)
        output_label = []
        output = []
        seed = None
        if not self.is_completely_random:
        # this will make the examples have determinstic random mask, i.e., it will generate the same masks for the same example for each time
            seed = 0
        random.seed(seed)
        # 15% of the tokens would be replaced
        for token_id in token_ids:
            prob = random.random()
            # remove cls and sep token
            #token_id = tokenizer(token)['input_ids'][1:-1]
            if (prob < self.masking_prob) and (not token_id in self.special_tokens_ids):
                prob /= self.masking_prob
                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(self.mask_token_id)
                # 10% chance change token to random token
                elif prob < 0.9:
                    random_token_id = random.randrange(len(self.letters_map))
                    # ensure that the sampled token is not a special token
                    while random_token_id in self.special_tokens_ids:
                        random_token_id = random.randrange(len(self.letters_map))
                    output.append(random_token_id)
                # 10% chance change token to current token
                else:
                    output.append(token_id)
                output_label.append(token_id)
            else:
                output.append(token_id)
                # IMPORTANT NOTE: we add pad_token_id to ignore calculating loss for it.
                # This is based on the assumption that the CrossEntropyLoss is used with ignore_index as follows:
                # torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
                output_label.append(self.pad_token_id)
        return token_ids, output, output_label

