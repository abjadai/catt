
import re
import numpy as np
import bw2ar
from utils import HARAKAT_PAT


class TashkeelTokenizer:

    def __init__(self):
        self.letters = [' ', '$', '&', "'", '*', '<', '>', 'A', 'D', 'E', 'H', 'S', 'T', 'Y', 'Z',
                        'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't',
                        'v', 'w', 'x', 'y', 'z', '|', '}'
                       ]
        self.letters = ['<PAD>', '<BOS>', '<EOS>'] + self.letters + ['<MASK>']

        self.no_tashkeel_tag = '<NT>'
        self.tashkeel_list = ['<NT>', '<SD>', '<SDD>', '<SF>', '<SFF>', '<SK>',
                               '<SKK>', 'F', 'K', 'N', 'a', 'i', 'o', 'u', '~']

        self.tashkeel_list = ['<PAD>', '<BOS>', '<EOS>'] + self.tashkeel_list

        self.tashkeel_map = {c:i for i,c in enumerate(self.tashkeel_list)}
        self.letters_map = {c:i for i,c in enumerate(self.letters)}
        self.inverse_tags = {
                 '~a': '<SF>',  # shaddah and fatHa
                 '~u': '<SD>',  # shaddah and Damma
                 '~i': '<SK>',  # shaddah and kasra
                 '~F': '<SFF>', # shaddah and fatHatayn
                 '~N': '<SDD>', # shaddah and Dammatayn
                 '~K': '<SKK>'  # shaddah and kasratayn
        }
        self.tags = {v:k for k,v in self.inverse_tags.items()}
        self.shaddah_last  = ['a~', 'u~', 'i~', 'F~', 'N~', 'K~']
        self.shaddah_first = ['~a', '~u', '~i', '~F', '~N', '~K']
        self.tahkeel_chars = ['F','N','K','a', 'u', 'i', '~', 'o']

    def clean_text(self, text):
        text = re.sub(u'[%s]' % u'\u0640', '', text) # strip tatweel
        text = text.replace('ٱ', 'ا')
        return ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]", " ", text,  flags=re.UNICODE).split())

    def check_match(self, text_with_tashkeel, letter_n_tashkeel_pairs):
        text_with_tashkeel = text_with_tashkeel.strip()
        # test if the reconstructed text with tashkeel is the same as the original one
        syn_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
        return syn_text == text_with_tashkeel or syn_text == self.unify_shaddah_position(text_with_tashkeel)

    def unify_shaddah_position(self, text_with_tashkeel):
        # unify the order of shaddah and the harakah to make shaddah always at the beginning
        for i in range(len(self.shaddah_first)):
            text_with_tashkeel = text_with_tashkeel.replace(self.shaddah_last[i], self.shaddah_first[i])
        return text_with_tashkeel

    def split_tashkeel_from_text(self, text_with_tashkeel, test_match=True):
        text_with_tashkeel = self.clean_text(text_with_tashkeel)
        text_with_tashkeel = bw2ar.transliterate_text(text_with_tashkeel, 'ar2bw')
        text_with_tashkeel = text_with_tashkeel.replace('`', '') # remove dagger 'alif

        # unify the order of shaddah and the harakah to make shaddah always at the beginning
        text_with_tashkeel = self.unify_shaddah_position(text_with_tashkeel)

        # remove duplicated harakat
        for i in range(len(self.tahkeel_chars)):
            text_with_tashkeel = text_with_tashkeel.replace(self.tahkeel_chars[i]*2, self.tahkeel_chars[i])

        letter_n_tashkeel_pairs = []
        for i in range(len(text_with_tashkeel)): # go over the whole text
            # check if the first character is a normal letter and the second character is a tashkeel
            if i < (len(text_with_tashkeel) - 1) and not text_with_tashkeel[i] in self.tashkeel_list and text_with_tashkeel[i+1] in self.tashkeel_list:
                # IMPORTANT: check if tashkeel is Shaddah, then there might be another Tashkeel char associated with it. If so,
                # replace both Shaddah and the Tashkeel chars with the appropriate tag
                if text_with_tashkeel[i+1] == '~':
                    # IMPORTANT: the following if statement depends on the concept of short circuit!!
                    # The first condition checks if there are still more chars before it access position i+2
                    # "text_with_tashkeel[i+2]" since it causes "index out of range" exception. Notice that
                    # Shaddah here is put in the first position before the Harakah.
                    if i+2 < len(text_with_tashkeel) and f'~{text_with_tashkeel[i+2]}' in self.inverse_tags:
                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.inverse_tags[f'~{text_with_tashkeel[i+2]}']))
                    else:
                        # if it is only Shaddah, just add it to the list
                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], '~'))
                else:
                    letter_n_tashkeel_pairs.append((text_with_tashkeel[i], text_with_tashkeel[i+1]))
            # if the character at position i is a normal letter and has no Tashkeel, then add
            # it with the tag "self.no_tashkeel_tag"
            # IMPORTANT: this elif block ensures also that there is no two or more consecutive tashkeel other than shaddah
            elif not text_with_tashkeel[i] in self.tashkeel_list:
                letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.no_tashkeel_tag))

        if test_match:
            # test if the split is done correctly by ensuring that we can retrieve back the original text
            assert self.check_match(text_with_tashkeel, letter_n_tashkeel_pairs)
        return [('<BOS>', '<BOS>')] + letter_n_tashkeel_pairs + [('<EOS>', '<EOS>')]

    def combine_tashkeel_with_text(self, letter_n_tashkeel_pairs):
        combined_with_tashkeel = []
        for letter, tashkeel in letter_n_tashkeel_pairs:
            combined_with_tashkeel.append(letter)
            if tashkeel in self.tags:
                combined_with_tashkeel.append(self.tags[tashkeel])
            elif tashkeel != self.no_tashkeel_tag:
                combined_with_tashkeel.append(tashkeel)
        text = ''.join(combined_with_tashkeel)
        return text

    def encode(self, text_with_tashkeel, test_match=True):
        letter_n_tashkeel_pairs = self.split_tashkeel_from_text(text_with_tashkeel, test_match)
        text, tashkeel = zip(*letter_n_tashkeel_pairs)
        input_ids = [self.letters_map[c] for c in text]
        target_ids = [self.tashkeel_map[c] for c in tashkeel]
        return np.array(input_ids, dtype=np.int64), np.array(target_ids, dtype=np.int64)

    def filter_tashkeel(self, tashkeel):
        tmp = []
        for i, t in enumerate(tashkeel):
            if i != 0 and t == '<BOS>':
                t = self.no_tashkeel_tag
            elif i != (len(tashkeel) - 1) and t == '<EOS>':
                t = self.no_tashkeel_tag
            tmp.append(t)
        tashkeel = tmp
        return tashkeel

    def decode(self, input_ids, target_ids):
        ar_texts = []
        for j in range(len(input_ids)):
            letters = [self.letters[i] for i in input_ids[j]]
            tashkeel = [self.tashkeel_list[i] for i in target_ids[j]]

            letters = list(filter(lambda x: x != '<BOS>' and x != '<EOS>' and x != '<PAD>', letters))
            tashkeel = self.filter_tashkeel(tashkeel)
            tashkeel = list(filter(lambda x: x != '<BOS>' and x != '<EOS>' and x != '<PAD>', tashkeel))

            # VERY IMPORTANT NOTE: zip takes min(len(letters), len(tashkeel)) and discard the reset of letters / tashkeels
            letter_n_tashkeel_pairs = list(zip(letters, tashkeel))
            bw_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
            ar_text = bw2ar.transliterate_text(bw_text, 'bw2ar')
            ar_texts.append(ar_text)
        return ar_texts

    def get_tashkeel_with_case_ending(self, text, case_ending=True):
        text_split = self.split_tashkeel_from_text(text, test_match=False)
        text_spaces_indecies = [i for i, el in enumerate(text_split) if el == (' ', '<NT>')]
        new_text_split = []
        for i, el in enumerate(text_split):
            if not case_ending and (i+1) in text_spaces_indecies:
                el = (el[0], '<NT>') # no case ending
            new_text_split.append(el)
        letters, tashkeel = zip(*new_text_split)
        return letters, tashkeel

    def remove_tashkeel(self, text):
        text = HARAKAT_PAT.sub('', text)
        text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
        text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
        return text

