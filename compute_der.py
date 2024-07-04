
from tashkeel_tokenizer import TashkeelTokenizer
from tqdm import tqdm
import sys
import xer

tokenizer = TashkeelTokenizer()

if len(sys.argv) < 4:
    print('USAGE: {} ref_file hyp_file case_ending (yes|no)'.format(sys.argv[0]))
    sys.exit(1)

ref_file = sys.argv[1]
hyp_file = sys.argv[2]

case_ending = (sys.argv[3].lower() == 'yes')

ref_lines = open(ref_file).readlines()
hyp_lines = open(hyp_file).readlines()

total_der_distance = 0
total_der_ref_length = 0
total_wer_distance = 0
total_wer_ref_length = 0

mismatch_samples_count = 0

assert len(ref_lines) == len(hyp_lines), f"len(ref_lines), len(hyp_lines): {len(ref_lines)}, {len(hyp_lines)}"

for i in range(len(ref_lines)):
    ref = ref_lines[i].strip()
    hyp = hyp_lines[i].strip()
    ref = tokenizer.clean_text(ref)
    hyp = tokenizer.clean_text(hyp)
    ref_text = tokenizer.remove_tashkeel(ref)
    hyp_text = tokenizer.remove_tashkeel(hyp)
    ref_text = ref_text.replace('آ', 'ا').replace('إ', 'ا').replace('أ', 'ا').strip()
    hyp_text = hyp_text.replace('آ', 'ا').replace('إ', 'ا').replace('أ', 'ا').strip()
    if ref_text != '':
        wer_err =  xer.wer(ref_text, hyp_text)['Error Rate']
        # IMPORTANT NOTE:
        # if there is a little difference between the ref text and the hyp text after diacritization, just ignore the difference
        # Ususally, some models alter the original text or introduce new chars that prevent exact text matching, i.e., ref_text != hyp_text
        # if the difference is large, then count the example as mismatch BUT calculate the DER.
        # This works as a warning for the user to check the output text of the model
        if wer_err > 1: # if WER > 5%
            mismatch_samples_count += 1

    der_res = tokenizer.compute_der(ref, hyp, case_ending=case_ending)
    wer_res = tokenizer.compute_wer(ref, hyp, case_ending=case_ending)
    total_der_distance += der_res['distance']
    total_der_ref_length += der_res['ref_length']
    total_wer_distance += wer_res['distance']
    total_wer_ref_length += wer_res['ref_length']


print('Total DER: %{:0.3f}'.format((total_der_distance / total_der_ref_length) * 100 ))
print('Total WER: %{:0.3f}'.format((total_wer_distance / total_wer_ref_length) * 100 ))
print('Total mismatch samples:', mismatch_samples_count)
print('-'*89)
