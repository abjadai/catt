"""
    @author
          ______         _                  _
         |  ____|       (_)           /\   | |
         | |__ __ _ _ __ _ ___       /  \  | | __ _ ___ _ __ ___   __ _ _ __ _   _
         |  __/ _` | '__| / __|     / /\ \ | |/ _` / __| '_ ` _ \ / _` | '__| | | |
         | | | (_| | |  | \__ \    / ____ \| | (_| \__ \ | | | | | (_| | |  | |_| |
         |_|  \__,_|_|  |_|___/   /_/    \_\_|\__,_|___/_| |_| |_|\__,_|_|   \__, |
                                                                              __/ |
                                                                             |___/
            Email: farisalasmary@gmail.com
            Date:  Mar 15, 2022
"""

# pip install git+https://github.com/pzelasko/kaldialign.git

from kaldialign import edit_distance


def cer(ref, hyp):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        ref (string): a space-separated ground truth string
        hyp (string): a space-separated hypothesis
    """
    ref, hyp, = ref.replace(' ', '').strip(), hyp.replace(' ', '').strip()
    info = edit_distance(ref, hyp)
    distance = info['total']
    ref_length = float(len(ref))
    
    data = {
                'insertions': info['ins'],
                'deletions': info['del'],
                'substitutions': info['sub'],
                'distance': distance,
                'ref_length': ref_length,
                'Error Rate': (distance / ref_length) * 100
           }
    
    return data


def wer(ref, hyp):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        ref (string): a space-separated ground truth string
        hyp (string): a space-separated hypothesis
    """
    
    # build mapping of words to integers
    b = set(ref.split() + hyp.split())
    word2char = dict(zip(b, range(len(b))))
    
    # map the words to a char array (Levenshtein packages only accepts strings)
    w1 = [chr(word2char[w]) for w in ref.split()]
    w2 = [chr(word2char[w]) for w in hyp.split()]
    
    info = edit_distance(''.join(w1), ''.join(w2))
    distance = info['total']
    ref_length = float(len(w1))
    
    data = {
                'insertions': info['ins'],
                'deletions': info['del'],
                'substitutions': info['sub'],
                'distance': distance,
                'ref_length': ref_length,
                'Error Rate': (distance / ref_length) * 100
           }
    
    return data 

