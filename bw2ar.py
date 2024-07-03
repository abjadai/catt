"""
functions to convert Arabic words/text into buckwalter encoding and vice versa
"""

import sys
import re
import utils

buck2uni = {
            "'": u"\u0621", # hamza-on-the-line
            "|": u"\u0622", # madda
            ">": u"\u0623", # hamza-on-'alif
            "&": u"\u0624", # hamza-on-waaw
            "<": u"\u0625", # hamza-under-'alif
            "}": u"\u0626", # hamza-on-yaa'
            "A": u"\u0627", # bare 'alif
            "b": u"\u0628", # baa'
            "p": u"\u0629", # taa' marbuuTa
            "t": u"\u062A", # taa'
            "v": u"\u062B", # thaa'
            "j": u"\u062C", # jiim
            "H": u"\u062D", # Haa'
            "x": u"\u062E", # khaa'
            "d": u"\u062F", # daal
            "*": u"\u0630", # dhaal
            "r": u"\u0631", # raa'
            "z": u"\u0632", # zaay
            "s": u"\u0633", # siin
            "$": u"\u0634", # shiin
            "S": u"\u0635", # Saad
            "D": u"\u0636", # Daad
            "T": u"\u0637", # Taa'
            "Z": u"\u0638", # Zaa' (DHaa')
            "E": u"\u0639", # cayn
            "g": u"\u063A", # ghayn
            "_": u"\u0640", # taTwiil
            "f": u"\u0641", # faa'
            "q": u"\u0642", # qaaf
            "k": u"\u0643", # kaaf
            "l": u"\u0644", # laam
            "m": u"\u0645", # miim
            "n": u"\u0646", # nuun
            "h": u"\u0647", # haa'
            "w": u"\u0648", # waaw
            "Y": u"\u0649", # 'alif maqSuura
            "y": u"\u064A", # yaa'
            "F": u"\u064B", # fatHatayn
            "N": u"\u064C", # Dammatayn
            "K": u"\u064D", # kasratayn
            "a": u"\u064E", # fatHa
            "u": u"\u064F", # Damma
            "i": u"\u0650", # kasra
            "~": u"\u0651", # shaddah
            "o": u"\u0652", # sukuun
            "`": u"\u0670", # dagger 'alif
            "{": u"\u0671", # waSla
}

# For a reverse transliteration (Unicode -> Buckwalter), a dictionary
# which is the reverse of the above buck2uni is essential.
uni2buck = {}

# Iterate through all the items in the buck2uni dict.
for (key, value) in buck2uni.items():
    # The value from buck2uni becomes a key in uni2buck, and vice
    # versa for the keys.
    uni2buck[value] = key

# add special characters
uni2buck[u"\ufefb"] = "lA"
uni2buck[u"\ufef7"] = "l>"
uni2buck[u"\ufef5"] = "l|"
uni2buck[u"\ufef9"] = "l<"

# clean the arabic text from unwanted characters that may cause problem while building the language model
def clean_text(text):
    text = re.sub(u"[\ufeff]", "", text,  flags=re.UNICODE) # strip Unicode Character 'ZERO WIDTH NO-BREAK SPACE' (U+FEFF). For more info, check http://www.fileformat.info/info/unicode/char/feff/index.htm
    text = utils.remove_non_arabic(text)
    text = utils.strip_tashkeel(text)
    text = utils.strip_tatweel(text)
    return text

# convert a single word into buckwalter and vice versa
def transliterate_word(input_word, direction='bw2ar'):
    output_word = ''
    # Loop over each character in the string, bw_word.
    for char in input_word:
        # Look up current char in the dictionary to get its
        # respective value. If there is no match, e.g., chars like
        # spaces, then just stick with the current char without any
        # conversion.
        # if type(char) == bytes:
        #    char = char.decode('ascii')
        if direction == 'bw2ar':
            #print('in bw2ar')
            output_word += buck2uni.get(char, char)
        elif direction == 'ar2bw':
            #print('in ar2bw')
            output_word += uni2buck.get(char, char)
        else:
            sys.stderr.write('Error: invalid direction!')
            sys.exit()
    return output_word


# convert a text into buckwalter and vice versa
def transliterate_text(input_text, direction='bw2ar'):
    output_text = ''
    for input_word in input_text.split(' '):
        output_text += transliterate_word(input_word, direction) + ' '

    return output_text[:-1] # remove the last space ONLY


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: INPUT TEXT | python {} DIRECTION(bw2ar|ar2bw)'.format(sys.argv[1]))
        exit(1)
    for line in sys.stdin:
        line = line if sys.argv[1] == 'bw2ar' else clean_text(line)
        output_text = transliterate_text(line, direction=str(sys.argv[1]))
        if output_text.strip() != '':
            sys.stdout.write('{}\n'.format(output_text.strip()))



