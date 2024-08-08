"""
functions to convert Arabic words/text into buckwalter encoding and vice versa
"""

import re
import sys

from catt.utils import arabic_utils

buck2uni = {
    "'": "\u0621",  # hamza-on-the-line
    "|": "\u0622",  # madda
    ">": "\u0623",  # hamza-on-'alif
    "&": "\u0624",  # hamza-on-waaw
    "<": "\u0625",  # hamza-under-'alif
    "}": "\u0626",  # hamza-on-yaa'
    "A": "\u0627",  # bare 'alif
    "b": "\u0628",  # baa'
    "p": "\u0629",  # taa' marbuuTa
    "t": "\u062A",  # taa'
    "v": "\u062B",  # thaa'
    "j": "\u062C",  # jiim
    "H": "\u062D",  # Haa'
    "x": "\u062E",  # khaa'
    "d": "\u062F",  # daal
    "*": "\u0630",  # dhaal
    "r": "\u0631",  # raa'
    "z": "\u0632",  # zaay
    "s": "\u0633",  # siin
    "$": "\u0634",  # shiin
    "S": "\u0635",  # Saad
    "D": "\u0636",  # Daad
    "T": "\u0637",  # Taa'
    "Z": "\u0638",  # Zaa' (DHaa')
    "E": "\u0639",  # cayn
    "g": "\u063A",  # ghayn
    "_": "\u0640",  # taTwiil
    "f": "\u0641",  # faa'
    "q": "\u0642",  # qaaf
    "k": "\u0643",  # kaaf
    "l": "\u0644",  # laam
    "m": "\u0645",  # miim
    "n": "\u0646",  # nuun
    "h": "\u0647",  # haa'
    "w": "\u0648",  # waaw
    "Y": "\u0649",  # 'alif maqSuura
    "y": "\u064A",  # yaa'
    "F": "\u064B",  # fatHatayn
    "N": "\u064C",  # Dammatayn
    "K": "\u064D",  # kasratayn
    "a": "\u064E",  # fatHa
    "u": "\u064F",  # Damma
    "i": "\u0650",  # kasra
    "~": "\u0651",  # shaddah
    "o": "\u0652",  # sukuun
    "`": "\u0670",  # dagger 'alif
    "{": "\u0671",  # waSla
}

# For a reverse transliteration (Unicode -> Buckwalter), a dictionary
# which is the reverse of the above buck2uni is essential.
uni2buck = {}

# Iterate through all the items in the buck2uni dict.
for key, value in buck2uni.items():
    # The value from buck2uni becomes a key in uni2buck, and vice
    # versa for the keys.
    uni2buck[value] = key

# add special characters
uni2buck["\ufefb"] = "lA"
uni2buck["\ufef7"] = "l>"
uni2buck["\ufef5"] = "l|"
uni2buck["\ufef9"] = "l<"


# clean the arabic text from unwanted characters that may cause problem while building the language model
def clean_text(text):
    text = re.sub(
        "[\ufeff]", "", text, flags=re.UNICODE
    )  # strip Unicode Character 'ZERO WIDTH NO-BREAK SPACE' (U+FEFF). For more info, check http://www.fileformat.info/info/unicode/char/feff/index.htm
    text = arabic_utils.remove_non_arabic(text)
    text = arabic_utils.strip_tashkeel(text)
    text = arabic_utils.strip_tatweel(text)
    return text


# convert a single word into buckwalter and vice versa
def transliterate_word(input_word, direction="bw2ar"):
    output_word = ""
    # Loop over each character in the string, bw_word.
    for char in input_word:
        # Look up current char in the dictionary to get its
        # respective value. If there is no match, e.g., chars like
        # spaces, then just stick with the current char without any
        # conversion.
        if direction == "bw2ar":
            # print('in bw2ar')
            output_word += buck2uni.get(char, char)
        elif direction == "ar2bw":
            # print('in ar2bw')
            output_word += uni2buck.get(char, char)
        else:
            sys.stderr.write("Error: invalid direction!")
            sys.exit()
    return output_word


# convert a text into buckwalter and vice versa
def transliterate_text(input_text, direction="bw2ar"):
    output_text = ""
    for input_word in input_text.split(" "):
        output_text += transliterate_word(input_word, direction) + " "
    return output_text[:-1]  # remove the last space ONLY
