
from text.pinyin_preprocess import BAKER_SYMBOLS, hans_to_pinyin, get_phoneme_from_pinyin
from typing import List

_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '

SYMBOLS = [_pad] + list(_punctuation) + BAKER_SYMBOLS

_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
_id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}

def text_to_sequence(text: List[str]) -> List[int]: 
    return [_symbol_to_id[symbol] for symbol in text]

def sequence_to_text(sequence: List[int]) -> str: 
    return ' '.join(_symbol_to_id[i] for i in sequence)

def cn_cleaners(text: str) -> List[str]: 
    pinyin = hans_to_pinyin(text)
    phone, problem = get_phoneme_from_pinyin(pinyin)
    if len(problem) != 0: 
        print(f'problem encountered in phonemization: ')
        for p in problem: 
            print(p)
    return phone

def cn_to_sequence(text: str) -> List[int]: 
    return text_to_sequence(cn_cleaners(text))

def cleaned_to_sequence(pinyins: List[str]) -> List[int]: 
    phone, problem = get_phoneme_from_pinyin(pinyins)
    if len(problem) != 0: 
        print(f'problem encountered in phonemization: ')
        for p in problem: 
            print(p)
    return text_to_sequence(phone)


