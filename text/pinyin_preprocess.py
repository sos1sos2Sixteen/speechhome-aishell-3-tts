# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base Processor for all processor."""

from typing import List
import re

_pad = ["pad"]
_eos = ["eos"]
_pause = ["sil", "#0", "#1", "#2", "#3"]

_initials = [
    "^",
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "x",
    "z",
    "zh",
]

_tones = ["1", "2", "3", "4", "5", "6"]

_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "ii",
    "iii",
    "in",
    "ing",
    "iong",
    "iou",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uen",
    "ueng",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
]

BAKER_SYMBOLS = _pad + _pause + _initials + [i + j for i in _finals for j in _tones] + _eos


PINYIN_DICT = {
    "a": ("^", "a"),
    "ai": ("^", "ai"),
    "an": ("^", "an"),
    "ang": ("^", "ang"),
    "ao": ("^", "ao"),
    "ba": ("b", "a"),
    "bai": ("b", "ai"),
    "ban": ("b", "an"),
    "bang": ("b", "ang"),
    "bao": ("b", "ao"),
    "be": ("b", "e"),
    "bei": ("b", "ei"),
    "ben": ("b", "en"),
    "beng": ("b", "eng"),
    "bi": ("b", "i"),
    "bian": ("b", "ian"),
    "biao": ("b", "iao"),
    "bie": ("b", "ie"),
    "bin": ("b", "in"),
    "bing": ("b", "ing"),
    "bo": ("b", "o"),
    "bu": ("b", "u"),
    "ca": ("c", "a"),
    "cai": ("c", "ai"),
    "can": ("c", "an"),
    "cang": ("c", "ang"),
    "cao": ("c", "ao"),
    "ce": ("c", "e"),
    "cen": ("c", "en"),
    "ceng": ("c", "eng"),
    "cha": ("ch", "a"),
    "chai": ("ch", "ai"),
    "chan": ("ch", "an"),
    "chang": ("ch", "ang"),
    "chao": ("ch", "ao"),
    "che": ("ch", "e"),
    "chen": ("ch", "en"),
    "cheng": ("ch", "eng"),
    "chi": ("ch", "iii"),
    "chong": ("ch", "ong"),
    "chou": ("ch", "ou"),
    "chu": ("ch", "u"),
    "chua": ("ch", "ua"),
    "chuai": ("ch", "uai"),
    "chuan": ("ch", "uan"),
    "chuang": ("ch", "uang"),
    "chui": ("ch", "uei"),
    "chun": ("ch", "uen"),
    "chuo": ("ch", "uo"),
    "ci": ("c", "ii"),
    "cong": ("c", "ong"),
    "cou": ("c", "ou"),
    "cu": ("c", "u"),
    "cuan": ("c", "uan"),
    "cui": ("c", "uei"),
    "cun": ("c", "uen"),
    "cuo": ("c", "uo"),
    "da": ("d", "a"),
    "dai": ("d", "ai"),
    "dan": ("d", "an"),
    "dang": ("d", "ang"),
    "dao": ("d", "ao"),
    "de": ("d", "e"),
    "dei": ("d", "ei"),
    "den": ("d", "en"),
    "deng": ("d", "eng"),
    "di": ("d", "i"),
    "dia": ("d", "ia"),
    "dian": ("d", "ian"),
    "diao": ("d", "iao"),
    "die": ("d", "ie"),
    "ding": ("d", "ing"),
    "din": ("d", "in"),         # non 普通话
    "diu": ("d", "iou"),
    "dong": ("d", "ong"),
    "dou": ("d", "ou"),
    "du": ("d", "u"),
    "duan": ("d", "uan"),
    "dui": ("d", "uei"),
    "dun": ("d", "uen"),
    "duo": ("d", "uo"),
    "e": ("^", "e"),
    "ei": ("^", "ei"),
    "en": ("^", "en"),
    "ng": ("^", "en"),
    "eng": ("^", "eng"),
    "er": ("^", "er"),
    "fa": ("f", "a"),
    "fan": ("f", "an"),
    "fang": ("f", "ang"),
    "fei": ("f", "ei"),
    "fen": ("f", "en"),
    "feng": ("f", "eng"),
    "fo": ("f", "o"),
    "fou": ("f", "ou"),
    "fu": ("f", "u"),
    "ga": ("g", "a"),
    "gai": ("g", "ai"),
    "gan": ("g", "an"),
    "gang": ("g", "ang"),
    "gao": ("g", "ao"),
    "ge": ("g", "e"),
    "gei": ("g", "ei"),
    "gen": ("g", "en"),
    "geng": ("g", "eng"),
    "gong": ("g", "ong"),
    "gou": ("g", "ou"),
    "gu": ("g", "u"),
    "gua": ("g", "ua"),
    "guai": ("g", "uai"),
    "guan": ("g", "uan"),
    "guang": ("g", "uang"),
    "gui": ("g", "uei"),
    "gun": ("g", "uen"),
    "guo": ("g", "uo"),
    "ha": ("h", "a"),
    "hai": ("h", "ai"),
    "han": ("h", "an"),
    "hang": ("h", "ang"),
    "hao": ("h", "ao"),
    "he": ("h", "e"),
    "hei": ("h", "ei"),
    "hen": ("h", "en"),
    "heng": ("h", "eng"),
    "hong": ("h", "ong"),
    "hou": ("h", "ou"),
    "hu": ("h", "u"),
    "hua": ("h", "ua"),
    "huai": ("h", "uai"),
    "huan": ("h", "uan"),
    "huang": ("h", "uang"),
    "hui": ("h", "uei"),
    "hun": ("h", "uen"),
    "huo": ("h", "uo"),
    "ji": ("j", "i"),
    "jia": ("j", "ia"),
    "jian": ("j", "ian"),
    "jiang": ("j", "iang"),
    "jiao": ("j", "iao"),
    "jie": ("j", "ie"),
    "jin": ("j", "in"),
    "jing": ("j", "ing"),
    "jiong": ("j", "iong"),
    "jiu": ("j", "iou"),
    "ju": ("j", "v"),
    "jv": ("j", "v"),          # alternative spelling of ju
    "juan": ("j", "van"),
    "jue": ("j", "ve"),
    "jun": ("j", "vn"),
    "ka": ("k", "a"),
    "kai": ("k", "ai"),
    "kan": ("k", "an"),
    "kang": ("k", "ang"),
    "kao": ("k", "ao"),
    "ke": ("k", "e"),
    "kei": ("k", "ei"),
    "ken": ("k", "en"),
    "keng": ("k", "eng"),
    "kong": ("k", "ong"),
    "kou": ("k", "ou"),
    "ku": ("k", "u"),
    "kua": ("k", "ua"),
    "kuai": ("k", "uai"),
    "kuan": ("k", "uan"),
    "kuang": ("k", "uang"),
    "kui": ("k", "uei"),
    "kun": ("k", "uen"),
    "kuo": ("k", "uo"),
    "la": ("l", "a"),
    "lai": ("l", "ai"),
    "lan": ("l", "an"),
    "lang": ("l", "ang"),
    "lao": ("l", "ao"),
    "le": ("l", "e"),
    "lei": ("l", "ei"),
    "leng": ("l", "eng"),
    "li": ("l", "i"),
    "lia": ("l", "ia"),
    "lian": ("l", "ian"),
    "liang": ("l", "iang"),
    "liao": ("l", "iao"),
    "lie": ("l", "ie"),
    "lin": ("l", "in"),
    "ling": ("l", "ing"),
    "liu": ("l", "iou"),
    "lo": ("l", "o"),
    "long": ("l", "ong"),
    "lou": ("l", "ou"),
    "lu": ("l", "u"),
    "lv": ("l", "v"),
    "luan": ("l", "uan"),
    "lve": ("l", "ve"),
    "lue": ("l", "ve"),
    "lun": ("l", "uen"),
    "luo": ("l", "uo"),
    "ma": ("m", "a"),
    "mai": ("m", "ai"),
    "man": ("m", "an"),
    "mang": ("m", "ang"),
    "mao": ("m", "ao"),
    "me": ("m", "e"),
    "mei": ("m", "ei"),
    "men": ("m", "en"),
    "meng": ("m", "eng"),
    "mi": ("m", "i"),
    "mian": ("m", "ian"),
    "miao": ("m", "iao"),
    "mie": ("m", "ie"),
    "min": ("m", "in"),
    "ming": ("m", "ing"),
    "miu": ("m", "iou"),
    "mo": ("m", "o"),
    "mou": ("m", "ou"),
    "mu": ("m", "u"),
    "na": ("n", "a"),
    "nai": ("n", "ai"),
    "nan": ("n", "an"),
    "nang": ("n", "ang"),
    "nao": ("n", "ao"),
    "ne": ("n", "e"),
    "nei": ("n", "ei"),
    "nen": ("n", "en"),
    "neng": ("n", "eng"),
    "ni": ("n", "i"),
    "nia": ("n", "ia"),
    "nian": ("n", "ian"),
    "niang": ("n", "iang"),
    "niao": ("n", "iao"),
    "nie": ("n", "ie"),
    "nin": ("n", "in"),
    "ning": ("n", "ing"),
    "niu": ("n", "iou"),
    "nong": ("n", "ong"),
    "nou": ("n", "ou"),
    "nu": ("n", "u"),
    "nv": ("n", "v"),
    "nuan": ("n", "uan"),
    "nve": ("n", "ve"),
    "nue": ("n", "ve"),
    "nuo": ("n", "uo"),
    "o": ("^", "o"),
    "ou": ("^", "ou"),
    "pa": ("p", "a"),
    "pai": ("p", "ai"),
    "pan": ("p", "an"),
    "pang": ("p", "ang"),
    "pao": ("p", "ao"),
    "pe": ("p", "e"),
    "pei": ("p", "ei"),
    "pen": ("p", "en"),
    "peng": ("p", "eng"),
    "pi": ("p", "i"),
    "pian": ("p", "ian"),
    "piao": ("p", "iao"),
    "pie": ("p", "ie"),
    "pin": ("p", "in"),
    "ping": ("p", "ing"),
    "po": ("p", "o"),
    "pou": ("p", "ou"),
    "pu": ("p", "u"),
    "qi": ("q", "i"),
    "qia": ("q", "ia"),
    "qian": ("q", "ian"),
    "qiang": ("q", "iang"),
    "qiao": ("q", "iao"),
    "qie": ("q", "ie"),
    "qin": ("q", "in"),
    "qing": ("q", "ing"),
    "qiong": ("q", "iong"),
    "qiu": ("q", "iou"),
    "qu": ("q", "v"),
    "quan": ("q", "van"),
    "que": ("q", "ve"),
    "qun": ("q", "vn"),
    "ran": ("r", "an"),
    "rang": ("r", "ang"),
    "rao": ("r", "ao"),
    "re": ("r", "e"),
    "ren": ("r", "en"),
    "reng": ("r", "eng"),
    "ri": ("r", "iii"),
    "rong": ("r", "ong"),
    "rou": ("r", "ou"),
    "ru": ("r", "u"),
    "rua": ("r", "ua"),
    "ruan": ("r", "uan"),
    "rui": ("r", "uei"),
    "run": ("r", "uen"),
    "ruo": ("r", "uo"),
    "sa": ("s", "a"),
    "sai": ("s", "ai"),
    "san": ("s", "an"),
    "sang": ("s", "ang"),
    "sao": ("s", "ao"),
    "se": ("s", "e"),
    "sen": ("s", "en"),
    "seng": ("s", "eng"),
    "sha": ("sh", "a"),
    "shai": ("sh", "ai"),
    "shan": ("sh", "an"),
    "shang": ("sh", "ang"),
    "shao": ("sh", "ao"),
    "she": ("sh", "e"),
    "shei": ("sh", "ei"),
    "shen": ("sh", "en"),
    "sheng": ("sh", "eng"),
    "shi": ("sh", "iii"),
    "shou": ("sh", "ou"),
    "shu": ("sh", "u"),
    "shua": ("sh", "ua"),
    "shuai": ("sh", "uai"),
    "shuan": ("sh", "uan"),
    "shuang": ("sh", "uang"),
    "shui": ("sh", "uei"),
    "shun": ("sh", "uen"),
    "shuo": ("sh", "uo"),
    "si": ("s", "ii"),
    "song": ("s", "ong"),
    "sou": ("s", "ou"),
    "su": ("s", "u"),
    "suan": ("s", "uan"),
    "sui": ("s", "uei"),
    "sun": ("s", "uen"),
    "suo": ("s", "uo"),
    "ta": ("t", "a"),
    "tai": ("t", "ai"),
    "tan": ("t", "an"),
    "tang": ("t", "ang"),
    "tao": ("t", "ao"),
    "te": ("t", "e"),
    "tei": ("t", "ei"),
    "teng": ("t", "eng"),
    "ti": ("t", "i"),
    "tian": ("t", "ian"),
    "tiao": ("t", "iao"),
    "tie": ("t", "ie"),
    "ting": ("t", "ing"),
    "tong": ("t", "ong"),
    "tou": ("t", "ou"),
    "tu": ("t", "u"),
    "tuan": ("t", "uan"),
    "tui": ("t", "uei"),
    "tun": ("t", "uen"),
    "tuo": ("t", "uo"),
    "wa": ("^", "ua"),
    "wai": ("^", "uai"),
    "wan": ("^", "uan"),
    "wang": ("^", "uang"),
    "wei": ("^", "uei"),
    "wen": ("^", "uen"),
    "weng": ("^", "ueng"),
    "wo": ("^", "uo"),
    "wu": ("^", "u"),
    "xi": ("x", "i"),
    "xia": ("x", "ia"),
    "xian": ("x", "ian"),
    "xiang": ("x", "iang"),
    "xiao": ("x", "iao"),
    "xie": ("x", "ie"),
    "xin": ("x", "in"),
    "xing": ("x", "ing"),
    "xiong": ("x", "iong"),
    "xiu": ("x", "iou"),
    "xu": ("x", "v"),
    "xuan": ("x", "van"),
    "xue": ("x", "ve"),
    "xun": ("x", "vn"),
    "ya": ("^", "ia"),
    "yan": ("^", "ian"),
    "yang": ("^", "iang"),
    "yao": ("^", "iao"),
    "ye": ("^", "ie"),
    "yi": ("^", "i"),
    "yin": ("^", "in"),
    "ying": ("^", "ing"),
    "yo": ("^", "iou"),
    "yong": ("^", "iong"),
    "you": ("^", "iou"),
    "yu": ("^", "v"),
    "yuan": ("^", "van"),
    "yue": ("^", "ve"),
    "yun": ("^", "vn"),
    "za": ("z", "a"),
    "zai": ("z", "ai"),
    "zan": ("z", "an"),
    "zang": ("z", "ang"),
    "zao": ("z", "ao"),
    "ze": ("z", "e"),
    "zei": ("z", "ei"),
    "zen": ("z", "en"),
    "zeng": ("z", "eng"),
    "zha": ("zh", "a"),
    "zhai": ("zh", "ai"),
    "zhan": ("zh", "an"),
    "zhang": ("zh", "ang"),
    "zhao": ("zh", "ao"),
    "zhe": ("zh", "e"),
    "zhei": ("zh", "ei"),
    "zhen": ("zh", "en"),
    "zheng": ("zh", "eng"),
    "zhi": ("zh", "iii"),
    "zhong": ("zh", "ong"),
    "zhou": ("zh", "ou"),
    "zhu": ("zh", "u"),
    "zhua": ("zh", "ua"),
    "zhuai": ("zh", "uai"),
    "zhuan": ("zh", "uan"),
    "zhuang": ("zh", "uang"),
    "zhui": ("zh", "uei"),
    "zhun": ("zh", "uen"),
    "zhuo": ("zh", "uo"),
    "zi": ("z", "ii"),
    "zong": ("z", "ong"),
    "zou": ("z", "ou"),
    "zu": ("z", "u"),
    "zuan": ("z", "uan"),
    "zui": ("z", "uei"),
    "zun": ("z", "uen"),
    "zuo": ("z", "uo"),
}


zh_pattern = re.compile("[\u4e00-\u9fa5]")


def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None



def get_pinyin_parser():
    from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
    from pypinyin.converter import DefaultConverter
    from pypinyin.core import Pinyin

    class MyConverter(NeutralToneWith5Mixin, DefaultConverter): pass
    my_pinyin = Pinyin(MyConverter())
    pinyin = my_pinyin.pinyin
    return pinyin

def get_phoneme_from_char_and_pinyin(chn_char: str, pinyin: List[str]) -> List[str]:
    # we do not need #4, use sil to replace it
    chn_char = chn_char.replace("#4", "")
    char_len = len(chn_char)
    i, j = 0, 0
    result = ["sil"]
    while i < char_len:
        cur_char = chn_char[i]
        if is_zh(cur_char):
            if pinyin[j][:-1] not in PINYIN_DICT:
                assert pinyin[j][-2] == "r"
                tone = pinyin[j][-1]
                a = pinyin[j][:-2]
                a1, a2 = PINYIN_DICT[a]
                result += [a1, a2 + tone, "er5"]
                if i + 2 < char_len and chn_char[i + 2] != "#":
                    result.append("#0")

                if chn_char[i + 1] == "儿": 
                    i += 2
                else: 
                    i += 1
                j += 1
            else:
                tone = pinyin[j][-1]
                a = pinyin[j][:-1]
                a1, a2 = PINYIN_DICT[a]
                result += [a1, a2 + tone]

                if i + 1 < char_len and chn_char[i + 1] != "#":
                    result.append("#0")

                i += 1
                j += 1
        elif cur_char == "#":
            result.append(chn_char[i : i + 2])
            i += 2
        else:
            # ignore the unknown char and punctuation
            # result.append(chn_char[i])
            i += 1
    if result[-1] == "#0":
        result = result[:-1]
    result.append("sil")
    assert j == len(pinyin)
    return result

pyparser = get_pinyin_parser()

def hans_to_pinyin(hans: str) -> List[str]: 
    from pypinyin import Style
    pinyin = pyparser(hans, style=Style.TONE3, errors="ignore")
    new_pinyin = []
    for x in pinyin:
        x = "".join(x)
        if "#" not in x:
            new_pinyin.append(x)
    return new_pinyin


import re 
from typing import List, Tuple, Optional
# note: the non-greedy match `*?`
pinyin_pattern = re.compile(r'(?P<body>[a-z]*?)(?P<erized>r?)(?P<tone>[0-9])')

def try_break_pinyin(pinyin: str) -> Optional[Tuple[str, str, bool]]: 
    # returns (body, tone, erized: bool)
    m = pinyin_pattern.match(pinyin)
    if m is None: return None
    else: 
        body, tone = m.group('body'), m.group('tone')
        erized = m.group('erized') == 'r'
        return body, tone, erized

def get_phoneme_from_pinyin(pinyins: List[str]) -> Tuple[List[str], List[str]]: 
    result = []
    problems = []
    for idx, cur_symbol in enumerate(pinyins): 
        breaked = try_break_pinyin(cur_symbol)

        if breaked is None: 
            problems.append(f'None pinyin symbol found: {cur_symbol}, ignored')
        else: 
            body, tone, erized = breaked
            if body not in PINYIN_DICT: 
                problems.append(f'unknown pinyin body: {body}, ignored')
            else: 
                initial, final = PINYIN_DICT[body]
                result += [initial, f'{final}{tone}']
                if erized: 
                    result.append('er5')
                if idx < len(pinyins) - 1:  # if not last
                    result.append('#0')
    result = ['sil'] + result + ['sil']
    return result, problems
            

if __name__ == '__main__': 
    text = '穿越亚细亚，探索一万五千公里的孤独。'
    pinyin = hans_to_pinyin(text)
    print(pinyin)
    print(get_phoneme_from_char_and_pinyin(text, pinyin))
