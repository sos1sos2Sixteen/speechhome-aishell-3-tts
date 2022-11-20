import os.path as osp 
import itertools
from dataclasses import dataclass, fields
from tqdm import tqdm
from collections import defaultdict
from text.pinyin_preprocess import get_phoneme_from_pinyin
from typing import List


@dataclass
class ContentLine: 
    sid: str
    hans: List[str]
    pinyin: List[str]

@dataclass
class MetaLine: 
    sid: str
    path: str
    spkid: str
    hans: str
    phones: str

def parse_content(path: str) -> List[ContentLine]: 
    res = []
    with open(path) as f: 
        for cols in (l.strip().split() for l in f): 
            sid, *content = cols
            hans = content[::2]
            pinyin = content[1::2]
            res.append(ContentLine(sid, hans, pinyin))
        return res 

def spk_from_sid(sid: str) -> str: 
    return sid[:7]

def content_to_meta(cl: ContentLine, join_path: str) -> MetaLine: 
    spkid = spk_from_sid(cl.sid)
    phones, problems = get_phoneme_from_pinyin(cl.pinyin)
    assert problems == [], f'in {cl}: {problems}'
    return MetaLine(
        cl.sid,
        osp.join(join_path, spkid, cl.sid), 
        spkid, 
        ''.join(cl.hans),
        ' '.join(phones)
    )

def fields_iter(dci): 
    for f in fields(dci): 
        yield getattr(dci, f.name)

def main(): 
    base = '/Netdata/shiyao/RAWDATA/AISHELL-3/'
    train_contents = parse_content(osp.join(base, 'train', 'content.txt'))
    test_contents  = parse_content(osp.join(base, 'test', 'content.txt'))


    metalines = [
        content_to_meta(tc, joinpath) for joinpath, tc in 
        tqdm(itertools.chain(
            itertools.product([osp.join('train', 'wav')], train_contents), 
            itertools.product([osp.join('test', 'wav')], test_contents)
        ), total=len(test_contents) + len(train_contents))
    ]

    metaline_by_spkid = defaultdict(list)
    for ml in metalines: metaline_by_spkid[ml.spkid].append(ml)

    print(f'total speakers: {len(metaline_by_spkid)}')

    abundant_speakers = [spkid for spkid, xs in metaline_by_spkid.items() if len(xs) > 219]
    print(f'number of abundant speakers: {len(abundant_speakers)}')


    train_metas = []
    test_metas  = [] 
    for ml in metalines: 
        if ml.spkid in abundant_speakers: 
            train_metas.append(ml)
        else: 
            test_metas.append(ml)

    print(f'train lines: {len(train_metas)}')
    print(f'test lines: {len(test_metas)}')


    with open(osp.join('filelists', 'ai3.train.txt'), 'w') as f: 
        for ml in train_metas: 
            print('|'.join(fields_iter(ml)), file=f)

    with open(osp.join('filelists', 'ai3.test.txt'), 'w') as f: 
        for ml in test_metas: 
            print('|'.join(fields_iter(ml)), file=f)

    test_speakers = sorted([spkid for spkid in metaline_by_spkid if spkid not in abundant_speakers])
    train_speakers = sorted(abundant_speakers)

    with open(osp.join('filelists', 'ai3.spkids'), 'w') as f: 
        for ids, spkid in enumerate(itertools.chain(train_speakers, test_speakers)): 
            print(f'{spkid}|{ids}', file=f)


if __name__ == '__main__': 
    main()