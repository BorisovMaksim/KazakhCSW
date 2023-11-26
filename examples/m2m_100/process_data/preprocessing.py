import gzip
import argparse
from string import punctuation
from pathlib import Path
from tqdm import tqdm


"""
python preprocessing.py --input ~/datasets/wikimatrix/processed/kk-ru --src_lang kk --tgt_lang ru 
"""

def len_no_punc(s, punc):
    return len([ch for ch in s if ch in punc])

def filter_overpunc(len_npunc, len_sen):
    return len_npunc < 0.5*len_sen

def read_hist(f, threshold_character):
    ch = []
    for line in f:
        c = line[0]
        if c == threshold_character:
            break
        ch.append(c)
    return ch
    

def main(input, encoding, src_lang, tgt_lang, threshold, threshold_character, histograms):
    punc = punctuation + "—|–"
    
    out_filename = Path(input).parent / (Path(input).stem + "_processed")
    print('Processing file {}'.format(input))
    with open(input + '.' + src_lang, 'r', encoding=encoding) as fsrc_in, open(input + '.' + tgt_lang, 'r', encoding=encoding) as ftgt_in:
        data_src = fsrc_in.read().split("\n")
        data_tgt = ftgt_in.read().split("\n")
        
    assert len(data_src) == len(data_tgt)
        
    data_src_no_punc = []
    data_tgt_no_punc = []
    print(f"---- Filtering lines with a lot of punctuation ----")
    for src, tgt in zip(data_src, data_tgt):
        if src != "" and tgt != "":
            nchar_npunc_src = len_no_punc(src, punc)
            nchar_npunc_tgt = len_no_punc(tgt, punc)
            if filter_overpunc(nchar_npunc_src, len(src)) and filter_overpunc(nchar_npunc_tgt, len(tgt)):
                data_src_no_punc.append(src.strip())
                data_tgt_no_punc.append(tgt.strip())
            else:
                print(f"Filtered:\n{src}\n{tgt}\n")
    print(f"---- Deduplicating data ----")
    data_dedup = list(set(zip(data_src_no_punc, data_tgt_no_punc)))
    
    print(f"---- Histogram cleaning ----")
    with(open("{}/{}".format(histograms, src_lang), 'r', encoding='utf8')) as f:
        ch1 = read_hist(f, threshold_character)

    with(open("{}/{}".format(histograms, tgt_lang), 'r', encoding='utf8')) as f:
        ch2 = read_hist(f, threshold_character)
        
    data_src_hist_cleaned = []
    data_tgt_hist_cleaned = []
    for src_line, tgt_line in data_dedup:
        if src_line != "" and tgt_line != "":
            cnt1 = len([c for c in src_line.strip() if c in ch1])
            cnt2 = len([c for c in tgt_line.strip() if c in ch2])
            if cnt1 / len(src_line) > threshold and cnt2 / len(tgt_line) > threshold:
                data_src_hist_cleaned.append(src_line)
                data_tgt_hist_cleaned.append(tgt_line)
            else:
                print("{} {} {} \n{} {} {}".format(src_lang, cnt1 / len(src_line), src_line.strip(), tgt_lang, cnt2 / len(tgt_line), tgt_line.strip()))
                
    assert len(data_src_hist_cleaned) == len(data_tgt_hist_cleaned)
    
    
    print(f"Source data length = {len(data_src)}")
    print(f"Num duplicates = {len(data_src) - len(data_dedup)}")
    print(f"Num dirty sentences = {len(data_dedup) - len(data_src_hist_cleaned)}")
    print(f"Total filtered = {len(data_src) - len(data_src_hist_cleaned)}")
    print(f"Preprocessed data length = {len(data_src_hist_cleaned)}")
    
    with open(str(out_filename) + '.' + src_lang, 'w', encoding=encoding) as fsrc, open(str(out_filename) + '.' + tgt_lang, 'w', encoding=encoding) as ftgt:
        for src_line, tgt_line in zip(data_src_hist_cleaned, data_tgt_hist_cleaned):
            fsrc.write(src_line +  '\n')
            ftgt.write(tgt_line +  '\n')
                        

                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
    parser.add_argument('--threshold_character', type=str, default=']', help='Threshold character')
    parser.add_argument('--histograms', type=str, default="./clean_hists", help='Path to histograms')

    main(**vars(parser.parse_args()))
