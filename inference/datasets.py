import pandas as pd
from torch import nn
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class CodeSwitchKKRU(Dataset):
    def __init__(self, annotations_file="/mnt/s3-data/itmo/datasets/mix_test/Final_Корпус.xlsx", src_lang='kk', tgt_lang='ru'):
        self.test_data = pd.read_excel(annotations_file)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        row = self.test_data.iloc[idx]
        if self.src_lang == 'kk':
            src_text = row['Перевод на казахский']
        elif self.src_lang == 'ru':
            src_text = row['Перевод на русский']
        elif self.src_lang == 'mix_kk' or self.src_lang == 'mix_ru':
            src_text = row['Оригинал']
        else:
            raise ValueError(f"Wrong src_lang = {self.src_lang}")
        
        if self.tgt_lang == 'kk':
            tgt_text = row['Перевод на казахский']
        elif self.tgt_lang == 'ru':
            tgt_text = row['Перевод на русский']
        elif self.tgt_lang == 'mix_kk' or self.tgt_lang == 'mix_ru':
            tgt_text = row['Оригинал']
        else:
            raise ValueError(f"Wrong tgt_lang = {self.tgt_lang}")
        return src_text, tgt_text
    
    
    
class RTC(Dataset):
    def __init__(self, annotations_file="/home/itmo/datasets/rutweetcorp/twitter_corpus.csv", src_lang='kk', tgt_lang='ru',
                 cache_path='/home/itmo/datasets/rutweetcorp/twitter_corpus_processed.npy'):
        
        if Path(cache_path).exists():
            print(f"Loading RTC dataset from cache {cache_path}")
            self.ru_tweets_cleaned = np.load(cache_path)
        else:
            ru_tweets = pd.read_csv(annotations_file, lineterminator='\n')
            self.ru_tweets_cleaned = []
            for text in ru_tweets['text']:
                tokens = [tok for tok in str(text).split() if 'http' not in tok and tok[0] != "@" and tok[0] != "#" and tok and tok != 'RT']
                processed_text = " ".join(tokens)
                if processed_text != '':
                    self.ru_tweets_cleaned.append(processed_text)
            np.save(cache_path, self.ru_tweets_cleaned)

    def __len__(self):
        return len(self.ru_tweets_cleaned)

    def __getitem__(self, idx):
        src_text = self.ru_tweets_cleaned[idx]
        tgt_text = "unk"
        return src_text, tgt_text
    
    
    
DATASETS = {
    'CSW' : CodeSwitchKKRU,
    'RTC' : RTC
}