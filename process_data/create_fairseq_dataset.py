import os 
from argparse import ArgumentParser
from pathlib import Path
import tempfile
import sentencepiece as spm
import shutil
from tqdm import tqdm
import uuid


from datasets import DATASETS


# Example
""" 
python create_fairseq_dataset.py --train_datasets nu kaznu opus RTC statmt wikimatrix --val_datasets mix_test --test_datasets nu kaznu ntrex mix_test ted --src_lang kk --tgt_lang ru --save_path /home/itmo/Projects/fairseq/examples/m2m_100/experiments/exp_transformer_all_data/
"""

class CreateFairseqDataset:
    def __init__(self, train_datasets, val_datasets, test_datasets, src_lang, tgt_lang, num_workers, bpe, joined_dictionary, save_path):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_workers = num_workers
        self.bpe = bpe
        self.joined_dictionary = joined_dictionary
        self.save_path = save_path



    def read_data(self, data_path):
        src_data = []
        tgt_data = []
        with open(str(data_path) + "." + self.src_lang) as f, open(str(data_path) + "." + self.tgt_lang) as g:
            for src_line, tgt_line in zip(f, g):
                if src_line and tgt_line:
                    src_data.append(src_line)
                    tgt_data.append(tgt_line)
        return src_data, tgt_data

    def save_temp_data(self, data_src_save, data_tgt_save):
        
        tmp_filename = str(uuid.uuid4())
        tmp_src_file = Path(self.save_path) / 'tmp' / (tmp_filename + f".{self.src_lang}") 
        tmp_tgt_file =  Path(self.save_path) / 'tmp' / (tmp_filename + f".{self.tgt_lang}")
        
        tmp_src_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(tmp_src_file, 'w') as f_out, open(tmp_tgt_file, 'w') as g_out:
            for src_line, tgt_line in tqdm(zip(data_src_save, data_tgt_save), total=len(data_src_save)):
                f_out.write(src_line + "\n")
                g_out.write(tgt_line + "\n")
        print(f"{tmp_src_file=}")
        print(f"{tmp_tgt_file=}\n")
        
        return Path(tmp_src_file).with_suffix('')
        
    def apply_bpe(self, sp, src_data, tgt_data, filter=False):
        data_src_save = []
        data_tgt_save = [] 
        for src_line, tgt_line in tqdm(zip(src_data, tgt_data), desc='Applying bpe', total=len(src_data)):
                src_encoded =sp.encode(src_line, out_type=str)
                tgt_encoded = sp.encode(tgt_line, out_type=str)
                min_length = min(len(src_encoded), len(tgt_encoded))
                max_length = max(len(src_encoded), len(tgt_encoded))
                
                    
                if filter and (max_length > 250 or max_length / min_length > 3):
                    continue
                data_src_save.append(" ".join(src_encoded))
                data_tgt_save.append(" ".join(tgt_encoded))
                
        print(f"Length of training data is {len(src_data)}")  
        print(f"Filtered {len(src_data) - len(data_src_save)} rows from training data")    
        print(f"Length of processed training data is {len(data_src_save) }\n")
        return data_src_save, data_tgt_save

    def train_spm(self, train_src_data, train_tgt_data):
        print("Training sentencepiece bpe model...")
        tmp_all_train_data = tempfile.NamedTemporaryFile()
        with open(tmp_all_train_data.name, 'w') as f:
            for line in train_src_data + train_tgt_data:
                f.write(line + "\n")
        spm.SentencePieceTrainer.train(input=tmp_all_train_data.name, model_prefix='spm', vocab_size=32000, model_type='bpe')
        shutil.move('spm.model', self.save_path)
        shutil.move('spm.vocab', self.save_path)
        print("Sentencepiece model is trained!")
        
    def process_dataset(self, sp, data_path):
        src_data, tgt_data = self.read_data(data_path)
        data_src_save, data_tgt_save = self.apply_bpe(sp, src_data, tgt_data, filter=False)
        tmp_data_path = self.save_temp_data(data_src_save, data_tgt_save)
        return tmp_data_path
        
        

    def fairseq_preprocess(self):    
        destdir = Path(self.save_path) / 'fairseq_data'
        model_path = Path(self.save_path) / 'spm.model'
        
        train_src_data = []
        train_tgt_data = []
        for dataset in self.train_datasets:
            train_path = DATASETS[dataset] / 'processed' / 'train' / 'kk-ru_processed'
            assert train_path.with_suffix("." + self.src_lang).exists() and train_path.with_suffix("." + self.tgt_lang).exists()
            src_data, tgt_data = self.read_data(train_path)
            train_src_data.extend(src_data)
            train_tgt_data.extend(tgt_data)      
            
            
        if not model_path.exists():
            self.train_spm(train_src_data, train_tgt_data)
            
        sp  = spm.SentencePieceProcessor(model_file=model_path.as_posix())
        train_data_src_save, train_data_tgt_save = self.apply_bpe(sp, train_src_data, train_tgt_data, filter=True)
        trainpref = self.save_temp_data(train_data_src_save, train_data_tgt_save)
                    
        
            
        validpref = []
        for dataset in self.val_datasets:
            valid_path = DATASETS[dataset] / 'processed' / 'dev' / 'kk-ru_processed'
            assert valid_path.with_suffix("." + self.src_lang).exists() and valid_path.with_suffix("." + self.tgt_lang).exists()
            temp_val_path = self.process_dataset(sp, valid_path)
            validpref.append(str(temp_val_path))
            
        validpref = ",".join(validpref)
        print(f"Dev datasets: {validpref}")
            
        testpref = []
        for dataset in self.test_datasets:
            test_path = DATASETS[dataset] / 'processed' / 'test' / 'kk-ru_processed'
            assert test_path.with_suffix("." + self.src_lang).exists() and test_path.with_suffix("." + self.tgt_lang).exists()
            temp_test_path = self.process_dataset(sp, test_path)
            testpref.append(str(temp_test_path))
            
        testpref = ",".join(testpref)
        print(f"Test datasets: {testpref}")
                
            
        os.system(f"""fairseq-preprocess \
        --source-lang {self.src_lang} --target-lang {self.tgt_lang} --bpe {self.bpe}  \
        --trainpref   {trainpref}  \
        --validpref {validpref}   \
        --testpref  {testpref} \
        --destdir {str(destdir)}    --workers {self.num_workers}  {'--joined-dictionary' if self.joined_dictionary else ''}
    """)
        
        tmp_path = Path(self.save_path) / 'tmp'
        assert tmp_path.is_dir()
        os.system(f"rm -r {str(tmp_path)}")





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_datasets",  nargs='+', required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--val_datasets",  nargs='+', required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--test_datasets",  nargs='+', required=True, choices=list(DATASETS.keys()))
    
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--bpe', type=str, default='sentencepiece')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--joined_dictionary', type=bool, default=True)

    dataset = CreateFairseqDataset(**vars(parser.parse_args()))
    dataset.fairseq_preprocess()