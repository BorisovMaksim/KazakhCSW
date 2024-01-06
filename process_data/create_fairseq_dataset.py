import os 
from argparse import ArgumentParser
from pathlib import Path
import tempfile
import sentencepiece as spm
import shutil
from tqdm import tqdm
import uuid
import random

from datasets import DATASETS


# Example
""" 
python create_fairseq_dataset.py --train_datasets nu kaznu opus RTC statmt wikimatrix --val_datasets mix_test --test_datasets nu kaznu ntrex mix_test ted --src_lang kk --tgt_lang ru --save_path /home/itmo/Projects/fairseq/examples/m2m_100/experiments/exp_transformer_all_data/
"""

class CreateFairseqDataset:
    def __init__(self, train_datasets, val_datasets, test_datasets, src_lang, tgt_lang, num_workers,
                 bpe, joined_dictionary, save_path, spm_model_path, data_prefix, align, percent, test_prefix, dict):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_workers = num_workers
        self.bpe = bpe
        self.joined_dictionary = joined_dictionary
        self.save_path = save_path
        self.spm_model_path = spm_model_path
        self.data_prefix = data_prefix
        self.align = align
        self.percent = percent
        self.test_prefix = test_prefix
        self.dict = dict



    def read_data(self, data_path):
        src_data = []
        tgt_data = []
        with open(str(data_path) + "." + self.src_lang) as f, open(str(data_path) + "." + self.tgt_lang) as g:
            for src_line, tgt_line in zip(f, g):
                if src_line != "" and tgt_line != "":
                    src_data.append(src_line.replace("\n", ""))
                    tgt_data.append(tgt_line.replace("\n", ""))
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
                    
                if min_length == 0 or (filter and (max_length > 250 or max_length / min_length > 2)):
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
    
    def align_dataset(self, src_data, tgt_data):
        tmp_dir =  Path(self.save_path) / 'tmp' 
        tmp_dir.mkdir(exist_ok=True, parents=True)
        
        tmp_align_file = tmp_dir / f"align.{self.src_lang}-{self.tgt_lang}"
        tmp_align_file_reverse = tmp_dir / f"align.{self.tgt_lang}-{self.src_lang}"
        tmp_align_forward = tmp_dir / "forward.align"
        tmp_align_reverse = tmp_dir / "reverse.align"
        tmp_align_gdf = tmp_dir / "gdf.align"        
        
        with open(tmp_align_file, 'w') as f, open(tmp_align_file_reverse, 'w') as f_reverse:
            for i, (src_line, tgt_line) in enumerate(zip(src_data, tgt_data)):
                src_line = src_line.replace("\n", "").strip().replace("|||", "")
                tgt_line = tgt_line.replace("\n", "").strip().replace("|||", "")
                
                save_line = src_line + " ||| " + tgt_line
                save_line_reverse = tgt_line  + " ||| " + src_line
                f.write(save_line + "\n")
                f_reverse.write(save_line_reverse + "\n")    
        print(f"Starting alignment: ")    
                
        os.system(f"../fast_align/build/fast_align -i {tmp_align_file} -d -o -v > {tmp_align_forward}")
        os.system(f"../fast_align/build/fast_align -i {tmp_align_file_reverse} -d -o -v -r > {tmp_align_reverse}")
        os.system(f"../fast_align/build/atools  -i {tmp_align_forward} -j {tmp_align_reverse} -c grow-diag-final-and > {tmp_align_gdf}")
        exit()
        
        
        all_minimal_units = []
        with open(tmp_align_gdf) as f:
            for line in f.read().split("\n"):
                if line == "":
                    continue
                indexes = line.split()
                src2tgt_units = {}
                for index in indexes:
                    src_index, tgt_index = index.split("-")
                    if src_index in src2tgt_units:
                        src2tgt_units[src_index] += [tgt_index]
                    else:
                        src2tgt_units[src_index] = [tgt_index]
                all_minimal_units.append(src2tgt_units)
                
        src_data_cs = []
        print(len(src_data), len(tgt_data), len(all_minimal_units))
        for src_line, tgt_line, minimal_units in zip(src_data, tgt_data, all_minimal_units):
            src_line = src_line.replace("\n", "").split()
            tgt_line = tgt_line.replace("\n", "").split()
    
            numwords = max(1, int(len(src_line) * self.percent / 100))
            minimal_units_keys = list(minimal_units.keys())
            minimal_units_keys = random.sample(minimal_units_keys, len(minimal_units_keys))
            
            num_replacements = 0
            for src_index in minimal_units_keys:
                if int(src_index) >= len(src_line):
                    continue
                for tgt_indexes in minimal_units[src_index]:
                    if len(tgt_indexes) > 1:
                        continue
                    if num_replacements > numwords:
                        break
                    num_replacements += len(tgt_indexes)
                    line_to_replace = " ".join([tgt_line[int(i)] for i in tgt_indexes if int(i) < len(tgt_line)])
                    # print(f"{src_line[int(src_index)]} -> {line_to_replace}")
                    src_line[int(src_index)] = line_to_replace
                else:
                    continue
                break
            src_data_cs.append(" ".join(src_line))
            
        assert len(src_data_cs) == len(tgt_data)
        return src_data_cs, tgt_data
        


    def fairseq_preprocess(self):    
        print(f"Start dataset creation")
        destdir = Path(self.save_path) / 'fairseq_data'
        model_path = Path(self.save_path) / 'spm.model' if self.spm_model_path is None  else Path(self.spm_model_path)
        Path(self.save_path).mkdir(exist_ok=True, parents=True)
        
        train_src_data = []
        train_tgt_data = []
        train_data_indexes = []
        for dataset in self.train_datasets:
            if dataset == 'RTC_subset':
                train_path = DATASETS[dataset] / 'processed' / 'train' / 'kk-ru_processed'
                
            else:
                train_path = DATASETS[dataset] / 'processed' / 'train' / self.data_prefix
            print(f"Processing {train_path}")
            assert train_path.with_suffix("." + self.src_lang).exists() and train_path.with_suffix("." + self.tgt_lang).exists()
            src_data, tgt_data = self.read_data(train_path)
            index = len(src_data) if len(train_data_indexes) == 0 else  len(src_data) + train_data_indexes[-1]
            
            train_src_data.extend(src_data)
            train_tgt_data.extend(tgt_data)    
            train_data_indexes.append(index)  
            
        if self.align:
            print(train_data_indexes)
            print(f"{train_data_indexes=}")
            train_src_data, train_tgt_data = self.align_dataset(src_data=train_src_data, 
                               tgt_data=train_tgt_data)
                        
        if not model_path.exists():
            self.train_spm(train_src_data, train_tgt_data)
            
        sp  = spm.SentencePieceProcessor(model_file=model_path.as_posix())
        
        train_data_src_save, train_data_tgt_save = self.apply_bpe(sp, train_src_data, train_tgt_data, filter=True)
        trainpref = self.save_temp_data(train_data_src_save, train_data_tgt_save)
        
        
            
        validpref = []
        for dataset in self.val_datasets:
            valid_path = DATASETS[dataset] / 'processed' / 'dev' / "kk-ru_processed"
            assert valid_path.with_suffix("." + self.src_lang).exists() and valid_path.with_suffix("." + self.tgt_lang).exists()
            temp_val_path = self.process_dataset(sp, valid_path)
            validpref.append(str(temp_val_path))
            
        validpref = ",".join(validpref)
        print(f"Dev datasets: {validpref}")
            
        testpref = []
        for dataset in self.test_datasets:
            test_path = DATASETS[dataset] / 'processed' / 'test' / self.test_prefix
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
        --destdir {str(destdir)}  --workers {self.num_workers}  {'--joined-dictionary' if self.joined_dictionary else ''} \
        {('--srcdict ' + self.dict) if self.dict else '' }  {('--tgtdict ' + self.dict) if self.dict else '' }
    """)
        
        tmp_path = Path(self.save_path) / 'tmp'
        assert tmp_path.is_dir()
        # os.system(f"rm -r {str(tmp_path)}")





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_datasets",  nargs='+', required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--val_datasets",  nargs='+', required=False, choices=list(DATASETS.keys()))
    parser.add_argument("--test_datasets",  nargs='+', required=False, choices=list(DATASETS.keys()))
    
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--bpe', type=str, default='sentencepiece')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--joined_dictionary', type=bool, default=False)
    parser.add_argument('--spm_model_path', type=str, default=None)
    parser.add_argument('--data_prefix', type=str, default="kk-ru_processed")
    parser.add_argument('--test_prefix', type=str, default="kk-ru_processed")
    parser.add_argument('--align', type=bool, default=False)
    parser.add_argument('--percent', type=int, default=15)
    parser.add_argument('--dict', type=str, default="")

    dataset = CreateFairseqDataset(**vars(parser.parse_args()))
    dataset.fairseq_preprocess()