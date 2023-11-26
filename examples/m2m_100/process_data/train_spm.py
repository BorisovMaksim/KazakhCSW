from argparse import ArgumentParser
import sentencepiece as spm
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

"""
python train_spm.py --dataset_paths ~/datasets/nu/processed/train/kk-ru_processed ~/datasets/kaznu/processed/train/kk
-ru_processed --src_lang kk --tgt_lang ru --save_path ~/Projects/fairseq/examples/m2m_100/experiments/exp_transformer/
"""


def main(src_lang, tgt_lang, dataset_paths, save_path):
    model_path = Path(save_path) / 'spm.model'
    
    if not model_path.exists():
        tmp = tempfile.NamedTemporaryFile()
        all_texts = []
        for data_path in dataset_paths:
            print(f"Addind {data_path} ...")
            data = []
            with open(data_path + "." + src_lang) as f, open(data_path + "." + tgt_lang) as g:
                for line in f:
                    if line:
                        data.append(line)
                for line in g:
                    if line:
                        data.append(line)
                all_texts += data
                
        with open(tmp.name, 'w') as f:
            for line in all_texts:
                f.write(line + "\n")
                
        
        spm.SentencePieceTrainer.train(input=tmp.name, model_prefix='spm', vocab_size=32000, model_type='bpe')
        shutil.move('spm.model', save_path)
        shutil.move('spm.vocab', save_path)
    
    sp  = spm.SentencePieceProcessor(model_file=model_path.as_posix())
    
    for data_path in dataset_paths:
        print(f"Encoding {data_path} ...")
        data_name = Path(data_path).parent.parent.parent.name
        data_save_path = Path(save_path) / 'data' / data_name / 'kk-ru'
        data_save_path.parent.mkdir(exist_ok=True, parents=True)
        
        data_src_save = []
        data_tgt_save = []
        with open(data_path + "." + src_lang) as f, open(data_path + "." + tgt_lang) as g:
            src_data, tgt_data = f.read().split("\n"), g.read().split("\n")
            for src_line, tgt_line in tqdm(zip(src_data, tgt_data), total=len(src_data)):
                if src_line and tgt_line:
                    src_encoded =sp.encode(src_line, out_type=str)
                    tgt_encoded = sp.encode(tgt_line, out_type=str)
                    min_length = min(len(src_encoded), len(tgt_encoded))
                    max_length = max(len(src_encoded), len(tgt_encoded))
                    
                    if max_length > 250 or max_length / min_length > 3:
                        continue
    
                    
                    data_src_save.append(" ".join(src_encoded))
                    data_tgt_save.append(" ".join(tgt_encoded))
                    
        print(f"Length of {data_name} is {len(src_data) }")  
        print(f"Filtered {len(src_data) - len(data_src_save)} rows from {data_name}")    
        print(f"Length of {data_name} processed is {len(data_src_save) }\n")  
        with open(str(data_save_path) + "." + src_lang, 'w') as f_out, open(str(data_save_path) + "." + tgt_lang, 'w') as g_out:
            for src_line, tgt_line in tqdm(zip(data_src_save, data_tgt_save), total=len(data_src_save)):
                f_out.write(src_line + "\n")
                g_out.write(tgt_line + "\n")
        



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_paths",  nargs='+', required=True)
    parser.add_argument("--save_path",  type=str, required=False)
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')
    args = parser.parse_args()
    main(**vars(args))
    