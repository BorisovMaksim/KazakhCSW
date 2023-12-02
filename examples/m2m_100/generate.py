import os
from argparse import ArgumentParser
from pathlib import Path

from calculate_blue_score import calculate_blue

os.environ['CUDA_VISIBLE_DEVICES'] = "1"



def fairseq_generate(data_path, subset, model_path, src_lang, tgt_lang, bpe, beam, save_path, batch_size, task, lang_pairs):
    out_path = Path(save_path) / 'generated' /f"{subset}.txt"
    ref_out_path = Path(save_path) / 'generated' / f"{subset}_ref.txt"
    hyp_out_path = Path(save_path) / 'generated' / f"{subset}_hyp.txt"
    
    out_path.parent.mkdir(exist_ok=True, parents=True)
    ref_out_path.parent.mkdir(exist_ok=True, parents=True)
    hyp_out_path.parent.mkdir(exist_ok=True, parents=True)
    
    os.system(f"""fairseq-generate {data_path} \
        --batch-size {batch_size} \
        --path {model_path} \
        --source-lang {src_lang} \
        --target-lang {tgt_lang} \
        --remove-bpe {bpe} \
        --beam {beam} \
        --task {task} \
        --scoring sacrebleu	{f'--lang-pairs {lang_pairs}' if lang_pairs else ""} \
        --gen-subset {subset} > {out_path}""") # --eval-bleu-remove-bpe \
    
    os.system(f"""cat {out_path}  | grep -P "^T" | sort -V | cut -f 2- > {ref_out_path}""")
    os.system(f"""cat {out_path}  | grep -P "^H" | sort -V | cut -f 3- > {hyp_out_path}""")
    
    calculate_blue(ref_path=ref_out_path,
                   hyp_path=hyp_out_path,
                   detokenize_hyp=True,
                   detokenize_ref=True)
    
    
    
    
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')
    parser.add_argument('--bpe', type=str, default='sentencepiece')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--task', type=str, default='translation')
    parser.add_argument('--lang_pairs', type=str, default='')

    fairseq_generate(**vars(parser.parse_args()))