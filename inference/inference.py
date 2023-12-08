import os
import urllib3

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TRANSFORMERS_CACHE'] = '/mnt/s3-data/itmo/models/'
os.environ['CURL_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sacrebleu.metrics import BLEU, CHRF, TER
import argparse
import json
from models import MODELS
from datasets import DATASETS
from pathlib import Path
from torch.utils.data import DataLoader



def infer(model_name, dataset_name, src_lang, tgt_lang, device, batch_size, num_workers, save_path=None):
    if save_path is None:
        save_path = Path(f'./inference_results/{dataset_name}_{model_name}_inference')
        save_path.parent.mkdir(parents=True, exist_ok=True)
    model = MODELS[model_name](src_lang=src_lang.replace("mix_", ""), 
                               tgt_lang=tgt_lang.replace("mix_", ""),
                               device=device).to(device)
    dataset = DATASETS[dataset_name](src_lang=src_lang,
                             tgt_lang=tgt_lang)
    dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=batch_size)
    inputs, translations, references = [], [], []        
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataset) // batch_size):
        src_text, reference = batch
        if not pd.isnull(src_text) and not pd.isnull(reference): 
            translated_text = model.predict(src_text)
            print(f"\nSource: {src_text[0]}\nReference: {reference[0]}\nTranslated: {translated_text[0]}\n\n")
            inputs += src_text
            translations +=  [str(s) for s in translated_text]
            references += [str(s) for s in reference]
        # torch.cuda.empty_cache()

            
       
    bleu = BLEU()
    blue_score = bleu.corpus_score(translations, [references])
    print(f"{blue_score=} for {dataset=} and {model_name=}")
    
    with open(str(save_path) + "_src.txt", 'w', encoding='utf-8') as f,  open(str(save_path) + "_translated.txt", 'w', encoding='utf-8') as g:
        for src_text, translated_text in zip(inputs, translations):
            f.write(src_text.replace("\n", "") + "\n")
            g.write(translated_text.replace("\n", "") + "\n")
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model_name', type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CSW', 'RTC'])
    parser.add_argument('--src_lang', type=str, required=True, choices=['mix_kk', 'mix_ru', 'kk', 'ru'])
    parser.add_argument('--tgt_lang', type=str, required=True, choices=['mix_kk', 'mix_ru' ,'kk', 'ru'])
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    infer(**vars(args))


    
    









