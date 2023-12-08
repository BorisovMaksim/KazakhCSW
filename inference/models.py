from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBartTokenizer, MBartForConditionalGeneration
from googletrans import Translator

import torch
from torch import nn

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class NLLB(nn.Module):
    def __init__(self, src_lang, tgt_lang, device):
        super(NLLB, self).__init__()
        self.lang_codes = {'ru': 'rus_Cyrl', 
                           'kk': 'kaz_Cyrl',
                           'es': 'spa_Latn',
                           'en': 'eng_Latn',
                           'ind': 'hin_Deva'}
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        
    def predict(self, text):
        self.tokenizer.src_lang = self.lang_codes[self.src_lang]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[self.tgt_lang]],  no_repeat_ngram_size=4
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text



class NLLB_600M(NLLB):
    def __init__(self, src_lang, tgt_lang, device):
        super(NLLB_600M, self).__init__(src_lang, tgt_lang, device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", resume_download=True) #"facebook/nllb-200-distilled-600M") nllb-200-3.3B
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", resume_download=True) 
        
class NLLB_3B(NLLB):
    def __init__(self, src_lang, tgt_lang, device):
        super(NLLB_3B, self).__init__(src_lang, tgt_lang, device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", resume_download=True) #"facebook/nllb-200-distilled-600M") nllb-200-3.3B
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", resume_download=True) 
       
class NLLB_54B(NLLB):
    def __init__(self, *args, **kwargs):
        super(NLLB_54B, self).__init__(*args, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b", resume_download=True) #"facebook/nllb-200-distilled-600M") nllb-200-3.3B
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b", resume_download=True)  
        

class M2M100(nn.Module):
    def __init__(self, src_lang, tgt_lang, device) -> None:
        super(M2M100, self).__init__()
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", resume_download=True)
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", resume_download=True)
        self.tokenizer.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        
    def predict(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)

        generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang), no_repeat_ngram_size=4)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    
class MBART(nn.Module):
    def __init__(self, src_lang, tgt_lang, device) -> None:
        super(MBART, self).__init__()
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", resume_download=True)
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", resume_download=True)
        self.lang_codes = {'ru': 'ru_RU', 
                           'kk': 'kk_KZ'}
        self.tokenizer.src_lang = self.lang_codes[src_lang] 
        self.tgt_lang = tgt_lang
        self.device = device
        
        
    def predict(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[self.tgt_lang]],
                                                no_repeat_ngram_size=4)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

       
       
class MBART_C25(nn.Module):
    def __init__(self, src_lang, tgt_lang, device) -> None:
        super(MBART_C25, self).__init__()
        self.lang_codes = {'ru': 'ru_RU', 
                           'kk': 'kk_KZ'}
        self.model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25', resume_download=True)
        self.tokenizer  = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25', resume_download=True)#, src_lang = self.lang_codes[src_lang])
        self.tokenizer.src_lang = self.lang_codes[src_lang] 
        self.tokenizer.tgt_lang = self.lang_codes[tgt_lang] 
        self.tgt_lang = tgt_lang
        self.device = device
    def predict(self, text):
        batch  = self.tokenizer(text, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**batch, decoder_start_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[self.tgt_lang]])
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output
    
    
       
class GoogleTranslate(nn.Module):
    def __init__(self, src_lang, tgt_lang) -> None:
        super(GoogleTranslate, self).__init__()
        self.translator = Translator()
        
        self.srs_lang = src_lang
        self.tgt_lang = tgt_lang
    def predict(self, text):
        while True:
            try:
                translated_text = self.translator.translate(text, src=self.srs_lang, dest=self.tgt_lang).text
            except Exception:
                continue
            break
        
        return translated_text
    
class Identity(nn.Module):
    def __init__(self, src_lang, tgt_lang, device) -> None:
        super(Identity, self).__init__()
    def predict(self, text):        
        return text
 



MODELS = {
    'NLLB_3B' : NLLB_3B,
    'NLLB_600M' : NLLB_600M,
    'NLLB_54B' : NLLB_54B,
    'M2M100': M2M100,
    'MBART': MBART,
    'MBART_C25': MBART_C25,
    'GOOGLE': GoogleTranslate,
    'Identity': Identity
}