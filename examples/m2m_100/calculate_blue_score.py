from sacrebleu.metrics import BLEU, CHRF, TER
from argparse import ArgumentParser
from sacremoses import MosesTokenizer, MosesDetokenizer



def calculate_blue(hyp_path, ref_path, detokenize_hyp, detokenize_ref):
    
    detokenizer = MosesDetokenizer()
    hyps = []
    refs = []
    with open(hyp_path, 'r') as f_hyp, open(ref_path, 'r') as f_ref:
        for hyp_line, ref_line in zip(f_hyp.read().split("\n"), f_ref.read().split("\n")):
            if hyp_line != "" and ref_line != "":
                if detokenize_hyp:
                    hyp_line = detokenizer.detokenize(hyp_line.split())
                if detokenize_ref:
                    ref_line = detokenizer.detokenize(ref_line.split())
                hyps.append(hyp_line)
                refs.append(ref_line)
    print(f"{len(hyps)=}")
    print(f"{len(refs)=}")
    print(hyps[:1])
    print(refs[:1], "\n")
    print(hyps[-1:])
    print(refs[-1:])
                
    
    bleu = BLEU(trg_lang='ru')
    score = bleu.corpus_score(hyps, [refs])
    print(f"{score = }")
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hyp_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--detokenize_hyp", type=bool, default=False)
    parser.add_argument("--detokenize_ref", type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
