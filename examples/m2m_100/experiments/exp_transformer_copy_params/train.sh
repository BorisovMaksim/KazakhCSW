export CUDA_VISIBLE_DEVICES="0"

fairseq-train ../exp_transformer/fairseq_data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-pairs "kk-ru" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 400000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 16384 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 --ignore-unused-valid-subsets --wandb-project exp_transformer_copy_params --fp16


#   --lang-dict "$lang_list" \
#   --lang-pairs "$lang_pairs" \