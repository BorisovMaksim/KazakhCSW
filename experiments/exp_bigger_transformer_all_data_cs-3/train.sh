export CUDA_VISIBLE_DEVICES="1"


fairseq-train ./fairseq_data  \
--valid-subset test3  \
--save-dir ./checkpoint/m2m/ --task translation_multi_simple_epoch \
--encoder-normalize-before \
--langs 'kk,ru' --lang-pairs 'kk-ru' --max-tokens 1800 --decoder-normalize-before \
--sampling-method temperature --sampling-temperature 1.5 --encoder-langtok src --decoder-langtok \
--criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 \
--max-update 3200000 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
 --log-interval 50 --no-epoch-checkpoints \
--seed 222 --log-format simple --patience 10 \
--arch transformer_wmt_en_de_big --encoder-layers 24 --decoder-layers 24 \
--encoder-ffn-embed-dim 8192 --decoder-ffn-embed-dim 8192 --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 \
--share-decoder-input-output-embed --share-all-embeddings --ddp-backend no_c10d  --wandb-project kk-ru-csw --fp16 --no-last-checkpoints \
--max-update 983680  --restore-file  ./checkpoint/m2m/checkpoint_best.pt
