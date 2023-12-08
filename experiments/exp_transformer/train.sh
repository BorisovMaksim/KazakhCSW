export CUDA_VISIBLE_DEVICES="0"


# fairseq-train fairseq_data \
#     --valid-subset valid,valid1      \
#     --clip-norm 0.0 --dropout 0.2 --max-tokens 8192 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --criterion label_smoothed_cross_entropy \
#     --keep-interval-updates 3 --save-interval-updates 500  --log-interval 50 \
#     --arch transformer --save-dir checkpoints/transformer --lr-scheduler inverse_sqrt \
#     --warmup-updates 8000 --warmup-init-lr 1e-7 --lr 2e-3 --upsample-primary 1 --wandb-project exp_transformer  --ignore-unused-valid-subsets \
#     --scoring sacrebleu




fairseq-train fairseq_data \
    --arch transformer \
    --max-tokens 4096 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --weight-decay 1e-4 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000  --lr 5e-4   \
    --dropout 0.1  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --no-epoch-checkpoints  --log-interval 50 \
    --save-dir checkpoints/transformer_copy_cs_paper\
    --wandb-project exp_transformer  --ignore-unused-valid-subsets \
    --scoring sacrebleu