# python create_fairseq_dataset.py --train_datasets nu kaznu opus statmt wikimatrix RTC \
#--val_datasets mix_test --test_datasets nu kaznu ntrex mix_test ted --src_lang kk --tgt_lang ru --save_path /home/itmo/Projects/fairseq/examples/m2m_100/experiments/exp_transformer_all_data/


export CUDA_VISIBLE_DEVICES="0"



fairseq-train ../exp_transformer_all_data/fairseq_data \
    --arch multilingual_transformer --task multilingual_translation --lang-pairs kk-ru,ru-kk   \
    --share-decoders --share-encoders --share-decoder-input-output-embed  \
    --max-tokens 4096 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --weight-decay 1e-4 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000  --lr 5e-4   \
    --dropout 0.1  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --no-epoch-checkpoints  --log-interval 50 \
    --save-dir checkpoints/transformer_copy_cs_paper\
    --wandb-project exp_transformer  --ignore-unused-valid-subsets \
    --scoring sacrebleu --valid-subset test3