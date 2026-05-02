ssl_type=wavlm-large
GPU=0

# Train
pool_type=AttentiveStatisticsPooling
experts=3
for seed in 7; do
    for ratio in 1; do
        CUDA_VISIBLE_DEVICES=$GPU python train.py \
            --seed=${seed} \
            --ssl_type=${ssl_type} \
            --batch_size=32 \
            --accumulation_steps=8 \
            --lr=5e-5 \
            --epochs=20 \
            --pooling_type=${pool_type} \
            --experts=${experts} \
            --gate_type=Sparse_GatingNetwork \
            --model_path=model/bsse_noisy_weight_cat_ser_4_classes_git_ws_1_${ratio}_5e-5_SSL_0_5_Original_weight_switch_mmoe_${experts}experts/wavLM_adamW/${seed} \
            --ratio=${ratio} \
            --checkpoint 0 || exit 0;
    done
done