ssl_type=wavlm-large
GPU=0

experts=3
pool_type=AttentiveStatisticsPooling
for model in bsse_noisy_weight_cat_ser_4_classes_git_ws_1_1_5e-5_SSL_0_5_Original_weight_switch_mmoe_3experts; do
    for data in NoisyAudios; do
        for seed in 7; do
            CUDA_VISIBLE_DEVICES=$GPU python eval.py \
                --ssl_type=${ssl_type} \
                --pooling_type=${pool_type} \
                --gate_type=Sparse_GatingNetwork \
                --experts=${experts} \
                --model_path=model/$model/wavLM_adamW/${seed}  \
                --store_path=result/$model/wavLM_adamW/${seed}.txt \
                --testset=dev \
                --audio_path=/homes/jingtong/EMO/dataset/MSP-PODCAST-Publish-1.11/$data|| exit 0;
        done
    done
done