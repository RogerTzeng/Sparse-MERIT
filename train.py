# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import math
import random

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from torchsummary import summary
from transformers import AutoModel
import importlib
# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils

from BSSE_SE.WavLM import WavLM, WavLMConfig 
from BSSE_SE.BLSTM import BLSTM_multi, BLSTM_multi_no_ws
from BSSE_SE.util import get_feature
from torch.utils.tensorboard import SummaryWriter 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--checkpoint", type=int, default=0)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument("--ratio", type=float)

parser.add_argument("--pooling_type", type=str, default="MeanPooling")
parser.add_argument("--gate_type", type=str, default="Sparse_GatingNetwork")
parser.add_argument("--experts", type=int, default=5)
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
MODEL_PATH = args.model_path

if not os.path.exists(MODEL_PATH):
   os.makedirs(MODEL_PATH)

writer = SummaryWriter(MODEL_PATH + '/log')

import json
from collections import defaultdict
config_path = "config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()

# Total number of samples
total_samples = len(train_df)

# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}

print(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)


# Print or return the tensor
print(class_weights_tensor)


total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype, classes)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    clean_audio_path = audio_path.replace('NoisyAudios', 'Audios')
    clean_wavs = utils.load_audio(clean_audio_path, cur_utts)
    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs)
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        clean_wav_set = utils.WavSet(clean_wavs)
        clean_wav_set.save_norm_stat(MODEL_PATH+"/clean_train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
            wav_mean_clean = total_dataset["train"].datasets[3].wav_mean
            wav_std_clean = total_dataset["train"].datasets[3].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
        clean_wav_set = utils.WavSet(clean_wavs, wav_mean=wav_mean_clean, wav_std=wav_std_clean)

    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.CAT_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts, clean_wav_set])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=16,
        collate_fn=utils.collate_fn_wav_lab_mask_data_dist
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

checkpoint = torch.load('./pretrained_models/WavLM-Large.pt', weights_only=True)
cfg = WavLMConfig(checkpoint['cfg'])
cfg.encoder_layerdrop = 0
ssl_model = WavLM(cfg)
ssl_model.load_state_dict(checkpoint['model'])

ssl_model.eval(); ssl_model.cuda()
for name, param in ssl_model.feature_extractor.named_parameters():
    param.requires_grad = False 

## Freeze ssl model
# for param in ssl_model.parameters():
#     param.requires_grad = False

########## Implement pooling method ##########
feat_dim = 1024

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)

else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)
pool_state_dict = torch.load("./pretrained_models/pretrained_pool.pt", weights_only=True)
pool_model.load_state_dict(pool_state_dict)
pool_model.cuda()

concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, len(classes), dropout=0.5)
ser_model_state_dict = torch.load("./pretrained_models/pretrained_ser.pt", weights_only=True)
ser_model.load_state_dict(ser_model_state_dict)
ser_model.eval(); ser_model.cuda()

se_model = BLSTM_multi_no_ws()
se_model_state_dict = torch.load("./pretrained_models/pretrained_se.pth.tar", weights_only=True)
se_model_state_dict = {k.replace('model_SE.', ''): v for k, v in se_model_state_dict['model'].items() if k.replace('model_SE.', '') in se_model.state_dict().keys()}
se_model.load_state_dict(se_model_state_dict)
se_model.eval(); se_model.cuda()

mmoe = net.MMoE(num_experts=args.experts, gate_type=args.gate_type, k=1, feature_dim=feat_dim, num_layers=25, num_tasks=2)
mmoe.eval(); mmoe.cuda()

min_epoch=0
min_loss=1e10
epoch=0

# ssl_total_params = sum(p.numel() for p in ssl_model.parameters())
# ws_total_params = sum(p.numel() for p in weighted_sum.parameters())
# pool_total_params = sum(p.numel() for p in pool_model.parameters())
# ser_total_params = sum(p.numel() for p in ser_model.parameters())
# se_total_params = sum(p.numel() for p in se_model.parameters())

# print("SE+SER: ", ssl_total_params+se_total_params+ssl_total_params+ws_total_params+pool_total_params+ser_total_params)
# print("Multi: ", ssl_total_params+se_total_params+ws_total_params+pool_total_params+ser_total_params)

ssl_opt = torch.optim.AdamW(ssl_model.parameters(), LR*0.5)
ser_opt = torch.optim.AdamW(ser_model.parameters(), LR)
se_opt = torch.optim.AdamW(se_model.parameters(), LR)
mmoe_opt = torch.optim.AdamW(mmoe.parameters(), LR)

# scaler = GradScaler()
ssl_opt.zero_grad(set_to_none=True)
ser_opt.zero_grad(set_to_none=True)
se_opt.zero_grad(set_to_none=True)
mmoe_opt.zero_grad(set_to_none=True)

if is_attentive_pooling:
    pool_opt = torch.optim.AdamW(pool_model.parameters(), LR)
    pool_opt.zero_grad(set_to_none=True)

if args.checkpoint:
    print("Continue training...")
    ssl_checkpoint = torch.load("/path/to/your/final_ssl.pt")
    ssl_model.load_state_dict(ssl_checkpoint['model'])
    ssl_opt.load_state_dict(ssl_checkpoint['optimizer'])

    pool_checkpoint = torch.load('/path/to/your/final_pool.pt')
    pool_model.load_state_dict(pool_checkpoint['model'])
    pool_opt.load_state_dict(pool_checkpoint['optimizer'])

    ser_checkpoint = torch.load("/path/to/your/final_ser.pt")
    ser_model.load_state_dict(ser_checkpoint['model'])
    ser_opt.load_state_dict(ser_checkpoint['optimizer'])

    se_checkpoint = torch.load("/path/to/your/final_se.pt")
    se_model.load_state_dict(se_checkpoint['model'])
    se_opt.load_state_dict(se_checkpoint['optimizer'])

    loss_checkpoint = torch.load("/path/to/your/final_loss.pt")
    min_loss = loss_checkpoint['best_loss']
    epoch = loss_checkpoint['epoch']

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_loss"])
lm.alloc_stat_type_list(["dev_loss"])

transform = get_feature()

iterations = 0
while epoch < EPOCHS:
    print("Epoch: ", epoch)
    lm.init_stat()
    ssl_model.train()
    pool_model.train()
    ser_model.train() 
    se_model.train()
    mmoe.train()   
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        clean_x = xy_pair[4]; clean_x=clean_x.cuda(non_blocking=True).float()

        rep, layer_results = ssl_model(x, padding_mask=~(mask.bool()), output_layer=ssl_model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        
        [se_features, ser_features], [se_gate_probs, ser_gate_probs] = mmoe(layer_reps)

        ssl = pool_model(ser_features, mask)

        emo_pred = ser_model(ssl)
        cat_loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)

        pred_log1p = se_model(x, layer_reps=se_features, output_wav=False, layer_norm=True)
        clean_x_re_norm = clean_x*(wav_std_clean+0.00000001) + wav_mean_clean
        clean_x_re_norm = (clean_x_re_norm - wav_mean)/(wav_std+0.00000001)
        clean_log1p = transform(clean_x_re_norm, ftype='log1p')[0][0]
        enhancement_criterion = nn.L1Loss()
        enhancement_loss = enhancement_criterion(pred_log1p, clean_log1p)

        loss = cat_loss + args.ratio * enhancement_loss

        writer.add_scalar('total_loss', loss, iterations)
        writer.add_scalar('cat_loss', cat_loss, iterations)
        writer.add_scalar('enh_loss', enhancement_loss, iterations)

        iterations+=1

        total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            ssl_opt.step()
            ser_opt.step()
            se_opt.step()
            mmoe_opt.step()

            if is_attentive_pooling:
                pool_opt.step()

            ssl_opt.zero_grad(set_to_none=True)
            ser_opt.zero_grad(set_to_none=True)
            se_opt.zero_grad(set_to_none=True)
            mmoe_opt.zero_grad(set_to_none=True)

            if is_attentive_pooling:
                pool_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_loss", loss)
 

    ssl_model.eval()
    pool_model.eval()
    ser_model.eval()
    se_model.eval()
    mmoe.eval()

    total_pred = [] 
    total_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()

        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        with torch.no_grad():
            rep, layer_results = ssl_model(x, padding_mask=~(mask.bool()), output_layer=ssl_model.cfg.encoder_layers, ret_layer_results=True)[0]
            layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

            [se_features, ser_features], _ = mmoe(layer_reps)

            ssl = pool_model(ser_features, mask)

            emo_pred = ser_model(ssl)

            total_pred.append(emo_pred)
            total_y.append(y)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)
    # Logging
    lm.add_torch_stat("dev_loss", loss)


    # Save model
    lm.print_stat()

        
    dev_loss = lm.get_stat("dev_loss")
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)
        save_model_list = ["ser", "ssl"]
        if is_attentive_pooling:
            save_model_list.append("pool")


        torch.save({'model': ser_model.state_dict(),
                    'optimizer': ser_opt.state_dict()}, \
            os.path.join(MODEL_PATH,  "final_ser.pt"))
        
        torch.save({'model': ssl_model.state_dict(),
                    'optimizer': ssl_opt.state_dict()}, \
            os.path.join(MODEL_PATH,  "final_ssl.pt"))
        
        torch.save({'model': se_model.state_dict(),
                    'optimizer': se_opt.state_dict()}, \
            os.path.join(MODEL_PATH,  "final_se.pt"))
        
        torch.save({'model': mmoe.state_dict(),
                    'optimizer': mmoe_opt.state_dict()}, \
            os.path.join(MODEL_PATH,  "final_mmoe.pt"))
        
        if is_attentive_pooling:
            torch.save({'model': pool_model.state_dict(),
                        'optimizer': pool_opt.state_dict()}, \
                os.path.join(MODEL_PATH,  "final_pool.pt"))
        
        torch.save({'epoch': epoch,
                    'best_loss': min_loss}, \
                os.path.join(MODEL_PATH,  "final_loss.pt"))
    
    epoch += 1