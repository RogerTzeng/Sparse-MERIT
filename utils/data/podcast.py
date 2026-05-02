import os
import numpy as np
import pandas as pd
SPLIT_MAP = {
    "train": "Train",
    "dev": "Development",
    "test1": "Test1",
    "test2": "Test2",
    "test3": "Test3"
}

# Load label
def load_utts(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    
    return cur_utts

def load_adv_emo_label(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct", "EmoDom", "EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_emo_label_snr(label_path, dtype, snr):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    
    if snr is not None:
        mix_noise_df = pd.read_csv('/mnt/External/Seagate/jingtong/SG-CustomNoise/mix_result_test3.csv', index_col=0)
        mix_noise_df = mix_noise_df[mix_noise_df["snr"] == snr]
        cur_df = cur_df[cur_df['FileName'].isin(mix_noise_df['file_name'])]

    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct", "EmoDom", "EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_arousal(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_valence(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_dominance(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoDom"]].to_numpy()

    return cur_utts, cur_labs

def load_cat_emo_label(label_path, dtype, classes):
    label_df = pd.read_csv(label_path, sep=",")
    if dtype == 'test3':
        label_df = label_df[(label_df['Surprise'] != 1) & (label_df['Fear'] != 1) & (label_df['Disgust'] != 1) & (label_df['Contempt'] != 1)]
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    # cur_labs = cur_df[["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]].to_numpy()
    cur_labs = cur_df[classes].to_numpy()

    return cur_utts, cur_labs

def load_cat_emo_label_snr(label_path, dtype, snr):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    
    if snr is not None:
        mix_noise_df = pd.read_csv('/mnt/External/Seagate/jingtong/SG-CustomNoise/mix_result_test3.csv', index_col=0)
        mix_noise_df = mix_noise_df[mix_noise_df["snr"] == snr]
        cur_df = cur_df[cur_df['FileName'].isin(mix_noise_df['file_name'])]

    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]].to_numpy()

    return cur_utts, cur_labs

def load_spk_id(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[(label_df["Split_Set"] == SPLIT_MAP[dtype])]
    cur_df = cur_df[(cur_df["SpkrID"] != "Unknown")]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_spk_ids = cur_df["SpkrID"].to_numpy().astype(np.int)
    # Cleanining speaker id
    uniq_spk_id = list(set(cur_spk_ids))
    uniq_spk_id.sort()
    for new_id, old_id in enumerate(uniq_spk_id):
        cur_spk_ids[cur_spk_ids == old_id] = new_id
    total_spk_num = len(uniq_spk_id)

    return cur_utts, cur_spk_ids, total_spk_num