import argparse
import os
import random
from glob import glob

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Mix clean audio with noise for the dataset.")
    parser.add_argument('--audio_dir', type=str, required=True, 
                        help="Path to the directory containing clean audio files.")
    parser.add_argument('--labels_csv', type=str, required=True, 
                        help="Path to the labels CSV file (e.g., labels_consensus.csv).")
    parser.add_argument('--noise_dir', type=str, required=True, 
                        help="Path to the directory containing noise files. Should have 'train', 'dev', and 'test' subdirectories.")
    parser.add_argument('--save_dir', type=str, required=True, 
                        help="Path to the directory where mixed audios will be saved.")
    parser.add_argument('--splits', nargs='+', default=['Test1'], 
                        help="Data splits to process (e.g., Train Development Test1 Test2).")
    parser.add_argument('--snr_train', type=float, default=5.0, 
                        help="SNR value for Train and Development splits.")
    parser.add_argument('--snr_test', type=float, default=-5.0, 
                        help="SNR value for Test splits.")
    parser.add_argument('--seed', type=int, default=0, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--save_csv', type=str, default=None, 
                        help="Optional path to save a CSV with mixed file names and SNR values.")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Target sample rate for audios.")
    return parser.parse_args()


def snr_noise_mix(s, ns, snr):
    """Mix speech and noise with a given Signal-to-Noise Ratio (SNR)."""
    snr_linear = 10 ** (snr / 10.0)
    speech_power = np.sum(np.power(s, 2)) / len(s)
    noise_power = np.sum(np.power(ns, 2)) / len(ns)
    
    if speech_power == 0:
        noise_update = ns
    else:
        noise_update = ns / np.sqrt(snr_linear * noise_power / speech_power)
    
    return noise_update + s


def s_n_p_align(s, ns):
    """Align noise to speech length by rolling and repeating/truncating."""
    if len(ns) > 0:
        ns = np.roll(ns, random.randint(0, len(ns)))
        
    if len(s) > len(ns):
        # Repeat noise if it's shorter than speech
        count = len(s) // len(ns) + 1
        new_ns = np.tile(ns, count)[:len(s)]
    else:
        # Truncate noise if it's longer than speech
        new_ns = ns[:len(s)]
        
    return s, new_ns


def process_split(args, split_name, df, split_save_dir):
    """Process a single data split, mixing all its files with noise."""
    filenames = df['FileName'].to_list()
    
    if split_name in ['Train', 'Development']:
        snr_range = [args.snr_train]
        noise_sub_dir = 'train' if split_name == 'Train' else 'dev'
    elif 'Test' in split_name:
        snr_range = [args.snr_test]
        noise_sub_dir = 'test'
    else:
        # Fallback for custom splits
        snr_range = [args.snr_test]
        noise_sub_dir = ''
        
    noise_files = glob(os.path.join(args.noise_dir, noise_sub_dir, '*'))
    if not noise_files:
        print(f"Warning: No noise files found in {os.path.join(args.noise_dir, noise_sub_dir)}. Skipping split {split_name}.")
        return [], []
        
    random.shuffle(noise_files)
    noise_idx = 0
    
    total_file_list = []
    total_snr_list = []
    
    for filename in tqdm(filenames, desc=f"Processing {split_name}"):
        clean_path = os.path.join(args.audio_dir, filename)
        if not os.path.exists(clean_path):
            print(f"File not found: {clean_path}")
            continue
            
        s, sr = librosa.load(clean_path, sr=args.sr)
        
        # Get next noise file
        if noise_idx >= len(noise_files):
            random.shuffle(noise_files)
            noise_idx = 0
            
        noise_path = noise_files[noise_idx]
        noise_idx += 1
        
        ns, nsr = librosa.load(noise_path, sr=args.sr)
        snr = random.choice(snr_range)
        
        s, ns = s_n_p_align(s, ns)
        mix_result = snr_noise_mix(s, ns, snr)
        
        save_file_path = os.path.join(split_save_dir, filename)
        sf.write(save_file_path, mix_result, args.sr)
        
        total_file_list.append(filename)
        total_snr_list.append(snr)
        
    return total_file_list, total_snr_list


def main():
    args = get_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Load labels dataframe
    label_df = pd.read_csv(args.labels_csv)
    
    all_files = []
    all_snrs = []
    
    for split in args.splits:
        cur_df = label_df[label_df["Split_Set"] == split]
        if cur_df.empty:
            print(f"No files found for split {split} in the labels CSV.")
            continue
            
        files, snrs = process_split(args, split, cur_df, args.save_dir)
        all_files.extend(files)
        all_snrs.extend(snrs)
        
    if args.save_csv:
        result_df = pd.DataFrame({'file_name': all_files, 'snr': all_snrs})
        result_df.to_csv(args.save_csv, index=False)
        print(f"Saved mix results to {args.save_csv}")


if __name__ == '__main__':
    main()