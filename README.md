# Sparse MERIT: Joint Learning using Mixture-of-Expert-Based Representation for Speech Enhancement and Robust Emotion Recognition

Official repository for the TASLP submission:

**“Joint Learning using Mixture-of-Expert-Based Representation for Speech Enhancement and Robust Emotion Recognition”**  
(aka **Sparse MERIT**)  
[Read the paper on arXiv](https://arxiv.org/abs/2509.08470)

---

## Overview

This work studies **joint learning** of:
- **Speech Enhancement (SE)**
- **Speech Emotion Recognition (SER)**

Speech emotion recognition often degrades under noisy conditions. While speech enhancement (SE) can improve robustness, it introduces artifacts that obscure emotional cues and adds overhead. We propose **Sparse MERIT**, a Mixture-of-Experts (MoE) representation framework that performs **frame-wise sparse routing** to encourage task-adaptive specialization while reducing interference between SE and SER objectives.

Key ideas:
- A shared SSL backbone (WavLM-Large) provides frame-level representations.
- Task-specific routing selects experts (Top-1 in Sparse MERIT) to produce task-adaptive features.
- Joint training optimizes SE and SER objectives while mitigating negative transfer.

## Model Architecture

![Sparse MERIT Framework](/homes/jingtong/EMO/MSP-Podcast/Sparse-MERIT/Framework.png)

Sparse MERIT applies a frame-wise MoE layer over multi-layer self-supervised speech representations:
1. **Layer-Wise Representation Construction**: Hidden representations are extracted from the pretrained WavLM model.
2. **Mixture-of-Experts Integration**: Frame-level sparse routing directs features to specific experts to avoid gradient interference.
3. **Task-Specific Heads**: The SE head reconstructs enhanced spectrograms, and the SER head applies attentive statistics pooling for emotion classification.

---

## Getting Started

### 1. Requirements

The code is developed and tested with **Python 3.9**.

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
pip install huggingface_hub
```

### 2. Dataset Setup
This project uses the **MSP-Podcast** dataset. Due to licensing, you will need to request and download it yourself:
- **MSP-Podcast**: [Request Access Here](https://www.lab-msp.com/MSP/MSP-Podcast.html)

**Note:** You also need noise datasets for data augmentation (e.g., CRSS-4ENGLISH-14, Freesound, DNS Challenge) to replicate the noisy conditions discussed in the paper. We provide a script in `preprocess/mix_noise.py` to help mix clean audio with your downloaded noise datasets.

Update the dataset paths in the `config_cat.json` file:
```json
{
  "wav_dir": "path/to/your/MSP-PODCAST/Audios",
  "label_path": "path/to/your/MSP-PODCAST/Labels/labels_consensus.csv"
}
```

### 3. Pre-trained Weights and Models
We provide a Python script to automatically download the necessary pre-trained WavLM-Large checkpoint and other model weights from our Hugging Face repository.

Run the following command:
```bash
python download_weight.py
```
This script downloads weights into two directories according to our two-step training process:
- `pretrained_models/`: Contains the `WavLM-Large.pt` checkpoint and the pre-trained weights for the SER and SE heads (obtained from the first step of training where the SSL backbone is frozen).
- `model/`: Contains our final fully trained model (obtained from the second step of training where the entire model, including the SSL backbone, is fine-tuned).

### 4. Training

We employ a two-step training process:
1. **First Step**: Freeze the SSL model and fine-tune the SER and SE heads. (The checkpoints saved in `pretrained_models/` represent this stage).
2. **Second Step**: Fine-tune the entire model, including the SSL backbone. (The final output is saved in the `model/` directory).

To run the second stage of training (fine-tuning the whole framework), use the provided bash script:

```bash
bash train.sh
```

You can customize training arguments directly inside `train.sh` or when calling `train.py`, such as:
- `--ssl_type`: Background SSL model (default: `wavlm-large`)
- `--experts`: Number of experts in the MoE module (default: `3`)
- `--pooling_type`: E.g., `AttentiveStatisticsPooling`
- `--gate_type`: E.g., `Sparse_GatingNetwork` (Top-1 routing)
- `--batch_size`, `--epochs`, `--lr`, etc.

### 5. Evaluation

```bash
bash eval.sh
```

---

## Repository Structure

- `model/`: Directory to save models and training logs (TensorBoard).
- `net/`: Contains core PyTorch module definitions, including `MMoE`, routing networks, and the emotion regression head.
- `BSSE_SE/`: Contains architectures related to the SE task (e.g., BLSTM decoder) and WavLM implementations.
- `preprocess/`: Data preprocessing scripts (e.g., `mix_noise.py` to dynamically mix background noise).
- `utils/`: Dataloaders, metrics calculation, and other utilities.
- `train.py` / `train.sh`: Main entry points for training.
- `eval.py`: Evaluation script.
- `download_weight.py`: Script to download pre-trained checkpoints from Hugging Face.

---

## Acknowledgements

The speech enhancement (SE) module and related components are adapted from the [BSSE-SE repository](https://github.com/khhungg/BSSE-SE). We thank the authors for their open-source contribution.

---

## Citation

If you find our work or this repository useful, please consider citing our paper:
```bibtex
@ARTICLE{11499418,
  author={Tzeng, Jing-Tong and Busso, Carlos and Lee, Chi-Chun},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={Joint Learning using Mixture-of-Expert-Based Representation for Speech Enhancement and Robust Emotion Recognition}, 
  year={2026},
  volume={},
  number={},
  pages={1-13},
  keywords={Feeds;Digital audio broadcasting;Broadcasting;MIMICs;Filtering;Millimeter wave integrated circuits;Monolithic integrated circuits;Integrated circuits;Filters;Filter banks;Speech emotion recognition;speech enhancement;multi-task learning;mixture of experts;noise robustness},
  doi={10.1109/TASLPRO.2026.3688928}
}
```
