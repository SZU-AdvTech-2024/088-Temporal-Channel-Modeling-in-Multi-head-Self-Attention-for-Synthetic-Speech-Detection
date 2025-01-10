# Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection

## Pretrained Model
The pretrained model XLSR can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

## Setting up environment
Python version: 3.7.16

Install PyTorch
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Install other libraries:
```bash
pip install -r requirements.txt
```

## Training & Evaling
To train and produce the score, run:
```bash
python main.py --algo 5 --device your_device # For LA track evaluation
python main.py --algo 3 --device your_device # For DF track evaluation
```

## Evaling with a pretrained model
To eval a pretrained model, run:
```bash
python main.py --algo 3 --device your_device --ckpt_path your_model_path.pth
```

## Scoring
To get evaluation results of minimum t-DCF and EER (Equal Error Rate), follow these steps:
```bash
cd 2021/eval-package
python main.py --cm-score-file your_LA_score.txt --track LA --subset eval # For LA track evaluation
python main.py --cm-score-file your_DF_score.txt --track DF --subset eval # For DF track evaluation
```
