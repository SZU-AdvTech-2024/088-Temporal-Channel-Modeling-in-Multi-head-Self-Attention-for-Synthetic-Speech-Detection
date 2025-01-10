import os

import torch
import argparse

from data_utils import Dataset_eval
from eval import produce_evaluation_file
from model import Model
import librosa

from utils import read_metadata


def load_wav_file(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    wav = torch.tensor(wav)
    return wav
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    parser.add_argument('--threshold', type=float, default=-3.73, 
                    help='threshold score')
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--wav_path', type=str, 
                    help='path to the wav file')
    parser.add_argument('--ckpt_path', type=str, 
                    help='path to the model weigth')
    
    args = parser.parse_args()
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Loading model
    model = Model(args,device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded : {}'.format(args.ckpt_path))

    # Loading input
    print('Model loaded : {}'.format(args.wav_path))
    wav = load_wav_file(args.wav_path).to(device)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    
    # Running inference
    with torch.no_grad():
        out, _ = model(wav)
    score = out[:, 1].item()
    print('Is the wav file bonafide? -> {}'.format(score > args.threshold))

    tracks='LA'
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    args = parser.parse_args()

    if not os.path.exists('Scores/{}.txt'.format(tracks)):
        prefix      = 'ASVspoof_{}'.format(tracks)
        prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
        prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

        file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'LA/ASVspoof2021_LA_eval/{}.cm.eval.trl.txt'.format(prefix_2021)), is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'{}/ASVspoof2021_{}_eval/'.format(tracks,tracks)),track=tracks)
        produce_evaluation_file(eval_set, model, device, 'Scores/{}.txt'.format(tracks))
    else:
        print('Score file already exists')
