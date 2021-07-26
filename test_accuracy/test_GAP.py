import logging
import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import soundfile as sf
import librosa    


# import denoiser
DENOISER_PATH = '/home/manuele/Work/denoiser'
sys.path.insert(0,DENOISER_PATH)
from denoiser.demucs_tf import DemucsTF
from denoiser import distrib, pretrained
from denoiser.data import NoisyCleanSet
from denoiser import distrib
from denoiser.evaluate import _run_metrics
from denoiser.tinylstm import TinyLSTMStreamer


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
pretrained.add_model_flags(parser)
parser.add_argument('--device', default="cpu")
parser.add_argument('--dry', type=float, default=0,
                    help='dry/wet knob coefficient. 0 is only input signal, 1 only denoised.')
parser.add_argument('--sample_rate', default=16_000, type=int, help='sample rate')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")

# becnhmark vars
group.add_argument("--model_path_torch", type=str, 
                default=DENOISER_PATH + '/outputs/exp_bandmask=0.2,batch_size=64,dset=valentini,model=tinylstm,remix=1,segment=4.5,shift=8000,shift_same=True,stft_loss=True,stride=0.5,tinylstm.encoder=True/best.th',
                help="directory including noisy wav files")
group.add_argument('--use_dns', action="store_true",
                        help="Use the valentini dataset")
group.add_argument("--dataset_path", type=str, 
                default='/home/manuele/Work/denoiser/egs/',
                help="dataset path")
group.add_argument("--onnx_file", type=str, 
                default='/home/manuele/Work/denoiser/test.onnx',
                help="dataset path")

args = parser.parse_args([])


# load dataset
if args.use_dns:
    # DNS Dataset
    argsdata_dir=args.dataset_path + 'dns/tt'
    argsmatching='dns'
    argssample_rate=16000   
else:
    # VALENTINI Dataset
    argsdata_dir=args.dataset_path + 'valentini/tt'
    argsmatching='sort'
    argssample_rate=16000
 
    
dataset = NoisyCleanSet(argsdata_dir, matching=argsmatching, sample_rate=argssample_rate)
data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)


RUN_ON_GAP = True
total_pesq = 0
total_stoi = 0
total_cnt = 0
updates = 5

args.pesq = True
for i, data in enumerate(data_loader):
    # Get batch data
    noisy, clean = [x.to(args.device) for x in data]


    print(noisy.size())
    noisy = F.pad(noisy, (300,300) ,"constant", 0 )
    print(noisy.size())
    sf.write('samples/input_file.wav', noisy.squeeze(), argssample_rate)
    os.system("make run platform=gvsoc SILENT=1")
    estimate, s = librosa.load('samples/test_gap.wav', sr=argssample_rate)
    estimate = torch.from_numpy(estimate)[300:].unsqueeze(0).unsqueeze(0)
    print('estimate size: ', estimate.size())
    
    sz0 = clean.size(2)
    sz1 = estimate.size(2)
    print(sz0, sz1)
    if sz0 > sz1:
        estimate = F.pad(estimate, (0,sz0-sz1),"constant", 0 )
    else:
        estimate = estimate[...,:sz0]
    
    pesq_i, stoi_i =  _run_metrics(clean, estimate, args)
    total_cnt += clean.shape[0]
    total_pesq += pesq_i
    total_stoi += stoi_i
    
metrics = [total_pesq, total_stoi]
pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
print(f'Test set performance:PESQ={pesq}, STOI={stoi}.')
