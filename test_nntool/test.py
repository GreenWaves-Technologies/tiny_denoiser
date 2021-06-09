import logging
import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse

# import denoiser
DENOISER_PATH = '/home/manuele/Work/denoiser'
sys.path.insert(0,DENOISER_PATH)
from denoiser.demucs_tf import DemucsTF
from denoiser import distrib, pretrained
from denoiser.data import NoisyCleanSet
from denoiser import distrib
from denoiser.evaluate import _run_metrics
from denoiser.tinylstm import TinyLSTMStreamer

#import nntool
sys.path.insert(0,'/home/manuele/Work/nntool')
from importer.importer import create_graph
from interpreter.nntool_shell import NNToolShell
from execution.graph_executer import GraphExecuter
from execution.quantization_mode import QuantizationMode


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
args.model_path = args.model_path_torch #bit ugly but works


# load the model
model = pretrained.get_model(args).to(args.device)

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
x = dataset[0]
noisy, clean = x[0], x[1]
sz = noisy.size()

# convert the model to the streaming form
model_streamer = TinyLSTMStreamer(model)


# load the graph in nntool
G = NNToolShell.get_graph_from_commands([
        'open '+args.onnx_file + ' --use_lut_sigmoid',
        'adjust',
        'fusions --scale8',
        'nodeoption LSTM_78 RNN_STATES_AS_INPUTS 1',
        'nodeoption LSTM_78 LSTM_OUTPUT_C_STATE 1',
        'nodeoption LSTM_144 RNN_STATES_AS_INPUTS 1',
        'nodeoption LSTM_144 LSTM_OUTPUT_C_STATE 1',
        'fquant',
        'qtune * float float16'
    ])
executer = GraphExecuter(G, qrecs=G.quantization)

RUN_ON_GAP = True
total_pesq = 0
total_stoi = 0
total_cnt = 0
updates = 5

args.pesq = True
for i, data in enumerate(data_loader):
    # Get batch data
    noisy, clean = [x.to(args.device) for x in data]
    # If device is CPU, we do parallel evaluation in each CPU worker.
    #pesq_i, stoi_i = _estimate_and_run_metrics(clean, model, noisy, args)
    with torch.no_grad():
        mags, DC_mags, phase =  model_streamer.get_spectrogram(noisy)
        len_seq = mags.size(2)
        if RUN_ON_GAP:
            lstm_0_i_state = np.zeros(256)
            lstm_0_c_state = np.zeros(256)
            lstm_1_i_state = np.zeros(256)
            lstm_1_c_state = np.zeros(256)
        else:
            nxt_lstm = None
        list_state = []
        mask_mags = torch.zeros_like(mags)
        for i in range(len_seq):
            single_mags = mags[:,:,i:i+1]
            if RUN_ON_GAP:
                single_mags = single_mags.numpy()
                data = [single_mags, lstm_0_i_state, lstm_0_c_state, lstm_1_i_state, lstm_1_c_state]
                outputs = executer.execute(data, qmode=QuantizationMode.all_float_quantize_dequantize(), silent=True)
                out = torch.Tensor(outputs[47][0]).unsqueeze(0)
                lstm_0_i_state = outputs[36][0].squeeze()
                lstm_0_c_state = outputs[39][0].squeeze()
                lstm_1_i_state = outputs[42][0].squeeze()
                lstm_1_c_state = outputs[48][0].squeeze()
                #output_mags_q[:,:,i:i+1] = out
                mask_mags[:,:,i:i+1] = torch.Tensor(outputs[0][0])
            else:
                out, nxt_lstm = model_streamer(single_mags,nxt_lstm =nxt_lstm ,ret_nxt_lstm = True)
                list_state.append(nxt_lstm)
            mask_mags[:,:,i:i+1] = out.unsqueeze(0)
        estimate = model_streamer.get_audio_from_mask(mask_mags, mags, DC_mags, phase)
        
        #estimate = model(noisy)
    sz0 = noisy.size(2)
    sz1 = estimate.size(2)
    print(sz0,sz1)
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