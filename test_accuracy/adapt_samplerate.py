import logging
import os
import sys
import numpy as np
import argparse

import soundfile as sf
import librosa    


parser = argparse.ArgumentParser( 'gap9.speech_denoiser',
        description="Python Script to run demo on GAP9")
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

args = parser.parse_args([])




samplerate = 16000
data, s = librosa.load('../samples/p232_001.wav', sr=samplerate)


sf.write('test_out.wav', data, samplerate)
#wavfile.write('test_out.wav',samplerate,data)
