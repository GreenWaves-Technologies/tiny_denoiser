import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import soundfile as sf
import librosa    

from pesq import pesq
from pystoi import stoi



def run_on_gap_gvsoc(input_file, output_file, compile=True):
    if compile:
        os.system("make clean all run platform=gvsoc SILENT=1")
    else:
        os.system("make run platform=gvsoc SILENT=1")
    return True

def denoise_sample(input_file, output_file, samplerate, padding):

    if os.path.isfile(output_file):
        os.remove(output_file)

    data, s = librosa.load(input_file, sr=samplerate)
    if padding:
        data = np.pad(data, (padding, padding))
    sf.write('samples/test_py.wav', data, samplerate)
    
    run_on_gap_gvsoc('samples/test_py.wav', output_file)

    if not os.path.isfile(output_file):
        print("Error! not any output fiule produced")
        exit(0)
    print("Clean audio file stored in: ", output_file)
    return 0

def test_on_gap(dataset_path, output_file, samplerate, padding, suffix_cleanfile):
    
    # set noisy and clean path
    noisy_path = dataset_path + '/noisy/'
    clean_path = dataset_path + '/clean/'

    # parse the dataset
    filenames = [os.path.splitext(item)[0] for item in os.listdir(noisy_path)]
    if len(filenames) == 0:
        print("Dataset is empty!")
        exit(1)

    # create a folder with the estimate
    estimate_path = dataset_path + '/estimate/'
    if suffix_cleanfile != '':
        if not os.path.exists(estimate_path):
            os.makedirs(estimate_path)

    # stas
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0

    # utils
    compile_GAP = True

    for i, file in enumerate(filenames):

        # check first if the clean signal has been already produced
        estimate_filepath = estimate_path + file + suffix_cleanfile + '.wav'
        if os.path.isfile(estimate_filepath):
            estimate, s = librosa.load(estimate_filepath, sr=samplerate)
        else: # compute the estimate
        
            # Get data
            if os.path.isfile(output_file):
                os.remove(output_file)

            input_file = noisy_path + file + '.wav'
            data, s = librosa.load(input_file, sr=samplerate)
            if padding:
                data = np.pad(data, (padding, padding))
            sf.write('samples/test_py.wav', data, samplerate)
    
            run_on_gap_gvsoc('samples/test_py.wav', output_file, compile=compile_GAP)
            compile_GAP = False

            if not os.path.isfile(output_file):
                print("Error! not any output file produced")
                exit(0)

            estimate, s = librosa.load(output_file, sr=samplerate)
            estimate = estimate[300:]

            if suffix_cleanfile != '':
                sf.write(estimate_filepath, estimate, samplerate)

        # get the clean file
        input_file = clean_path + file + '.wav'
        clean_data, s = librosa.load(input_file, sr=samplerate)

        # compute the metrics
        print(clean_data.shape, estimate.shape)
        sz0 = clean_data.shape[0]
        sz1 = estimate.shape[0]
        print(sz0, sz1)
        if sz0 > sz1:
            estimate = np.pad(estimate, (0,sz0-sz1))
        else:
            estimate = estimate[:sz0]
   
        pesq_i, stoi_i =  _run_metrics(clean_data, estimate, samplerate)
        total_cnt += clean_data.shape[0]
        total_pesq += pesq_i
        total_stoi += stoi_i

    pesq = total_pesq / total_cnt
    stoi = total_stoi / total_cnt
    print(f'Test set performance:PESQ={pesq}, STOI={stoi}.')


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

def _run_metrics(clean, estimate, samplerate):
    pesq_i = pesq(samplerate, clean, estimate, 'wb')
    stoi_i = stoi(clean, estimate, samplerate, extended=False)
    return pesq_i, stoi_i

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'GAP denoiser',
        description="Speech enhancement using TinyLSTM on GAP")
    parser.add_argument('--sample_rate', default=16_000, type=int, help='sample rate')
    parser.add_argument("--mode", type=str, default="test",
                        help="Choose between sample | test")
    parser.add_argument("--wav_input", type=str, default="samples/p232_001.wav",
                        help="Path and filename of the input wav")
    parser.add_argument("--dataset_path", type=str, default="samples/dataset/",
                        help="Path of the dataset w/ subdirectories noisy and clean")
    parser.add_argument('--pad_input', type=int, default=0,
                        help="Pad the input left/right: computed as FRAME_SIZE - FRAME_HOP")
    parser.add_argument("--wav_output", type=str, default="/home/manuele/GWT_apps/denoiser_tiny/BUILD/GAP9_V2/GCC_RISCV/test_gap.wav",
                        help="Path and filename of the output wav")
    parser.add_argument("--suffix_clean", type=str, default='_f16',
                        help="Suffix of the clean smaples in test mode. If empy no clean sample is stored")

    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only input signal, 1 only denoised.')
    parser.add_argument('--streaming', action="store_true",
                            help="true streaming evaluation for Demucs")

    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    args = parser.parse_args()
    print(args)
    if args.mode == 'sample':
        print(args.pad_input)
        denoise_sample(args.wav_input, args.wav_output, args.sample_rate, args.pad_input)
    elif args.mode == 'test':
        test_on_gap(args.dataset_path, args.wav_output, args.sample_rate, args.pad_input, args.suffix_clean)
    else:
        print("Selected --mode is not supported!")
        exit(1)