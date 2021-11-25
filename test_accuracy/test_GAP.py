import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import soundfile as sf
import librosa    
import shutil

from pesq import pesq
from pystoi import stoi


def run_on_gap_gvsoc(input_file, output_file, compile=True, gru=False, 
                quant_bfp16=False, approxRNN='', approxSigm='' ):
    runner_args = "" 
    runner_args += " GRU=1" if gru else "" 
    runner_args += " BF16=1" if gru else ""
    runner_args += " WAV_FILE="+input_file

    if approxRNN == 'LUT':
        runner_args += " ACCURATE_MATH_RNN=2"
    elif approxRNN == 'float':
        runner_args += " ACCURATE_MATH_RNN=1"

    if approxSigm == 'LUT':
        runner_args += " ACCURATE_MATH_SIG=2"
    elif approxSigm == 'float':
        runner_args += " ACCURATE_MATH_SIG=1"

    
    if compile:
        run_command = "make clean all run platform=gvsoc SILENT=1"+ runner_args
    else:
        run_command = "make run platform=gvsoc SILENT=1"+ runner_args
    print("Going to run: ", run_command)
    os.system(run_command)
    return True

def denoise_sample(input_file, output_file, samplerate, padding):

    if os.path.isfile(output_file):
        os.remove(output_file)

    data, s = librosa.load(input_file, sr=samplerate)
    if padding:
        data = np.pad(data, (padding, padding))

    file_name =  os.getcwd() + '/samples/test_py.wav'
    sf.write(file_name, data, samplerate)
    run_on_gap_gvsoc(file_name, output_file)
    shutil.copyfile('BUILD/GAP9_V2/GCC_RISCV_PULPOS/test_gap.wav', output_file)
    if not os.path.isfile(output_file):
        print("Error! not any output fiule produced")
        exit(0)
    print("Clean audio file stored in: ", output_file)
    return 0

def test_on_gap(    dataset_path, output_file, samplerate, padding, 
                    suffix_cleanfile, gru, quant_bfp16, approxRNN, approxSigm ):
    

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

            file_name =  os.getcwd() + '/samples/test_py.wav'
            sf.write(file_name, data, samplerate)

            run_on_gap_gvsoc(file_name, output_file, compile=compile_GAP, 
                gru=gru, quant_bfp16=quant_bfp16, approxRNN=approxRNN, approxSigm=approxSigm)
            compile_GAP = False
            shutil.copyfile('BUILD/GAP9_V2/GCC_RISCV_PULPOS/test_gap.wav', output_file)

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
        sz0 = clean_data.shape[0]
        sz1 = estimate.shape[0]
        if sz0 > sz1:
            estimate = np.pad(estimate, (0,sz0-sz1))
        else:
            estimate = estimate[:sz0]
   
        pesq_i, stoi_i =  _run_metrics(clean_data, estimate, samplerate)
        print("Sample ", i,'\t', file,"\twith pesq=\t", pesq_i, "\tand stoi=\t", stoi_i )
        total_cnt += 1
        total_pesq += pesq_i
        total_stoi += stoi_i

    pesq = total_pesq / total_cnt
    stoi = total_stoi / total_cnt
    print("Test set performance:PESQ=\t", pesq, "\t STOI=\t", stoi)


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
    parser.add_argument('--sample_rate', default=16000, type=int, help='sample rate')
    parser.add_argument("--mode", type=str, default="test",
                        help="Choose between sample | test")
    parser.add_argument("--wav_input", type=str, default="samples/p232_001.wav",
                        help="Path and filename of the input wav")
    parser.add_argument("--dataset_path", type=str, default="samples/dataset/",
                        help="Path of the dataset w/ subdirectories noisy and clean")
    parser.add_argument('--pad_input', type=int, default=0,
                        help="Pad the input left/right: computed as FRAME_SIZE - FRAME_HOP")
    parser.add_argument("--wav_output", type=str, default="test_gap.wav",
                        help="Path and filename of the output wav")
    parser.add_argument("--suffix_clean", type=str, default='',
                        help="Suffix of the clean smaples in test mode. If empy no clean sample is stored")
    parser.add_argument('--gru', action="store_true",
                            help="Set GRU in case of a GRU model")
    parser.add_argument('--bfp16', action="store_true",
                            help="Set quantization to BFP16")
    parser.add_argument("--approxRNN", type=str, default='',
                        help="Empty | LUT | float")
    parser.add_argument("--approxSigm", type=str, default='',
                        help="Empty | LUT | float")


    args = parser.parse_args()
    print(args)
    if args.mode == 'sample':
        print(args.pad_input)
        denoise_sample(args.wav_input, args.wav_output, args.sample_rate, args.pad_input)
    elif args.mode == 'test':
        test_on_gap(args.dataset_path, args.wav_output, args.sample_rate, args.pad_input, 
            args.suffix_clean, args.gru, args.bfp16, args.approxRNN, args.approxSigm)
    else:
        print("Selected --mode is not supported!")
        exit(1)
