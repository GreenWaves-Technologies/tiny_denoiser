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

# nntool
sys.path.insert(0, os.environ['NNTOOL_DIR'])
from execution.graph_executer import GraphExecuter
from execution.quantization_mode import QuantizationMode
from interpreter.nntool_shell import NNToolShell
from quantization.quantizer.new_quantizer import NewQuantizer

def run_on_gap_gvsoc(input_file, output_file, compile=True, gru=False, 
                quant_bfp16=False,  quant_int8=False, approx='' ):
    runner_args = "" 
    runner_args += " GRU=1" if gru else "" 
    runner_args += " WAV_FILE="+input_file
    runner_args += " QUANT_BITS=BFP16" if quant_bfp16 else " QUANT_BITS=8" if quant_int8 else ""

#    if approxRNN == 'LUT':
#        runner_args += " ACCURATE_MATH_RNN=2"
#    elif approxRNN == 'float':
#        runner_args += " ACCURATE_MATH_RNN=1"
#
#    if approxSigm == 'LUT':
#        runner_args += " ACCURATE_MATH_SIG=2"
#    elif approxSigm == 'float':
#        runner_args += " ACCURATE_MATH_SIG=1"

    if approx == 'LUT':
        runner_args += " APPROX_LUT=1"
    
    if compile:
        run_command = "make all run platform=gvsoc SILENT=1"+ runner_args
    else:
        run_command = "make run platform=gvsoc SILENT=1"+ runner_args
    print("Going to run: ", run_command)
    os.system(run_command)
    return True

def denoise_sample_on_gap_gvsoc(input_file, output_file, samplerate, padding = False, compile_GAP=True, 
                    gru=False, quant_bfp16=False, quant_int8=False, approx=''):

    if os.path.isfile(output_file):
        os.remove(output_file)

    data, s = librosa.load(input_file, sr=samplerate)
    if padding:
        data = np.pad(data, (padding, padding))

    file_name =  os.getcwd() + '/samples/test_py.wav'
    sf.write(file_name, data, samplerate)
    run_on_gap_gvsoc(file_name, output_file, compile=compile_GAP, 
                    gru=gru, quant_bfp16=quant_bfp16, quant_int8=quant_int8, approx=approx)
    shutil.copyfile('BUILD/GAP9_V2/GCC_RISCV_PULPOS/test_gap.wav', output_file)
    if not os.path.isfile(output_file):
        print("Error! not any output fiule produced")
        exit(0)
    print("Clean audio file stored in: ", output_file)
    return 0

def test_on_gap(    dataset_path, output_file, samplerate, padding, 
                    suffix_cleanfile, gru, quant_bfp16, quant_int8, quant_ne16, nntool, approx  ):
    
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

    # prepare nntool graph
    if nntool:
        model_name = 'model/denoiser_GRU.onnx' if gru else 'model/denoiser.onnx'
        G = NNToolShell.get_graph_from_commands([
            'open ' + model_name + ' --use_lut_sigmoid --use_lut_tanh',
            'adjust',
            'fusions --scale8',
            'nodeoption GRU_74 RNN_STATES_AS_INPUTS 1',
            'nodeoption GRU_136 RNN_STATES_AS_INPUTS 1',

        ])

        if quant_ne16 or quant_int8:

            # quantized inference
            nntool_quantizer = True

            # change this with other script...
            import pickle
            fp = open('model/data_quant_gru.json', 'rb')
            astats = pickle.load(fp)
            fp.close()
            
            Opts = { 'float_type': 'float32', 'kernel_type': 'fastfloat', 'hwc': False, 
                     'sq_bits': 8, 
                     'weight_bits': 8, 
                     'force_external_size': 16, # 8
                     'narrow_weights': True, 
                     'use_ne16': True, # False
                     'narrow_state': True, 
                     'quantized_dimension': 'channel', 
                     'force_ne16': False, 
                     'allow_asymmetric': False, 
                     'force_input_size': 16, # 8 
                     'force_output_size': 16, #8 
                     'softmax_out_8bits': False, 
                     'bits': 16, 
                     'pow2_biases': 0 
                    }
            if quant_int8: 
                Opts['force_external_size'] = 8
                Opts['use_ne16'] = False
                Opts['force_input_size'] = 8
                Opts['force_output_size'] = 8



            quantizer = NewQuantizer(G, reset_all=True)
            quantizer.options = Opts
            quantizer.schemes.append('SQ8')
            quantizer.set_stats(astats)
            quantizer.quantize()
            G.add_dimensions()
            
            if quant_ne16: # adjust after NE16 = True
                NNToolShell.run_commands_on_graph(G, ['adjust', 'fusions --scale8'])

            NNToolShell.run_commands_on_graph(G, [ 'qshow'])
            print("The graph is QUANTIZED")


        else: # fp16
            NNToolShell.run_commands_on_graph(G, [ 
                'fquant',
                'qtune --step * scheme=float float_type=float16', 
                'qshow'])



        # define the executed
        executer = GraphExecuter(G, qrecs=G.quantization)


    for i, file in enumerate(filenames):

        # check first if the clean signal has been already produced
        estimate_filepath = estimate_path + file + suffix_cleanfile + '.wav'
        if os.path.isfile(estimate_filepath):
            estimate, s = librosa.load(estimate_filepath, sr=samplerate)
        else: # compute the estimate
        
            if nntool:
                # Get data
                if os.path.isfile(output_file):
                    os.remove(output_file)

                input_file = noisy_path + file + '.wav'    
    #            input_file = '/home/manuele/GWT_apps/denoiser/samples/quant/p286_035.wav'        
                data, s = librosa.load(input_file, sr=samplerate)

                print(input_file)

                if padding:
                    data = np.pad(data, (padding, padding))
                    print(data)
            
                win_len = 400
                win_inc = 100 
                fft_len = 512
                print('data input shape:', data.shape)

                stft_frame_i = librosa.stft(
                    data, win_length=win_len, 
                    n_fft=fft_len, hop_length=win_inc,
                    window='hann', center=False)
                fft_feat, num_win = stft_frame_i.shape
                print('stft input shape:', stft_frame_i.shape)


                stft_frame_i_T = np.transpose (stft_frame_i) # swap the axis to select the tmestamp
                stft_frame_o_T = np.empty_like(stft_frame_i_T)

                rnn_0_i_state = np.zeros(256)
                #rnn_0_c_state = np.zeros(256)
                rnn_1_i_state = np.zeros(256)
                #rnn_1_c_state = np.zeros(256)
                

                for i in range (num_win):
                    print('*****Frame ' + str(i) + ' ******')
                    stft_clip = stft_frame_i_T[i]
                    stft_clip_mag = np.abs(stft_clip)
#                    print(stft_clip_mag)
                    #stft_clip_mag_estimate = np.random.rand(fft_feat)
                    data = [stft_clip_mag, rnn_0_i_state, rnn_1_i_state]
                    outputs = executer.execute(data, 
                        qmode=QuantizationMode.all_dequantize() if nntool_quantizer else None, 
                        silent=True)
                    if nntool_quantizer:
                        n_id = [4, 32, 36, 40] if quant_ne16 else [4, 31, 34, 39]
                    else:
                        n_id = [4, 31, 34, 39]
                    
                    conv_0_out = outputs[n_id[0]][0]
                    rnn_0_i_state = outputs[n_id[1]][0]
                    rnn_1_i_state = outputs[n_id[2]][0]
                    mag_out = outputs[n_id[3]][0]

#                    print('conv_0_out= ', conv_0_out.squeeze())
#                    print('rnn_0_i_state= ', rnn_0_i_state.squeeze())
#                    print('rnn_1_i_state= ', rnn_1_i_state.squeeze())
#                    print('mag_out= ', mag_out.squeeze())
#
#                ## this is for print values ####
#                    v = ''
#                    for item in rnn_1_i_state.squeeze():
#                        v += str(item) + ', '
#                    print(v)
#                ################################



                    stft_clip_mag_estimate = mag_out.squeeze()
                    #stft_clip_mag_estimate = np.ones(257)

                    stft_clip = stft_clip * stft_clip_mag_estimate
                    stft_frame_o_T[i] = stft_clip

#                    if i>-1:
#                        exit(0)


                stft_frame_o = np.transpose (stft_frame_o_T)
                print('stft output shape:', stft_frame_o.shape)

       
                data = librosa.istft(stft_frame_o, hop_length=win_inc, 
                    win_length=win_len, window='hann', center=False )
                
                print('data output shape:', data.shape)

                #print(np.mean((outputs1 - np.abs(librosa_stft))**2))
                
                estimate = data[padding:] # place holder
            else:

                input_file = noisy_path + file + '.wav'    

                denoise_sample_on_gap_gvsoc(
                    input_file, output_file, samplerate, 
                    padding = padding, compile=compile_GAP, 
                    gru=gru, quant_bfp16=quant_bfp16, 
                    quant_int8=quant_int8, approx=approx
                )

                compile_GAP = False
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
    parser.add_argument("--quant", type=str, default="fp16",
                        help="fp16 | bfp16 | int8 | ne16")
#    parser.add_argument('--bfp16', action="store_true",
#                            help="Set quantization to BFP16")
#    parser.add_argument('--int8', action="store_true",
#                            help="Set quantization to 8 bit ") 
    parser.add_argument('--nntool', action="store_true",
                            help="Run inference on nntool. if False, run inference on GVSOC")
    parser.add_argument("--approx", type=str, default='',
                        help="Empty | LUT")

    args = parser.parse_args()

    # parse the quantization method
    bfp16 = int8 = ne16 = False
    if args.quant == 'bfp16':
        bfp16 = True
    elif args.quant == 'int8':
        int8 = True
    elif args.quant == 'ne16':
        ne16 = True

    # call the test
    if args.mode == 'sample':
        print(args.pad_input)
        denoise_sample_on_gap_gvsoc(args.wav_input, args.wav_output, args.sample_rate, args.pad_input)
    elif args.mode == 'test':
        test_on_gap(args.dataset_path, args.wav_output, args.sample_rate, args.pad_input, 
            args.suffix_clean, args.gru, bfp16, int8, ne16, args.nntool, args.approx)
    else:
        print("Selected --mode is not supported!")
        exit(1)
