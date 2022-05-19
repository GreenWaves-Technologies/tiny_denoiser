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

from threading import Thread

def run_on_gap_gvsoc(input_file, output_file, compile=True, gru=False, 
                quant_bfp16=False,  quant_int8=False, approx='' ):
    runner_args = "" 
    runner_args += " GRU=1" if gru else "" 
    runner_args += " WAV_FILE="+input_file
    runner_args += " QUANT_BITS=BFP16" if quant_bfp16 else " QUANT_BITS=8" if quant_int8 else ""

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
    shutil.copyfile('BUILD/GAP9_V2/GCC_RISCV_FREERTOS/test_gap.wav', output_file)
    if not os.path.isfile(output_file):
        print("Error! not any output fiule produced")
        exit(0)
    print("Clean audio file stored in: ", output_file)
    return 0

def nntool_get_model(model_onnx, gru, real, quant_fp16,  quant_bfp16, quant_int8, 
                    quant_ne16, ne_16_type, quant_stats_file=None, clip_type=None, 
                    max_rnn=False, linear_fp16=False):
    # nntool
    sys.path.insert(0, os.environ['NNTOOL_DIR'])
    from nntool.api import NNGraph

    model = NNGraph.load_graph(
        model_onnx,
        load_quantization=False, # Whether tflite quant should be loaded or not (default: False)
        use_hard_sigmoid=False,
        use_hard_tanh=False
    )
    
    model.adjust_order()
    model.fusions('scaled_match_group')
    
    if gru:
        model["GRU_74"].set_states_as_inputs(model)
        model["GRU_136"].set_states_as_inputs(model)
    else:
        model["LSTM_78"].set_states_as_inputs(model)
        model["LSTM_78"].set_c_state_as_output(model)
        model["LSTM_144"].set_states_as_inputs(model)
        model["LSTM_144"].set_c_state_as_output(model)
    
    astats = None
    if quant_ne16 or quant_int8:
        print('Loading parameters from: ', quant_stats_file)
        import pickle
        fp = open(quant_stats_file, 'rb')
        astats = pickle.load(fp)
        fp.close()
        
        graph_options ={}
        node_options = {}
        if quant_int8: 
            scheme = ['scaled']
            graph_options = {
                "use_ne16": False,
            }
            
            if clip_type:
                graph_options["clip_type"] = clip_type
            if max_rnn:
                if gru:
                    node_options["h_state_GRU_74"] = {"clip_type": "none" }
                    node_options["h_state_GRU_136"] = {"clip_type": "none" }
                    
                    node_options["GRU_74"] = {"clip_type": "none" }
                    node_options["GRU_136"] = {"clip_type": "none" }
                else:
                    node_options["i_state_LSTM_78"] = {"clip_type": "none" }
                    node_options["i_state_LSTM_144"] = {"clip_type": "none" }
                    node_options["c_state_LSTM_78"] = {"clip_type": "none" }
                    node_options["c_state_LSTM_144"] = {"clip_type": "none" }
                    
                    node_options["LSTM_78"] = {"clip_type": "none" }
                    node_options["LSTM_144"] = {"clip_type": "none" }


            if linear_fp16:
                if gru:
                    node_options["input_1"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_reshape_in"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_139_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_142_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_142_reshape_out"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Sigmoid_143"] = {"scheme": "float", "float_type": "float16"}
                    node_options["output_1"] = {"scheme": "float", "float_type": "float16"}
                else:
                    node_options["input_1"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_reshape_in"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_144_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_147_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_147_reshape_out"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Sigmoid_148"] = {"scheme": "float", "float_type": "float16"}
                    node_options["output_1"] = {"scheme": "float", "float_type": "float16"}
                    
        else: #quant_ne16: 
            scheme = ['scaled']
            node_options = {}
            graph_options = {
                "use_ne16": True,
            }
            if ne_16_type == 'a8w8':
                graph_options['force_external_size'] = 8
                graph_options['force_input_size'] = 8
                graph_options['force_output_size'] = 8
            elif ne_16_type == 'a16w8':
                graph_options['force_external_size'] = 16
                graph_options['force_input_size'] = 16
                graph_options['force_output_size'] = 16 

            else:
                print('going to quntize this way')
                pass
      
            if clip_type:
                graph_options["clip_type"] = clip_type
            if max_rnn:
                if gru:
                    node_options["h_state_GRU_74"] = {"clip_type": "none" }
                    node_options["h_state_GRU_136"] = {"clip_type": "none" }
                    
                    node_options["GRU_74"] = {"clip_type": "none" }
                    node_options["GRU_136"] = {"clip_type": "none" }
                else:
                    node_options["i_state_LSTM_78"] = {"clip_type": "none" }
                    node_options["i_state_LSTM_144"] = {"clip_type": "none" }
                    node_options["c_state_LSTM_78"] = {"clip_type": "none" }
                    node_options["c_state_LSTM_144"] = {"clip_type": "none" }
                    
                    node_options["LSTM_78"] = {"clip_type": "none" }
                    node_options["LSTM_144"] = {"clip_type": "none" }


            if linear_fp16:
                if gru:
                    node_options["input_1"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_reshape_in"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_139_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_142_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_142_reshape_out"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Sigmoid_143"] = {"scheme": "float", "float_type": "float16"}
                    node_options["output_1"] = {"scheme": "float", "float_type": "float16"}


                else:
                    node_options["input_1"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_reshape_in"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_0_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_147_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_150_fusion"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Conv_150_reshape_out"] = {"scheme": "float", "float_type": "float16"}
                    node_options["Sigmoid_151"] = {"scheme": "float", "float_type": "float16"}
                    node_options["output_1"] = {"scheme": "float", "float_type": "float16"}
            
            # adjust after NE16 = True
            #NNToolShell.run_commands_on_graph(G, ['adjust', 'fusions --scale8'])
            model.adjust_order()
            model.fusions('scaled_match_group')

            if ne_16_type == 'a16arnn8w8':
                if gru:
                    NNToolShell.run_commands_on_graph(G, 
                        ['qtune --step GRU_74,GRU_136 force_external_size=8'])
                else:
                    NNToolShell.run_commands_on_graph(G, 
                        ['qtune --step LSTM_78,LSTM_144 force_external_size=8'])

    elif quant_fp16: # fp16
        
        scheme = ['float']
        graph_options = {
            "scheme": 'float',
            "float_type" : 'float16'
        }
        node_options = {}

        
    print(model.show())
    if not real:
        model.quantize(
            astats,
            schemes=scheme, # Schemes present in the graph
            graph_options = graph_options,
            node_options=node_options,
        )
        
        print(model.qshow())
    

    return model


def nntool_inf(nntool_model, filenames, noisy_path, clean_path, estimate_path, results, thread_id, samplerate, padding, gru, h_state_len, dry=0.0):
    from nntool.api.utils import qsnrs

    metric=[]
    suffix_cleanfile = ''
    
    for i, file in enumerate(filenames):
        
        print('Thread ', thread_id, ':',i, file )

        # check first if the clean signal has been already produced
        estimate_filepath = estimate_path + file + suffix_cleanfile + '.wav'
        if os.path.isfile(estimate_filepath):
            estimate, s = librosa.load(estimate_filepath, sr=samplerate)
        else: # compute the estimate
            output_file = 'test_gap'+str(thread_id)+'.wav'
            if nntool_model is not False:
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

                rnn_0_i_state = np.zeros(h_state_len)
                rnn_1_i_state = np.zeros(h_state_len)

                if gru == 0:
                    rnn_0_c_state = np.zeros(h_state_len)
                    rnn_1_c_state = np.zeros(h_state_len)


                for i in range (num_win):
        #                    print('*****Frame ' + str(i) + ' ******')
                    stft_clip = stft_frame_i_T[i]
                    stft_clip_mag = np.abs(stft_clip)
        #                    print(stft_clip_mag)

                    if gru == 1:
                        data = [stft_clip_mag, rnn_0_i_state, rnn_1_i_state]
                    else:
                        data = [stft_clip_mag, rnn_0_i_state, rnn_0_c_state, rnn_1_i_state, rnn_1_c_state]
                        #data = [rnn_1_c_state, rnn_0_c_state, rnn_1_i_state, rnn_0_i_state, stft_clip_mag]

                    if real:
                        outputs = nntool_model.execute(data)
                    else:
                        outputs = nntool_model.execute(data, quantize=True, dequantize=True)
#                    outputs_real = nntool_model.execute(data)
#                    qsnr = qsnrs(outputs, outputs_real)
#                    print(qsnr)
                    
                    #for i,item in enumerate(qsnr):
                    #    print('Layer ',i, qsnr[i], np.array(outputs[i]).shape)
                    #    print('Quant')
                    #    print(np.squeeze(outputs[i])) 
                    #    print('Real')
                    #    print(np.squeeze(outputs_real[i]))                         
                    
                    
#                    exit(0)
                    
                    #LAYER = 4
                    #print('Quant')
                    #print(np.squeeze(outputs[3])) 
                    #print('Real')
                    #print(np.squeeze(outputs_real[3])) 
                    
                    

                    mag_out = outputs[nntool_model['output_1'].step_idx][0]

                    if gru == 1:
                        rnn_0_i_state = outputs[nntool_model['GRU_74'].step_idx][0]
                        rnn_1_i_state = outputs[nntool_model['GRU_136'].step_idx][0]
                    else:
                        rnn_0_i_state = outputs[nntool_model['LSTM_78'].step_idx][0]
                        rnn_0_c_state = outputs[nntool_model['output_2'].step_idx][0]
                        rnn_1_i_state = outputs[nntool_model['LSTM_144'].step_idx][0]
                        rnn_1_c_state = outputs[nntool_model['output_3'].step_idx][0]


                    stft_clip_mag_estimate = mag_out.squeeze()

                    stft_clip = stft_clip * stft_clip_mag_estimate
                    stft_frame_o_T[i] = stft_clip


                stft_frame_o = np.transpose (stft_frame_o_T)
                print('stft output shape:', stft_frame_o.shape)

                data = librosa.istft(stft_frame_o, hop_length=win_inc, 
                    win_length=win_len, window='hann', center=False )

                print('data output shape:', data.shape)


                estimate = data[padding:] # place holder
            else:

                input_file = noisy_path + file + '.wav'    

                denoise_sample_on_gap_gvsoc(
                    input_file, output_file, samplerate, 
                    padding = padding, compile_GAP=compile_GAP, 
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

        # cut signals
        sz0 = clean_data.shape[0]
        sz1 = estimate.shape[0]
        if sz0 > sz1:
            estimate = np.pad(estimate, (0,sz0-sz1))
        else:
            estimate = estimate[:sz0]

        # dry avg
        if dry > 0.0:
            input_file = noisy_path + file + '.wav'
            noisy_data, s = librosa.load(input_file, sr=samplerate)
            estimate = dry*noisy_data + (1-dry)*estimate

        # compute the metrics
        pesq_i, stoi_i =  _run_metrics(clean_data, estimate, samplerate)
        print("Sample ", i,'\t', file,"\twith pesq=\t", pesq_i, "\tand stoi=\t", stoi_i )

        metric.append([pesq_i,stoi_i])
        
    results[thread_id] = metric


def test_on_dset(   noisy_path, clean_path, n_threads, output_file, samplerate, padding, 
                    suffix_cleanfile, gru, real, quant_fp16,  quant_bfp16, quant_int8, 
                    quant_ne16, ne_16_type, nntool_model, approx, h_state_len=256, dry=0.0  ):
    
    # set noisy and clean path
    #noisy_path = dataset_path + '/noisy/'
    #clean_path = dataset_path + '/clean/'

    # parse the dataset
    filenames = [os.path.splitext(item)[0] for item in os.listdir(noisy_path)]
    if len(filenames) == 0:
        print("Dataset is empty!")
        exit(1)

    # create a folder with the estimate
    estimate_path = '/home0/manuele/Work/audio/GAP_projects/denoiser/samples/dataset/estimate/'
    if suffix_cleanfile != '':
        if not os.path.exists(estimate_path):
            os.makedirs(estimate_path)

    # stas
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0

    # utils
    compile_GAP = True
    
    # Fork ot each thread the computation of part of input_
    n_threads = n_threads
    batch_size = len(filenames)
    chunk_size = int( batch_size / n_threads )
    print('Numbers of file is: ', len(filenames), ' and chuck size: ', chunk_size)    
    
    results = [0 for x in range(n_threads)]
    threads = [0 for x in range(n_threads)]

    for thread_id in range(n_threads):
        first = thread_id * chunk_size
        last = min(first + chunk_size, batch_size)
        if (thread_id == n_threads-1):
            last = batch_size
        idxs = list(range(first, last))
        filenames_th = [filenames[x] for x in range(first, last)]
        print(filenames_th)
        threads[thread_id] = Thread(target=nntool_inf, args=(nntool_model, filenames_th, noisy_path, clean_path, estimate_path, results, thread_id, samplerate, padding, gru, h_state_len, dry))
        threads[thread_id].start()


    # Wait all the threads to complete and join the results
    pesq_i = 0
    stoi_i = 0
    count = 0
    for thread_id in range(n_threads):
        threads[thread_id].join()
        for item in results[thread_id]:
            print(item)
            a, b = item
            print('Thread ', a, b)
            pesq_i += item[0]
            stoi_i += item[1]
            
        count += len(results[thread_id])
        print('Thread {} returned {} results'.format(thread_id, len(results[thread_id])) )

    pesq = pesq_i / count
    stoi = stoi_i / count
    print("Test set performance:PESQ=\t", pesq, "\t STOI=\t", stoi, '\t over', count, 'samples')


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
    parser.add_argument("--noisy_dataset_path", type=str, default="samples/dataset/noisy/",
                        help="Path of the dataset w/ subdirectories noisy")
    parser.add_argument("--clean_dataset_path", type=str, default="samples/dataset/clean/",
                        help="Path of the dataset w/ subdirectories clean")
    parser.add_argument('--pad_input', type=int, default=0,
                        help="Pad the input left/right: computed as FRAME_SIZE - FRAME_HOP")
    parser.add_argument("--wav_output", type=str, default="test_gap.wav",
                        help="Path and filename of the output wav")
    parser.add_argument("--suffix_clean", type=str, default='',
                        help="Suffix of the clean smaples in test mode. If empy no clean sample is stored")
    parser.add_argument('--gru', action="store_true",
                            help="Set GRU in case of a GRU model")
    parser.add_argument("--quant", type=str, default="fp16",
                        help="fp16 | bfp16 | int8 | ne16 | real")
    parser.add_argument('--ne_16_type', type=str, default="a8w8",
                        help="a16w8 | a8w8 | a16arnn8w8")
    parser.add_argument('--nntool', action="store_true",
                            help="Run inference on nntool. if False, run inference on GVSOC")
    parser.add_argument("--approx", type=str, default='',
                        help="Empty | LUT")
    
    parser.add_argument("--model_onnx", type=str, default="model/denoiser.onnx",
                        help="Path to the onnx model")
    parser.add_argument("--quant_stats_file", type=str, default="BUILD_MODEL_8BIT/data_quant.json",
                        help="Path to the quant stats file")
    parser.add_argument('--n_threads', type=int, default=1,
                        help="Number of threads for nntool inference")
    parser.add_argument("--clip_type", type=str, default=None,
                        help="can be std3")    
    parser.add_argument("--max_rnn", action="store_true",
                        help="force rnn quantization to be max")    
    parser.add_argument("--linear_fp16", action="store_true",
                        help="force linear layer to be fp16")
    parser.add_argument('--h_state_len', type=int, default=256,
                        help="Number of states of the rnn hidden layers")    
    parser.add_argument('--dry', type=float, default=0.0,
                        help="Setting the dry parameter")  
    
    args = parser.parse_args()
    
    print( args)
    for arg in vars(args):
        print (arg, '\t\t',getattr(args, arg))

    # parse the quantization method
    real = fp16 = bfp16 = int8 = ne16 = False
    ne_16_type = False
    if args.quant == 'real':
        real = True
    elif args.quant == 'fp16':
        fp16 = True
    elif args.quant == 'bfp16':
        bfp16 = True
    elif args.quant == 'int8':
        int8 = True
    elif args.quant == 'ne16':
        ne16 = True
        ne_16_type = 'a16w8'
        if args.ne_16_type == 'a16w8':  
            ne_16_type = 'a16w8'
        elif args.ne_16_type == 'a8w8':
            ne_16_type = 'a8w8'
        elif args.ne_16_type == 'a16arnn8w8':
            ne_16_type = 'a16arnn8w8'
    
    # prepare nntool executer if needed
    if args.nntool:
        print('Going to setup the nntool executer form model {}'.format(args.model_onnx) )
        nntool_model = nntool_get_model(args.model_onnx, args.gru, real, fp16, bfp16, int8, ne16, ne_16_type, args.quant_stats_file,
                                       clip_type=args.clip_type, max_rnn=args.max_rnn, linear_fp16=args.linear_fp16)
    else:
        nntool_model = False
    
    # call the test
    if args.mode == 'sample':
        print(args.pad_input)
        denoise_sample_on_gap_gvsoc(args.wav_input, args.wav_output, args.sample_rate, args.pad_input)
    elif args.mode == 'test':
        test_on_dset(args.noisy_dataset_path,args.clean_dataset_path, args.n_threads, args.wav_output, args.sample_rate, args.pad_input, 
            args.suffix_clean, args.gru, real, fp16, bfp16, int8, ne16, ne_16_type, nntool_model, args.approx, h_state_len=args.h_state_len, dry=args.dry)
    else:
        print("Selected --mode is not supported!")
        exit(1)
