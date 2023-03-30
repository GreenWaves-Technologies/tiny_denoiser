
import argparse
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from nntool.api import NNGraph
from nntool.graph.types import LSTMNode, RNNNodeBase
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
from nntool.api.utils import model_settings
import librosa
import numpy as np
import os
from pesq import pesq
from pystoi import stoi
import soundfile as sf

EUCLIDEAN_DISTANCES = []
RNN_STATE_SIZE = 256

WIN_LENGTH = 400
HOP_LENGTH = 100
N_FFT = 512
SAMPLERATE = 16000
WIN_FUNC = "hann"

def open_wav(file, expected_sr=SAMPLERATE):
    data, sr = sf.read(file)
    assert sr == expected_sr
    return data

def preprocessing(input_file):
    data = open_wav(input_file)
    stft = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WIN_FUNC, center=False)
    return stft

def postprocessing(stfts):
    data = librosa.istft(stfts, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WIN_FUNC, center=False)
    return data

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

UPDATE_MASK_EACH = 1
def single_audio_inference(G: NNGraph, stft_frame_i_T, stats_collector: ActivationRangesCollector = None, quant_exec=False):
    stft_frame_o_T = np.empty_like(stft_frame_i_T)
    rnn_nodes = [node for node in G.nodes(node_classes=RNNNodeBase, sort=True)]
    count_rnn_states = sum([2 if isinstance(node, LSTMNode) else 1 for node in G.nodes(node_classes=RNNNodeBase)])
    len_seq = stft_frame_i_T.shape[0]

    #init lstm to zeros
    rnn_states = [np.zeros(RNN_STATE_SIZE)] * count_rnn_states
    stft_mask = np.zeros(257)
    masks = []
    for i in tqdm(range(len_seq)):
        stft_clip = stft_frame_i_T[i]
        stft_clip_mag = np.abs(stft_clip)
        if not (i % UPDATE_MASK_EACH):
            data = [stft_clip_mag, *rnn_states]
            outputs = G.execute(data, dequantize=quant_exec)

            cnt = 0
            for node in rnn_nodes:
                rnn_states[cnt] = outputs[node.step_idx][0]
                cnt += 1
                if isinstance(node, LSTMNode):
                    rnn_states[cnt] = outputs[node.step_idx][1]
                    cnt += 1

            if stats_collector:
                stats_collector.collect_stats(G, data)

            new_stft_mask = outputs[G['output_1'].step_idx][0].squeeze()
            # See how the mask changes over time
            # EUCLIDEAN_DISTANCES.append(np.linalg.norm(new_stft_mask - stft_mask))
            # masks.append( new_stft_mask)
            stft_mask = new_stft_mask

        stft_clip = stft_clip * stft_mask
        stft_frame_o_T[i] = stft_clip
    # fig, ax = plt.subplots()
    # ax.plot(EUCLIDEAN_DISTANCES)
    # ax.imshow(np.array(masks))
    # plt.show()
    return stft_frame_o_T

def get_astats(G: NNGraph, dataset):
    stats_collector = ActivationRangesCollector(use_ema=False)
    files = os.listdir(dataset)
    for c, filename in tqdm(enumerate(files)):
        print(f"Collecting Stats from file {c+1}/{len(files)}")
        input_data = os.path.join(dataset, filename)
        stft = preprocessing(input_data)

        stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
        _ = single_audio_inference(G, stft_frame_i_T, stats_collector=stats_collector, quant_exec=False)
    return stats_collector.stats


def test_on_single_audio(G: NNGraph, audio_file, output_file, quant_exec=True):
    print(f"Running model on sample {audio_file} and writing result to {output_file}")
    stft = preprocessing(audio_file)
    stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
    stft_frame_o_T = single_audio_inference(G, stft_frame_i_T, quant_exec=quant_exec, stats_collector=None)
    estimate = postprocessing(stft_frame_o_T.T)
    # Write out audio as 24bit PCM WAV
    sf.write(output_file, estimate, SAMPLERATE)


def test_on_target(G: NNGraph, audio_file):
    print(f"Running model on sample {audio_file} on target")
    stft = preprocessing(audio_file)
    stft_frame_i_T = np.transpose(stft)[0] # swap the axis to select the tmestamp
    count_rnn_states = sum([2 if isinstance(node, LSTMNode) else 1 for node in G.nodes(node_classes=RNNNodeBase)])
    rnn_states = [np.zeros(RNN_STATE_SIZE)] * count_rnn_states
    res = G.execute_on_target(
        directory="/tmp/test_denoiser",
        input_tensors=[stft_frame_i_T, *rnn_states],
        check_on_target=True,
        print_output=True,
        at_loglevel=2,
        settings=model_settings(
            l1_size=128000,
            l2_size=1300000,
            graph_const_exec_from_flash=True,
            graph_group_weights=True,
            graph_l1_promotion=2
        ),
        tolerance=0.02
    )
    return res


def test_model_on_dataset(G: NNGraph, noisy_dataset, clean_dataset, quant_exec=False, output_dataset=None, dns_dataset=False):
    print(f"Testing on dataset: {noisy_dataset}")
    files = os.listdir(noisy_dataset)
    metric = []
    for c, filename in enumerate(files):
        noisy_file = os.path.join(noisy_dataset, filename)
        stft = preprocessing(noisy_file)

        stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
        stft_frame_o_T = single_audio_inference(G, stft_frame_i_T, quant_exec=quant_exec, stats_collector=None)

        estimate = postprocessing(stft_frame_o_T.T)

        # compute the metrics
        if dns_dataset:
            clean_filename = "clean_fileid_" + filename.split("_")[-1]
        else:
            clean_filename = filename

        clean_file = os.path.join(clean_dataset, clean_filename)
        clean_data = open_wav(clean_file)
        sz0 = clean_data.shape[0]
        sz1 = estimate.shape[0]
        if sz0 > sz1:
            estimate = np.pad(estimate, (0,sz0-sz1))
        else:
            estimate = estimate[:sz0]

        if output_dataset:
            output_file = os.path.join(output_dataset, filename)
            # Write out audio as 24bit PCM WAV
            sf.write(output_file, estimate, SAMPLERATE)

        pesq_i, stoi_i =  _run_metrics(clean_data, estimate, SAMPLERATE)
        print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")

        metric.append([pesq_i, stoi_i])
    return metric

def build_nntool_graph(model_path, astats_file, test_float=False, requantize=False, quant_dataset="./samples/quant/", states_as_inout=True):
    G = NNGraph.load_graph(model_path)
    G.adjust_order()
    G.fusions("scaled_match_group")
    if states_as_inout:
        for rnn_node in G.nodes(node_classes=RNNNodeBase):
            rnn_node.set_states_as_inputs(G)
            if isinstance(rnn_node, LSTMNode):
                rnn_node.set_c_state_as_output(G)
    if not test_float:
        if astats_file and os.path.exists(astats_file) and not requantize:
            with open(astats_file, 'rb') as fp:
                astats = pickle.load(fp)
        else:
            if not quant_dataset:
                raise ValueError("You need to provide quanti dataset for quantization")
            astats = get_astats(G, quant_dataset)
            if astats_file:
                with open(astats_file, 'wb') as fp:
                    pickle.dump(astats, fp, protocol=pickle.HIGHEST_PROTOCOL)

        graph_opts = {
            "clip_type": "std3" 
        }
        node_opts = {
            nname: { "scheme": "FLOAT", "float_type": "float16" }
            for nname in ["input_1", "Conv_0_reshape_in", "Conv_0_fusion", "Conv_147_fusion", "Conv_150_fusion", "Conv_150_reshape_out", "Sigmoid_151", "output_1"]
        }
        G.quantize(
            astats,
            graph_options=graph_opts,
            node_options=node_opts
        )
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'GAP denoiser', description="Speech enhancement using TinyDenoiser on GAP")
    #script mode
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to onnx file")
    parser.add_argument("--mode", type=str, default="test", choices=["sample", "test", "inference"],
                        help="Choose between sample | test")
    parser.add_argument("--astats_file", type=str, default=None,
                        help="Path to statistics pickle")
    parser.add_argument("--quant_dataset", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--clean_dataset", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--noisy_dataset", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--dns_testing", action="store_true")
    parser.add_argument("--output_dataset", type=str, default=None,
                        help="Path to output audio files dataset")
    parser.add_argument("--test_float", action="store_true",
                        help="Test full precision model")
    parser.add_argument("--requantize", action="store_true",
                        help="Requantize model")
    parser.add_argument("--audio_sample", type=str, default=None,
                        help="Path to audio file to test in sample mode")
    parser.add_argument("--output_sample", type=str, default=None,
                        help="Path to output audio file from sample")
    args = parser.parse_args()

    G = build_nntool_graph(args.model_path, args.astats_file, test_float=args.test_float, quant_dataset=args.quant_dataset, requantize=args.requantize, states_as_inout=not args.mode == "inference")
    print(G.show(G.input_nodes()))
    if not args.test_float:
        print(G.qshow())

    if args.mode == "test":
        if args.output_dataset and not os.path.exists(args.output_dataset):
            os.makedirs(args.output_dataset)

        results = test_model_on_dataset(G, args.noisy_dataset, args.clean_dataset, quant_exec=not args.test_float, output_dataset=args.output_dataset, dns_dataset=args.dns_testing)
        pesq_i = 0
        stoi_i = 0
        for p, s in results:
            pesq_i += p
            stoi_i += s
        final_pesq = pesq_i / len(results)
        final_stoi = stoi_i / len(results)
        print("Test set performance:PESQ=\t", final_pesq, "\t STOI=\t", final_stoi, '\t over', len(results), 'samples')
    elif args.mode == "sample":
        test_on_single_audio(G, args.audio_sample, args.output_sample, quant_exec=not args.test_float)
    elif args.mode == "inference":
        res = test_on_target(G, args.audio_sample)
        print(res.pretty_performance())
        fig = res.plot_memory_boxes()
        plt.show()
