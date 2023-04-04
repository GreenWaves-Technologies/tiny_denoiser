
from nntool.api import NNGraph
from nntool.graph.types import LSTMNode, RNNNodeBase
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
from tqdm import tqdm
import librosa
import numpy as np
import os
from pesq import pesq
from pystoi import stoi

WIN_LENGTH = 400
HOP_LENGTH = 100
N_FFT = 512
SAMPLERATE = 16000
WIN_FUNC = "hann"

def open_wav(file, expected_sr=SAMPLERATE, verbose=False):
    data, sr = librosa.load(file, sr=expected_sr)
    if sr != expected_sr:
        if verbose:
            print(f"expected sr: {expected_sr} real: {sr} -> resampling")
        data = librosa.resample(data, orig_sr=sr, target_sr=expected_sr)
    return data

def preprocessing(input_file):
    if isinstance(input_file, str):
        data = open_wav(input_file)
    else:
        data = input_file
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
    rnn_states = []
    for rnn_node in rnn_nodes:
        rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))
        if isinstance(rnn_node, LSTMNode):
            rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))

    len_seq = stft_frame_i_T.shape[0]

    #init lstm to zeros
    stft_mask = np.zeros(257)
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
                    rnn_states[cnt] = outputs[node.step_idx][-1]
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
