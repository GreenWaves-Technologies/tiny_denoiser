
import argparse
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from nntool.api import NNGraph
from nntool.graph.types import LSTMNode, RNNNodeBase
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
from nntool.api.utils import model_settings
import numpy as np
import os
import soundfile as sf
from denoiser_utils import single_audio_inference, get_astats, _run_metrics, preprocessing, postprocessing, open_wav, SAMPLERATE

EUCLIDEAN_DISTANCES = []
RNN_STATE_SIZE = 256

def test_on_single_audio(G: NNGraph, noisy_file, output_file, quant_exec=True, clean_file=None):
    print(f"Running model on sample {noisy_file} and writing result to {output_file}")
    noisy_data = open_wav(noisy_file)
    stft = preprocessing(noisy_data)
    stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
    stft_frame_o_T = single_audio_inference(G, stft_frame_i_T, quant_exec=quant_exec, stats_collector=None)
    estimate = postprocessing(stft_frame_o_T.T)
    # Write out audio as 24bit PCM WAV
    sf.write(output_file, estimate, SAMPLERATE)
    if clean_file:
        clean_data = open_wav(clean_file)
        sz0 = clean_data.shape[0]
        sz1 = estimate.shape[0]
        if sz0 > sz1:
            estimate = np.pad(estimate, (0,sz0-sz1))
        else:
            estimate = estimate[:sz0]
        pesq_org_i, stoi_org_i =  _run_metrics(clean_data, noisy_data, SAMPLERATE)
        pesq_denoised_i, stoi_denoised_i =  _run_metrics(clean_data, estimate, SAMPLERATE)
        print(f"{clean_file}\twith pesq, stoi=\t({pesq_denoised_i:.4f},{stoi_denoised_i:.4f})\torg: ({pesq_org_i:.4f},{stoi_org_i:.4f})")


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


def test_model_on_dataset(G: NNGraph, noisy_dataset, clean_dataset, quant_exec=False, output_dataset=None, dns_dataset=False, verbose=True):
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
        if verbose:
            print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")

        metric.append([filename, pesq_i, stoi_i])
    return metric

def build_nntool_graph(model_path, astats_file, test_float=False, requantize=False, quant_dataset="./samples/quant/", states_as_inout=True, qtype="mixed"):
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

        if qtype == "mixed":
            graph_opts = {
                "clip_type": "none",
                "allow_asymmetric_out": True
            }
            node_opts = {
                nname: { "scheme": "FLOAT", "float_type": "float16" }
                for nname in [
                    "input_1",
                    "Conv_0_reshape_in",
                    "Conv_0_fusion",
                    "Conv_147_fusion",
                    "Conv_150_fusion",
                    "Conv_150_reshape_out",
                    "Conv_139_fusion",
                    "Conv_142_fusion",
                    "Conv_142_reshape_out",
                    "Sigmoid_151",
                    "output_1"
                ]
            }
        elif qtype == "int8":
            graph_opts = {
                "clip_type": "none",
                "allow_asymmetric_out": True
            }
            node_opts = None
        elif qtype == "fp16":
            graph_opts = {
                "scheme": "FLOAT",
                "float_type": "float16"
            }
            node_opts = None
        else:
            raise ValueError(f"Quant Type {qtype} not supported")

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
    parser.add_argument("--mode", type=str, default="test", choices=["sample", "test", "inference", "quantize"],
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
    parser.add_argument("--clean_sample", type=str, default=None,
                        help="Path to audio file to test in sample mode")
    parser.add_argument("--output_sample", type=str, default=None,
                        help="Path to output audio file from sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Print pesq after each sample")
    args = parser.parse_args()

    G = build_nntool_graph(
        args.model_path,
        args.astats_file,
        test_float=args.test_float and not args.mode == "quantize",
        quant_dataset=args.quant_dataset,
        requantize=args.requantize or args.mode == "quantize",
        states_as_inout=True
    )
    print(G.show(G.input_nodes()))
    if not args.test_float:
        print(G.qshow())
    if args.mode == "quantize":
        exit

    if args.mode == "test":
        if args.output_dataset and not os.path.exists(args.output_dataset):
            os.makedirs(args.output_dataset)

        results = test_model_on_dataset(G, args.noisy_dataset, args.clean_dataset, quant_exec=not args.test_float, output_dataset=args.output_dataset, dns_dataset=args.dns_testing, verbose=args.verbose)
        pesq_i = 0
        stoi_i = 0
        for _, p, s in results:
            pesq_i += p
            stoi_i += s
        final_pesq = pesq_i / len(results)
        final_stoi = stoi_i / len(results)
        print("Test set performance:PESQ=\t", final_pesq, "\t STOI=\t", final_stoi, '\t over', len(results), 'samples')
        results.append(["average", final_pesq, final_stoi])
        df = pd.DataFrame(results, columns=["filename", "PESQ", "STOI"])
        print(df)

    elif args.mode == "sample":
        test_on_single_audio(G, args.audio_sample, args.output_sample, quant_exec=not args.test_float, clean_file=args.clean_sample)
    elif args.mode == "inference":
        res = test_on_target(G, args.audio_sample)
        print(res.pretty_performance())
        fig = res.plot_memory_boxes()
        plt.show()
