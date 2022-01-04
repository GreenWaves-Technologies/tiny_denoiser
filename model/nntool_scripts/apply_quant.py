import numpy as np
import librosa
import sys, os
import pickle

#import nntool 
from interpreter.nntool_shell import NNToolShell
from execution.graph_executer import GraphExecuter
from stats.activation_ranges_collector import ActivationRangesCollector
from quantization.quantizer.new_quantizer import NewQuantizer
from graph.matches.matchers.remove_unnecessary_quantize_operators import RemoveUnnecessaryQuantizeOperators

# collect statistics
from cmd2 import Cmd2ArgumentParser, with_argparser
from utils.stats_funcs import STATS_BITS
QUANTIZATION_SCHEMES = ['SQ8', 'POW2']
from interpreter.shell_utils import glob_input_files, input_options
from quantization.handlers_helpers import (add_options_to_parser,
										   get_options_from_args)
parser_aquant = Cmd2ArgumentParser()
parser_aquant.add_argument('-f',
							   '--force_width',
							   choices=STATS_BITS, type=int, default=0,
							   help='force all layers to this bit-width in case of POW2 scheme, ' +
							   'SQ8 will automatically force 8-bits')
parser_aquant.add_argument('-s', '--scheme',
							   type=str, choices=QUANTIZATION_SCHEMES, default='SQ8',
							   help='quantize with scaling factors (TFlite quantization-like) [default] or POW2')

# read options
add_options_to_parser(parser_aquant)
input_options(parser_aquant)
args_quant = parser_aquant.parse_args([])
args_quant
opts = get_options_from_args(args_quant)
print(opts)

# read args
quant_sample_path = sys.argv[1]
quantization_bits = sys.argv[2]
gru = int(sys.argv[3])
print(gru)

path_model_build = sys.argv[4]
print(path_model_build)

# read stats
quantization_file = path_model_build + "data_quant.json"
fp = open(quantization_file, 'rb')
astats = pickle.load(fp)
fp.close()

scheme = 'SQ8' if quantization_bits == '8' else 'POW2'



quantizer = NewQuantizer(G, reset_all=True)
quantizer.options = opts
quantizer.schemes.append(args_quant.scheme)
quantizer.set_stats(astats)
quantizer.quantize()
G.add_dimensions()
