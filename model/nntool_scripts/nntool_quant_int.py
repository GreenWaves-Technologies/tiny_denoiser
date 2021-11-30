import numpy as np
import librosa
import sys, os

#import nntool 
from interpreter.nntool_shell import NNToolShell
from execution.graph_executer import GraphExecuter
from stats.activation_ranges_collector import ActivationRangesCollector
from quantization.quantizer.new_quantizer import NewQuantizer
from graph.matches.matchers.remove_unnecessary_quantize_operators import RemoveUnnecessaryQuantizeOperators


# export some layers to be moved to nntool script
#NNToolShell.run_commands_on_graph(G, ['nodeoption LSTM_78 RNN_STATES_AS_INPUTS 1',
#									   'nodeoption LSTM_78 LSTM_OUTPUT_C_STATE 1',
#									   'nodeoption LSTM_144 RNN_STATES_AS_INPUTS 1',
#									   'nodeoption LSTM_144 LSTM_OUTPUT_C_STATE 1',
#									   'show'])
#

# input variables
quant_sample_path = sys.argv[1]
quantization_bits = sys.argv[2]
gru = int(sys.argv[3])
print(gru)
if gru == 1:
	print('This is a GRU-based model')
else:
	print('This is a LSTM-based model')

print('Quantization samples from: ', quant_sample_path, 'quantized to ', quantization_bits, ' bits')

# parameters
SR = 16000
use_ema = True
lstm_hidden_states = 256

# defines
executer = GraphExecuter(G, qrecs=None)

stats_collector = ActivationRangesCollector(use_ema=use_ema)
G.quantization = None



for filename in os.listdir(quant_sample_path):
	input_file = quant_sample_path + filename
	data, _ = librosa.load(input_file, sr=SR)
	stft = librosa.stft(data, n_fft=512, hop_length=100, win_length=400, 
		window='hann', center=False )
	rstft = np.abs(stft)
	len_seq = rstft.shape[1]

	#init lstm to zeros
	lstm_0_i_state = np.zeros(lstm_hidden_states)
	lstm_1_i_state = np.zeros(lstm_hidden_states)
	lstm_0_c_state = np.zeros(lstm_hidden_states)
	lstm_1_c_state = np.zeros(lstm_hidden_states)

	# debug stuff
	lim_0 = 0
	lim_1 = 0
	lim_2 = 0
	lim_3 = 0


	for i in range(len_seq): 
		single_mags = rstft[:,i]

		if gru == 1:
			data = [single_mags, lstm_0_i_state, lstm_1_i_state]
		else:
			data = [single_mags, lstm_0_i_state, lstm_0_c_state, lstm_1_i_state, lstm_1_c_state]


		stats_collector.collect_stats(G, data)
		outputs = executer.execute(data, qmode=None, silent=True)
		
		if gru == 1:
			lstm_0_i_state = outputs[31][0]
			lstm_1_i_state = outputs[35][0]
		else:
			lstm_0_i_state = outputs[36][0]
			lstm_0_c_state = outputs[38][0]
			lstm_1_i_state = outputs[41][0]
			lstm_1_c_state = outputs[47][0]

		print(lstm_0_i_state.shape)


		# debug monitor lstm state quantization
		if gru == 0:
			max_stats = np.max(np.abs(lstm_0_c_state))
			lim_1 = max_stats if max_stats > lim_1 else lim_1
			print('rnn_0_c_state | Sample: ',i,', Max: ', max_stats, 'Glob Max', lim_1)

			max_stats = np.max(np.abs(lstm_1_c_state))
			lim_3 = max_stats if max_stats > lim_3 else lim_3
			print('rnn_1_c_state | Sample: ',i,', Max: ', max_stats, 'Glob Max', lim_3)

		max_stats = np.max(np.abs(lstm_0_i_state))
		lim_0 = max_stats if max_stats > lim_0 else lim_0    
		print('rnn_0_i_state | Sample: ',i,', Max: ', max_stats, 'Glob Max', lim_0)

		max_stats = np.max(np.abs(lstm_1_i_state))
		lim_2 = max_stats if max_stats > lim_2 else lim_2   
		print('rnn_1_i_state | Sample: ',i,', Max: ', max_stats, 'Glob Max', lim_2)
				


		
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

add_options_to_parser(parser_aquant)
input_options(parser_aquant)
args_quant = parser_aquant.parse_args([])
args_quant
opts = get_options_from_args(args_quant)
print(opts)


scheme = 'SQ8' if quantization_bits == '8' else 'POW2'

astats = stats_collector.stats

quantizer = NewQuantizer(G, reset_all=True)
quantizer.options = opts
quantizer.schemes.append(args_quant.scheme)
quantizer.set_stats(astats)
quantizer.quantize()
G.add_dimensions()

