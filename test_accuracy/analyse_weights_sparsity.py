
import numpy as np
import matplotlib.pyplot as plt
import pickle

#import nntool APIs
from nntool.graph.types import LSTMNode, ConstantInputNode

from nntool.api import NNGraph

def plot_weights_dist(w_nodes, G, quantize):
    fig, axs = plt.subplots(len(w_nodes), figsize=(17, 10))
    idx = 0
    for w_node in w_nodes:
        print(w_node.name, w_node.dqvalue.shape)
        if quantize:
            out_q = G.quantization[w_node.name].out_qs[0]
            values = out_q.dequantize(w_node.value_as(out_q).flatten())
        else:
            values = w_node.dqvalue.flatten()

        n_values = values.size
        max_val = np.abs(values).max()
        values_under_5_percent = np.sum(np.where(np.abs(values) < 0.05*max_val, 1, 0)) / n_values
        values_under_10_percent = np.sum(np.where(np.abs(values) < 0.10*max_val, 1, 0)) / n_values
        print(f"{w_node.name}:\n\t5%: {100*values_under_5_percent:.2f}%\n\t10%: {100*values_under_10_percent:.2f}%")
        axs[idx].set_title(f"{w_node.name}: 5%: {100*values_under_5_percent:.2f}% - 10%: {100*values_under_10_percent:.2f}%")
        axs[idx].hist(values, bins=1000)
        axs[idx].axvline(x=values.min(), color = 'r', label='min/max')
        axs[idx].axvline(x=values.max(), color = 'r')
        axs[idx].axvline(x=3*values.std(), color = 'g', label='+-3 std')
        axs[idx].axvline(x=-3*values.std(), color = 'g')
        axs[idx].axvline(x=5*values.std(), color = 'y', label='+-5 std')
        axs[idx].axvline(x=-5*values.std(), color = 'y')
        axs[idx].set_xlim([-2.5, 2.5])
        idx += 1
        
    plt.tight_layout(pad=0.6)
    plt.legend()



G = NNGraph.load_graph("model/denoiser.onnx")
G.adjust_order()
G.fusions("scaled_match_group")
for lstm_node in G.nodes(node_classes=LSTMNode):
    lstm_node.set_c_state_as_output(G)

with open("lstm_stats", 'rb') as fp:
    astats = pickle.load(fp)

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
print(G.qshow())

w_nodes = [const_node for const_node in G.nodes(node_classes=ConstantInputNode) if len(const_node.dqvalue.shape) == 2 ]
plot_weights_dist(w_nodes[:4], G, quantize=True)
plot_weights_dist(w_nodes[4:8], G, quantize=True)
plot_weights_dist(w_nodes[8:12], G, quantize=True)
plot_weights_dist(w_nodes[12:16], G, quantize=True)
plt.show()
