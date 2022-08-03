# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9.  
The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap-and-add.
As depitcted in the Figure below, the nosiy signal is windowed (frame size of 25 msec with an hop lenght of 6.25 msec and Hanning windowing) and the STFT is computed. 
The RNN is fed with the magnitude of the STFT components and return a suppression mask. After weighting, the inverse STFT returns a cleaned audio clip.

![alt text](imgs/TinyDenoiser.png?raw=true "Title" | width=50)

## Demo Getting Started
The demo runs on the GAP9 Audio EVK, using the microphone of the GAPmod board.
```
make clean all run [APP_MODE=0]
```
Optionally, the application can run on GVSOC to denoise a custom audio file (.wav).
```
make clean all run platform=gvsoc APP_MODE=1 [WAV_FILE=/<path_to_audio_file>/<file_name>.wav]
```
In the latter case do not forget to source GAP9_V2 target. Output wav file will be written to 
test_gap.wav inside the BUILD folder.

## Project Structure
* `denoiser.c` is the main file, including the application code
* `model/` includes the necessary files to feed GAPflow for NN model code generation: 
    * the _onnx_ denoiser files
        * `denoiser_dns.onnx` is a GRU based models trained on the [DNS][dns] dataset. It is used for demo purpose.
        * `denoiser.onnx` and `denoiser_GRU.onnx` are respectively LSTM and GRU models trained on the [Valentini][valentini]. they are used for testing purpose.
    * `nntool_scripts/` includes the nntool recipes to quantize the LSTM or GRU models. You can refer to the [quantization section](#quantization-details) for more details. 
* `samples/` contains the audio samples for testing and quantization claibration
* `stft_model.mk` and `model/STFTModel.c` are respectively the Makefile and the AT generator model for the STFT ad iSTFT functions. This files are manually configured. The baseline implementation exploits FP32 datatype.
*  `Graph.src` is the configuation file for Audio IO. It is used only for board target.
*  `test_accuracy/` includes the python scripts for model accuracy tests. You can refer to the [Python Utilities](#python-utilities) for more details.

## NN Quantization Settings
The Post-Training quantization process of the RNN model is operated by the GAPflow.
Both LSTM and GRU models can be quantized using one of the different options:
* `FP16`: quantizing both activations and weights to _float16_ format. This does not require any calibration samples.
* `INT8`: quantizing both activations and weights to _int_8_ format. A calibration step is required to quantize the activation functions. Samples included within `samples/quant/` are used to this aim. This option is currently not suggested because of the not-negligible accuracy degradation.
* `FP16MIXED`: only RNN layers are quantized to 8 bits, while the rest is kept to FP16. This option achives the **best** trade-off between accuracy degration and inference speed.
* `NE16`: currently not supported. 


## Project Configuration
The application code provides mulitple options, depending also if running on _board_ or _gvsoc_ target.
A list of available options includes:
* `FREQ_CL` and `FREQ_FC`: to set respectively the clock frequency of the cluster and the fabric controller. Also the periph frequency is set to FREQ_FC. Max frequnecy depends on the voltage: 370 if 0.8V and 240 if 0.65V.
* `VOLTAGE` (_board_ target only): to select between 0.8V (VOLTAGE=800) and 0.65V (VOLTAGE=650).
* `FLASH_TYPE`: type of L3 (external) FLASH memory. Set to 'DEFAULT' to adapt to the current board configuration (defined when sourcing the sdk, e.g. AUDIO_EVK). Optimal configuration is 'MRAM', if the model can fit.
* `RAM_TYPE`: type of L3 (external) RAM memory. Set to 'DEFAULT' to adapt to the current board configuration (defined when sourcing the sdk, e.g. AUDIO_EVK). 
* `IS_SFU` (_board_ target only): input data from microphone sensor.
* `IS_INPUT_STFT` (_gvsoc_ target only): 
    * [0]: input data from file as multiple STFT frames. Hence, STFT preprocessing is not applied and model inference runs over the loaded STFT spectrograms. Mainly used for testing.
    * [1]: input audio data from file. The wav file is configured with WAV_FILE.
* `WAV_FILE`: absolute path of the input wav file. 
* `DISABLE_NN_INFERENCE`: if set to 1, the inference task is skipped. Default is 0. Mainly used for testing.
* `SILENT`: to enable debug printf (default is 0).

## APP_MODE Configuration
In addition to individua settings, some application mode are made available to simplify the APP code configuration. This is done by setting the APP_MODE varaible (default is 0).
### Demo Setting (APP_MODE = 0 or 1)
The code runs inference using the `denoiser_dns.onnx` model with  `FP16MIXED` quantization. 
* `APP_MODE = 0` is meant to run on _board_ target and audio data comes from the microphone.
* `APP_MODE = 1` is meant to run on _gvsoc_ target and audio data comes from the WAV_FILE file. The wav cleaned audio can be retrieved from the _BUILD_ folder.

You can refer to the commands in the [Demo Getting Started section](#demo-getting-started).
### Tests on STFT and iSTFT (APP_MODE 2)
In this configuration, the NN inference is disabled and an audio frame feeds the STFT + iSTFT pipeline. A final checksum checks the similarity between the input and output signals. 

### Tests on TinyDenoisers (APP_MODE 3)
* `GRU`
* `QUANT_BITS`

```
make all run platform=gvsoc APP_MODE=3 SILENT=0 STFT_FRAMES=10
```


## Python Utilities


## Test on GAP
To denoise a wav file:
```
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input /<path_to_audio_file>/<file_name>.wav
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input samples/dataset/noisy/p232_050.wav --quant fp16mixed
```

To test on dataset: 
```
python test_accuracy/test_GAP.py --mode test --pad_input 300 --dataset_path ./<path_to_audio_dataset>/
```

[dns]: https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/
[valentini]: https://datashare.ed.ac.uk/handle/10283/2791

