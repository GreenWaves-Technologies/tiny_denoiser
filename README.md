# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9.  
The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap-and-add.
As depitcted in the Figure below, the nosiy signal is windowed (frame size of 25 msec with an hop lenght of 6.25 msec and Hanning windowing) and the STFT is computed. 
The RNN is fed with the magnitude of the STFT components and return a suppression mask. After weighting, the inverse STFT returns a cleaned audio clip.

![alt text](imgs/TinyDenoiser.png?raw=true "Title")

## Demo Getting Started
The demo runs on the GAP9 Audio EVK, using the microphone of the GAPmod board.
```
cmake -B build
cmake --build build --target run
```
Optionally, the application can run on GVSOC (or board) to denoise a custom audio file (.wav).
```
cmake -B build
cmake --build build --target menuconfig # Select the options DenoiseWav in the DENOISER APP -> Application mode menu
cmake --build build --target run
```
Output wav file will be written to test_gap.wav inside the project folder.

## Project Structure
* `denoiser.c` is the main file, including the application code
* `model/` includes the necessary files to feed GAPflow for NN model code generation: 
    * the _onnx_ denoiser files
        * `denoiser_dns.onnx` is a GRU based models trained on the [DNS][dns] dataset. It is used for demo purpose.
        * `denoiser.onnx` and `denoiser_GRU.onnx` are respectively LSTM and GRU models trained on the [Valentini][valentini]. they are used for testing purpose.
    * `nntool_scripts/` includes the nntool recipes to quantize the LSTM or GRU models. You can refer to the [quantization section](#nn-quantization-settings) for more details. 
* `samples/` contains the audio samples for testing and quantization claibration
* `model/STFTModel.c` is the AT generator model for the STFT ad iSTFT functions. This files are manually configured. The baseline implementation exploits FP32 datatype.
*  `Graph.src` is the configuation file for Audio IO. It is used only for board target.
*  `test_accuracy/` includes the python scripts for model accuracy tests. You can refer to the [Python Utilities](#python-utilities) for more details.

## NN Quantization Settings
The Post-Training quantization process of the RNN model is operated by the GAPflow.
Both LSTM and GRU models can be quantized using one of the different options:
* `FP16`: quantizing both activations and weights to _float16_ format. This does not require any calibration samples.
* `INT8`: quantizing both activations and weights to _int_8_ format. A calibration step is required to quantize the activation functions. Samples included within `samples/quant/` are used to this aim. This option is currently not suggested because of the not-negligible accuracy degradation.
* `FP16MIXED`: only RNN layers are quantized to 8 bits, while the rest is kept to FP16. This option achives the **best** trade-off between accuracy degration and inference speed.
* `NE16`: currently not supported. 

## Application Mode Configuration
In addition to individual settings, some application mode are made available to simplify the APP code configuration. This is done by setting the Application Mode in the `make menuconfig` DENOISER APP menu
### Demo Setting (Application Mode DEMO or DenoiserWav)
The code runs inference using the `denoiser_dns.onnx` model with  `FP16MIXED` quantization. More accurate at higher energy costs can be obtained with `FP16` quantization by changing the `nntool_script_demo`.
* `Demo` is meant to run on _board_ target and audio data comes from the microphone and output is sent to jack output on audio add on.
* `DenoiserWav = 1` is meant to run on _gvsoc_ and _board_ target and audio data comes from the WAV_FILE file. The wav cleaned audio can be retrieved from the root folder test_gap.wav folder.


## Python Utilities
The `test_accuracy/test_GAP.py` file provides the routines for testing the NN inference model using the NNtool API. The script can be used to run tests on entire datasets (`--mode test`) or to denoise individual audio files (`--mode test`). Some examples are provided below. 

### To denoise a wav file
```
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input /<path_to_audio_file>/<file_name>.wav
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input samples/dataset/noisy/p232_050.wav --quant fp16mixed
```
The output is saved in a file called test_gap.wav in the home of the repository

### To test on dataset
```
python test_accuracy/test_GAP.py --mode test --pad_input 300 --noisy_dataset_path ./<path_to_noisy_audio_dataset>/ --clean_dataset_path ./<path_to_clean_audio_dataset>/
```

[dns]: https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/
[valentini]: https://datashare.ed.ac.uk/handle/10283/2791

