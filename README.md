# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9.  
The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap-and-add.


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

## Configs

The application code 

## Python Utilities


## Tests on TinyDenoisers 
```
make all run platform=gvsoc APP_MODE=3 SILENT=0 STFT_FRAMES=10
```

## Test on GAP
To denoise a wav file:
```
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input /<path_to_audio_file>/<file_name>.wav
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input samples/dataset/noisy/p232_050.wav --quant fp16mixed
```

To test on dataset: 
```
python test_accuracy/test_GAP.py --mode test --pad_input 300 --dataset_path ./<path_to_audio_dataset>/

