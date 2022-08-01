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
* denoiser.c is the 


## Configuration
The application code provides mulitple options, depending also if running on _board_ or _gvsoc_ target.
A list of available options includes:
* ds: 


### Demo Setting (APP_MODE = 0 or 1)


### Tests on TinyDenoisers (APP_MODE = 0 or 3)
test
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

