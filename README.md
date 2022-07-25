# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9.  
The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap and add.



## Getting Started
```
make clean all run platform=gvsoc
```
To denoise a custom wav audio file:
```
make clean all run platform=gvsoc WAV_FILE=/<path_to_audio_file>/<file_name>.wav
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
```