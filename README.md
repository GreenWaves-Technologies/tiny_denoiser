# TinyDenoiser on GAP



## Getting Started
```
make clean all run platform=gvsoc WAV_FILE=/<path_to_audio_file>/<file_name>.wav
```


## Test on GAP
To denoise a wav file:
```
python test_accuracy/test_GAP.py --mode sample --pad_input 300 --sample_rate 16000 --wav_input /<path_to_audio_file>/<file_name>.wav
```

To test on dataset: 
```
python test_accuracy/test_GAP.py --mode test --pad_input 300 --dataset_path ./<path_to_audio_dataset>/
```