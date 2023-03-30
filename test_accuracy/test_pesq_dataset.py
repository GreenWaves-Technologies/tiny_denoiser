import os
import librosa
from test_nntool_pesq_stoi import _run_metrics, open_wav
import argparse

SAMPLERATE = 16000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'PESQ dataset', description="Get pesq stoi for given dataset")
    parser.add_argument("--clean_dataset", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--noisy_dataset", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--dns_testing", action="store_true")
    args = parser.parse_args()

    files = os.listdir(args.noisy_dataset)
    metric = []
    for c, filename in enumerate(files):
        noisy_file = os.path.join(args.noisy_dataset, filename)
        noisy_data = open_wav(noisy_file, expected_sr=SAMPLERATE)

        # compute the metrics
        if args.dns_testing:
            clean_filename = "clean_fileid_" + filename.split("_")[-1]
        else:
            clean_filename = filename

        clean_file = os.path.join(args.clean_dataset, clean_filename)
        clean_data = open_wav(clean_file, expected_sr=SAMPLERATE)
        pesq_i, stoi_i =  _run_metrics(clean_data, noisy_data, SAMPLERATE)
        print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")

        metric.append([pesq_i, stoi_i])
    pesq_i = 0
    stoi_i = 0
    for p, s in metric:
        pesq_i += p
        stoi_i += s
    final_pesq = pesq_i / len(metric)
    final_stoi = stoi_i / len(metric)
    print("Test set performance:PESQ=\t", final_pesq, "\t STOI=\t", final_stoi, '\t over', len(metric), 'samples')
