DNS_PATH="/scratch/datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb"

python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser.onnx --mode test --astats_file model/denoiser_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing --test_float         > denoiser_fp32.log
python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser.onnx --mode test --astats_file model/denoiser_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing                      > denoiser_mixed.log

python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser_GRU.onnx --mode test --astats_file model/denoiser_GRU_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing --test_float > denoiser_GRU_fp32.log
python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser_GRU.onnx --mode test --astats_file model/denoiser_GRU_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing              > denoiser_GRU_mixed.log

python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser_dns.onnx --mode test --astats_file model/denoiser_dns_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing --test_float > denoiser_dns_fp32.log
python test_accuracy/test_nntool_pesq_stoi.py --model_path model/denoiser_dns.onnx --mode test --astats_file model/denoiser_dns_stats.pickle --clean_dataset ${DNS_PATH}/clean --noisy_dataset ${DNS_PATH}/noisy --dns_testing              > denoiser_dns_mixed.log
