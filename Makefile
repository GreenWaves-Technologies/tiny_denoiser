# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif


##############################################
############ Application Mode ################
# 0:	Demo: input SFU, Run Denoiser, Output SFU
# 1:	DenoiseWav: Input file Wav, Run Denoiser, Output file Wav
# 2: 	DSPWav_test: Input file Wav, Run Denoiser but not NN, Check Output Wav
# 3:  NN_Test: Input file STFT, Run NN Denoiser only, check NN Output
APP_MODE?=3
############################################## 
# 0:	Demo
ifeq ($(APP_MODE), 0)
	IS_SFU=1 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=0
endif
# 1:	DenoiseWav
ifeq ($(APP_MODE), 1)
	IS_SFU=0 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=0
endif
# 2: 	DSPWav_test
ifeq ($(APP_MODE), 2)
	IS_SFU=0 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=1
endif
# 3:  NN_Test
ifeq ($(APP_MODE), 3)
	IS_SFU=0 
	IS_INPUT_STFT=1
	DISABLE_NN_INFERENCE=0
	STFT_FRAMES=1
endif
############################################## 


#quantization dependent features

# Quantization Mode
# FP16=float16, BFP16 = float16alt
QUANT_BITS?=FP16
GRU?=0

SILENT?=1
DEBUG?=0
DEBUG_STFT?=0


FREQ_CL=370
FREQ_FC=370


NNTOOL_EXTRA_FLAGS =
ifeq 		'$(QUANT_BITS)' '8'
	MODEL_SQ8=1
	MODEL_FP16=1

	NNTOOL_EXTRA_FLAGS=--use_lut_sigmoid --use_lut_tanh
	ifeq ($(GRU), 0)
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_int8
	else
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_int8_gru 
	endif

else ifeq 	'$(QUANT_BITS)' '16'
	NNTOOL_SCRIPT=model/nntool_scripts/nntool_script

else ifeq 	'$(QUANT_BITS)' 'NE16'
	MODEL_NE16=1
	MODEL_SQ8=1
	ifeq ($(GRU), 0)
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_ne16
	else
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_ne16_gru 
	endif


else ifeq 	'$(QUANT_BITS)' 'FP16'
	MODEL_FP16=1
	NNTOOL_EXTRA_FLAGS=--use_lut_sigmoid --use_lut_tanh
	ifeq ($(GRU), 0)
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16
	else
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16_gru
	endif

else ifeq 	'$(QUANT_BITS)' 'FP16MIXED'
	MODEL_FP16=1
	MODEL_SQ8=1

	NNTOOL_EXTRA_FLAGS=--use_lut_sigmoid --use_lut_tanh
	ifeq ($(GRU), 0)
		NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16_mixed
	else
		NNTOOL_SCRIPT=
	endif



else ifeq 	'$(QUANT_BITS)' 'BFP16'
	NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_bfp16
	MODEL_FP16=1
	NNTOOL_EXTRA_FLAGS=--use_lut_sigmoid --use_lut_tanh



else
	$(error Quantization mode is not recognized. Choose among 8, 16, FP16 or NE16)
endif


## Model Definition Parameters ##
BUILD_DIR?=BUILD

MODEL_SUFFIX = _$(QUANT_BITS)BIT

MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

TRAINED_MODEL_PATH=model
ifeq ($(GRU), 0)
	MODEL_PREFIX = denoiser
else
	MODEL_PREFIX = denoiser_GRU
endif

TRAINED_MODEL = $(TRAINED_MODEL_PATH)/$(MODEL_PREFIX).onnx

MODEL_PATH = $(MODEL_BUILD)/$(MODEL_PREFIX).onnx
TENSORS_DIR = $(MODEL_BUILD)/tensors
MODEL_TENSORS = $(MODEL_BUILD)/$(MODEL_PREFIX)_L3_Flash_Const.dat



# set the input files
WAV_FILE?=$(CURDIR)/samples/sample_0000.wav
STFT_FILE=

STFT_FRAMES?=10
FRAME_SIZE=400
FRAME_STEP=100
FRAME_NFFT=512
NUM_FRAME_OVERLAP=3
SAMPLING_FREQ=16000
AT_INPUT_WIDTH=257 #1088
AT_INPUT_HEIGHT=1




ifeq '$(TARGET_CHIP)' 'GAP9_V2'
	FREQ_CL?=370
	FREQ_FC?=370

	CLUSTER_STACK_SIZE=4096
	CLUSTER_SLAVE_STACK_SIZE=2048
	CLUSTER_NUM_CORES=8
	TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES))
#	MODEL_L1_MEMORY=$(shell expr 120000 \- $(TOTAL_STACK_SIZE))
	MODEL_L1_MEMORY=$(shell expr  50000 \- $(TOTAL_STACK_SIZE))
#	MODEL_L2_MEMORY=1300000
	MODEL_L2_MEMORY=1000000
	MODEL_L3_MEMORY=8000000

else
	ifeq '$(TARGET_CHIP)' 'GAP9'
		FREQ_CL?=50
		FREQ_FC?=50

		CLUSTER_STACK_SIZE=4096
		CLUSTER_SLAVE_STACK_SIZE=2048
		CLUSTER_NUM_CORES=8
		TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES))
		MODEL_L1_MEMORY=$(shell expr 120000 \- $(TOTAL_STACK_SIZE))
	#	MODEL_L2_MEMORY=1300000
		MODEL_L2_MEMORY=300000
		MODEL_L3_MEMORY=8000000

	else
		ifeq '$(TARGET_CHIP)' 'GAP8_V3'
			FREQ_CL?=175
		else
			FREQ_CL?=50
		endif
		FREQ_FC?=250

		CLUSTER_STACK_SIZE=8096
		CLUSTER_SLAVE_STACK_SIZE=8096
		CLUSTER_NUM_CORES=8
		TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
		MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
		MODEL_L2_MEMORY=350000
		MODEL_L3_MEMORY=8000000
	endif
endif
MODEL_SIZE_CFLAGS = -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT) -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH) -DAT_INPUT_COLORS=$(AT_INPUT_COLORS)


include common/model_decl.mk
include $(RULES_DIR)/at_common_decl.mk
include stft_model.mk

RAM_FLASH_TYPE ?= HYPER
PMSIS_OS=freertos

ifeq '$(RAM_FLASH_TYPE)' 'HYPER'
APP_CFLAGS += -DUSE_HYPER
CONFIG_HYPERRAM = 1
MODEL_L3_EXEC=hram
MODEL_L3_CONST=hflash
else
APP_CFLAGS += -DUSE_SPI
CONFIG_SPIRAM = 1
MODEL_L3_EXEC=qspiram
MODEL_L3_CONST=qpsiflash
endif


io=host

## File Definition ##
APP_SRCS += denoiser.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB) 
APP_SRCS += $(GAP_LIB_PATH)/wav_io/wavIO.c
#APP_SRCS += BUILD_MODEL_STFT/MFCCKernels.c  
APP_SRCS += BUILD_MODEL_STFT/RFFTKernels.c  
#APP_SRCS += $(MFCC_KER_SRCS)
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/TwiddlesDef.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/RFFTTwiddlesDef.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/SwapTablesDef.c

#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/MfccBasicKernels.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/FFT_Library.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/math_funcs.c
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/CmplxFunctions.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/PreProcessing.c 

#include paths
APP_CFLAGS += -Icommon -I$(GAP_SDK_HOME)/libs/gap_lib/include/gaplib/
APP_CFLAGS += -O3 -s -mno-memcpy -fno-tree-loop-distribute-patterns 



APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) -I$(MODEL_BUILD) $(CNN_LIB_INCLUDE)
APP_CFLAGS += -I$(MFCC_GENERATOR) -I$(TILER_DSP_KERNEL_PATH) -I$(TILER_DSP_KERNEL_PATH)/LUT_Tables
APP_CFLAGS += -IBUILD_MODEL_STFT
APP_CFLAGS += -Isamples

#defines
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE) 
APP_CFLAGS += -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DWAV_FILE=$(WAV_FILE) #-DWRITE_WAV #-DPRINT_AT_INPUT #-DPRINT_WAV 

APP_LDFLAGS		+= -lm



APP_CFLAGS += -DIS_SFU=$(IS_SFU)
APP_CFLAGS += -DIS_AUDIO_FILE=$(IS_AUDIO_FILE)
APP_CFLAGS += -DIS_INPUT_STFT=$(IS_INPUT_STFT)

APP_CFLAGS += -DSTFT_FRAMES=$(STFT_FRAMES)
APP_CFLAGS += -DFRAME_SIZE=$(FRAME_SIZE)
APP_CFLAGS += -DFRAME_STEP=$(FRAME_STEP)
APP_CFLAGS += -DFRAME_NFFT=$(FRAME_NFFT)
APP_CFLAGS += -DNUM_FRAME_OVERLAP=$(NUM_FRAME_OVERLAP)
APP_CFLAGS += -DSAMPLING_FREQ=$(SAMPLING_FREQ)
APP_CFLAGS += -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH)
APP_CFLAGS += -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT)
APP_CFLAGS += -DMAX_L2_BUFFER=$(MODEL_L2_MEMORY)


ifeq 	'$(QUANT_BITS)' 'FP16'
	APP_CFLAGS += -DDTYPE=0
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	 	'$(QUANT_BITS)' 'FP16MIXED'
	APP_CFLAGS += -DDTYPE=0
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	'$(QUANT_BITS)' 'BFP16'
	APP_CFLAGS += -DDTYPE=1
	APP_CFLAGS += -DF16_DSP_BFLOAT

else ifeq 	'$(QUANT_BITS)' '8'
	APP_CFLAGS += -DDTYPE=2
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	'$(QUANT_BITS)' 'NE16'
	APP_CFLAGS += -DDTYPE=2
	APP_CFLAGS += -DSTD_FLOAT

else
	APP_CFLAGS += -DDTYPE=3

endif

ifeq ($(platform), gvsoc)
	APP_CFLAGS += -DPERF
else
	APP_CFLAGS += -DPERF #-DFROM_SENSOR -DSILENT
endif

ifeq ($(SILENT), 1)
	APP_CFLAGS += -DSILENT
endif

ifeq ($(DEBUG_STFT), 1)
	APP_CFLAGS += -DPRINTDEB
endif

ifeq ($(DEBUG), 1)
	APP_CFLAGS += -DPRINTDEBUG
endif


ifeq ($(DISABLE_NN_INFERENCE), 1)
	APP_CFLAGS += -DDISABLE_NN_INFERENCE
endif

ifeq ($(GRU), 1)
	APP_CFLAGS += -DGRU
endif



READFS_FILES=$(abspath $(MODEL_TENSORS))



generate_samples:
	python utils/generate_samples_images.py --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP)

test_accuracy:
	python utils/test_accuracy_emul.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP) --test_with_wav $(WITH_MFCC) --use_power_spectrogram $(USE_POWER)

test_accuracy_tflite:
	python utils/test_accuracy_tflite.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP) --use_power_spectrogram $(USE_POWER)

# all depends on the model
all:: model gen_fft_code

clean:: clean_model clean_fft_code
	rm -rf BUILD_MODEL*

include common/model_rules.mk

$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))

include $(RULES_DIR)/pmsis_rules.mk
