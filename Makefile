# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif


#quantization dependent features

# Quantization Mode
QUANT_BITS?=FP16
ifeq 		'$(QUANT_BITS)' '8'
	NNTOOL_SCRIPT=model/nntool_script_8
	MODEL_SQ8=1

else ifeq 	'$(QUANT_BITS)' '16'
	NNTOOL_SCRIPT=model/nntool_script

else ifeq 	'$(QUANT_BITS)' 'NE16'
	NNTOOL_SCRIPT=model/nntool_script_ne16
	MODEL_NE16=1
	MODEL_SQ8=1

else ifeq 	'$(QUANT_BITS)' 'FP16'
	NNTOOL_SCRIPT=model/nntool_script_fp16
	MODEL_FP16=1

else
	$(error Quantization mode is not recognized. Choose among 8, 16, FP16 or NE16)
endif


## Model Definition Parameters ##
BUILD_DIR=BUILD

MODEL_PREFIX = denoiser
MODEL_SUFFIX = _$(QUANT_BITS)BIT

MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

TRAINED_MODEL_PATH=model
TRAINED_MODEL = $(TRAINED_MODEL_PATH)/denoiser.onnx
MODEL_PATH = $(MODEL_BUILD)/$(MODEL_PREFIX).onnx
TENSORS_DIR = $(MODEL_BUILD)/tensors
MODEL_TENSORS = $(MODEL_BUILD)/$(MODEL_PREFIX)_L3_Flash_Const.dat
NNTOOL_EXTRA_FLAGS =

#Test Samples
IS_FAKE_SIGNAL_IN=0 
IS_AUDIO_FILE?=0
IS_AUDIO_FILE_STREAM?=0
IS_STFT_FILE_STREAM?=1

WAV_PATH = $(CURDIR)/samples/
WAV_SIGNAL_NAME=test.wav
WAV_FRAME_NAME=
STFT_FRAME_NAME=

TOT_FRAMES = 2
FRAME_SIZE = 400
FRAME_STEP = 100
NUM_FRAME_OVERLAP = 3
SAMPLING_FREQ = 16000
AT_INPUT_WIDTH=257 #1088
AT_INPUT_HEIGHT=1




ifeq '$(TARGET_CHIP)' 'GAP9_V2'
	FREQ_CL?=50
	FREQ_FC?=50

	CLUSTER_STACK_SIZE=10096
	CLUSTER_SLAVE_STACK_SIZE=2048
	CLUSTER_NUM_CORES=8
	TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
	MODEL_L1_MEMORY=$(shell expr 128000 \- $(TOTAL_STACK_SIZE))
#	MODEL_L2_MEMORY=1300000
	MODEL_L2_MEMORY=500000
	MODEL_L3_MEMORY=8000000

else
	ifeq '$(TARGET_CHIP)' 'GAP8_V3'
		FREQ_CL?=175
	else
		FREQ_CL?=50
	endif
	FREQ_FC?=250

	CLUSTER_STACK_SIZE=8096
	CLUSTER_SLAVE_STACK_SIZE=1024
	CLUSTER_NUM_CORES=8
	TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
	MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
	MODEL_L2_MEMORY=350000
	MODEL_L3_MEMORY=8000000
endif
MODEL_SIZE_CFLAGS = -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT) -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH) -DAT_INPUT_COLORS=$(AT_INPUT_COLORS)


include common/model_decl.mk
include $(RULES_DIR)/at_common_decl.mk
#include mfcc_model.mk
include stft_model.mk

RAM_FLASH_TYPE ?= HYPER
PMSIS_OS=pulpos

ifeq '$(RAM_FLASH_TYPE)' 'HYPER'
APP_CFLAGS += -DUSE_HYPER
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
APP_SRCS += $(MFCC_KER_SRCS)
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/TwiddlesDef.c 
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/RFFTTwiddlesDef.c 
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/LUT_Tables/SwapTablesDef.c
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/MfccBasicKernels.c 
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/FFT_Library.c 
#APP_SRCS += $(TILER_DSP_KERNEL_PATH)/math_funcs.c
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/CmplxFunctions.c 
APP_SRCS += $(TILER_DSP_KERNEL_PATH)/PreProcessing.c 

#include paths
APP_CFLAGS += -Icommon -I$(GAP_SDK_HOME)/libs/gap_lib/include
APP_CFLAGS += -O3 -s -mno-memcpy -fno-tree-loop-distribute-patterns 
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) -I$(MODEL_BUILD) $(CNN_LIB_INCLUDE)
APP_CFLAGS += -I$(MFCC_GENERATOR) -I$(TILER_DSP_KERNEL_PATH)
APP_CFLAGS += -I$(TILER_DSP_KERNEL_PATH) -I$(TILER_DSP_KERNEL_PATH)/LUT_Tables
APP_CFLAGS += -IBUILD_MODEL_STFT
#defines
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE) 
APP_CFLAGS += -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DAT_WAV=$(WAV_PATH) #-DWRITE_WAV #-DPRINT_AT_INPUT #-DPRINT_WAV 

APP_LDFLAGS		+= -lm



APP_CFLAGS += -DIS_FAKE_SIGNAL_IN=$(IS_FAKE_SIGNAL_IN)
APP_CFLAGS += -DIS_AUDIO_FILE=$(IS_AUDIO_FILE)
APP_CFLAGS += -DIS_AUDIO_FILE_STREAM=$(IS_AUDIO_FILE_STREAM)
APP_CFLAGS += -DIS_STFT_FILE_STREAM=$(IS_STFT_FILE_STREAM)
APP_CFLAGS += -DWAV_PATH=$(WAV_PATH)
APP_CFLAGS += -DWAV_SIGNAL_NAME=$(WAV_SIGNAL_NAME)
APP_CFLAGS += -DTOT_FRAMES=$(TOT_FRAMES)
APP_CFLAGS += -DFRAME_SIZE=$(FRAME_SIZE)
APP_CFLAGS += -DFRAME_STEP=$(FRAME_STEP)
APP_CFLAGS += -DNUM_FRAME_OVERLAP=$(NUM_FRAME_OVERLAP)
APP_CFLAGS += -DSAMPLING_FREQ=$(SAMPLING_FREQ)
APP_CFLAGS += -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH)
APP_CFLAGS += -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT)

APP_CFLAGS += -DPRINTDEB


ifeq ($(platform), gvsoc)
	APP_CFLAGS += -DPERF
else
	APP_CFLAGS += -DPERF #-DFROM_SENSOR -DSILENT
endif

READFS_FILES=$(abspath $(MODEL_TENSORS))



generate_samples:
	python utils/generate_samples_images.py --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP)

test_accuracy:
	python utils/test_accuracy_emul.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP) --test_with_wav $(WITH_MFCC) --use_power_spectrogram $(USE_POWER)

test_accuracy_tflite:
	python utils/test_accuracy_tflite.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE) --window_stride_ms $(FRAME_STEP) --use_power_spectrogram $(USE_POWER)

# all depends on the model
all:: model

clean:: clean_model
	rm -rf BUILD_MODEL*

include common/model_rules.mk

$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))

include $(RULES_DIR)/pmsis_rules.mk
