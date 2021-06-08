# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif



## Model Definition Parameters ##
QUANT_BITS=8
BUILD_DIR=BUILD

MODEL_PREFIX = denoiser
MODEL_SUFFIX = _$(QUANT_BITS)BIT

MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

TRAINED_MODEL = model/denoiser.onnx
MODEL_PATH = $(MODEL_BUILD)/$(MODEL_PREFIX).onnx
TENSORS_DIR = $(MODEL_BUILD)/tensors
MODEL_TENSORS = $(MODEL_BUILD)/$(MODEL_PREFIX)_L3_Flash_Const.dat

#Test Samples
WAV_PATH = $(CURDIR)/samples/temp.wav
FRAME_SIZE_ms = 40
FRAME_STEP_ms = 20
AT_INPUT_WIDTH=257 #1088
AT_INPUT_HEIGHT=1

# Quantization Library
#MODEL_SQ8=1
MODEL_FP16=1



NNTOOL_SCRIPT=model/nntool_script
NNTOOL_EXTRA_FLAGS =


CLUSTER_STACK_SIZE=8096
CLUSTER_SLAVE_STACK_SIZE=1024
CLUSTER_NUM_CORES=8
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY=350000
MODEL_L3_MEMORY=8000000
MODEL_SIZE_CFLAGS = -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT) -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH) -DAT_INPUT_COLORS=$(AT_INPUT_COLORS)
ifeq '$(TARGET_CHIP)' 'GAP8_V3'
	FREQ_CL?=175
else
	FREQ_CL?=50
endif
FREQ_FC?=250

include common/model_decl.mk
include $(RULES_DIR)/at_common_decl.mk

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
APP_SRCS    += denoiser.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB) common/wavIO.c

APP_CFLAGS += -Icommon -I$(GAP_SDK_HOME)/libs/gap_lib/include
APP_CFLAGS += -O3 -s -mno-memcpy -fno-tree-loop-distribute-patterns 
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) -I$(MODEL_BUILD) $(CNN_LIB_INCLUDE)
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE) -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DAT_WAV=$(WAV_PATH) #-DWRITE_WAV #-DPRINT_AT_INPUT #-DPRINT_WAV 
ifeq ($(platform), gvsoc)
	APP_CFLAGS += -DPERF
else
	APP_CFLAGS += -DPERF #-DFROM_SENSOR -DSILENT
endif

READFS_FILES=$(abspath $(MODEL_TENSORS))



generate_samples:
	python utils/generate_samples_images.py --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE_ms) --window_stride_ms $(FRAME_STEP_ms)

test_accuracy:
	python utils/test_accuracy_emul.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE_ms) --window_stride_ms $(FRAME_STEP_ms) --test_with_wav $(WITH_MFCC) --use_power_spectrogram $(USE_POWER)

test_accuracy_tflite:
	python utils/test_accuracy_tflite.py --tflite_model $(TRAINED_TFLITE_MODEL) --dct_coefficient_count $(DCT_COUNT) --window_size_ms $(FRAME_SIZE_ms) --window_stride_ms $(FRAME_STEP_ms) --use_power_spectrogram $(USE_POWER)

# all depends on the model
all:: model

clean:: clean_model

include common/model_rules.mk
#include mfcc_model.mk

$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))


$(info TILER_EMU_INC... $(TILER_EMU_INC))
$(info TILER_INC... $(TILER_INC))
$(info CNN_LIB_INCLUDE... $(CNN_LIB_INCLUDE))
include $(RULES_DIR)/pmsis_rules.mk
