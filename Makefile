# Copyright (C) 2020 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

include $(RULES_DIR)/pmsis_defs.mk

##############################################
############ Application Mode ################
# 0:	Demo: input SFU, Run Denoiser, Output SFU
# 1:	Demo DenoiseWav: Input file Wav, Run Denoiser, Output file Wav
# 2: 	DSPWav_test: Input file Wav, Run Denoiser but not NN, Check Output Wav
# 3:  	NN_Test: Input file STFT, Run NN Denoiser only, check NN Output
APP_MODE?=0
############################################## 
# 0:	Demo
ifeq ($(APP_MODE), 0)
	IS_SFU=1 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=0

	APP_SRCS   += $(TARGET_BUILD_DIR)/GraphINOUT_L2_Descr.c $(SFU_RUNTIME)/SFU_RT.c
	APP_CFLAGS += -I$(TARGET_BUILD_DIR) -I$(SFU_RUNTIME)/include
	io=uart
	DEMO=1

endif
# 1:	DenoiseWav
ifeq ($(APP_MODE), 1)
	IS_SFU=0 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=0
	io=host
	WAV_FILE?=$(CURDIR)/samples/real_samples/phone_call.wav
	DEMO=1
endif
# 2: 	DSPWav_test
ifeq ($(APP_MODE), 2)
	IS_SFU=0 
	IS_INPUT_STFT=0
	DISABLE_NN_INFERENCE=1
	WAV_FILE?=$(CURDIR)/samples/dataset/noisy/p232_050.wav
	io=host
	DEMO=0
	CHECKSUM=1
	STFT_FRAMES=1
endif
# 3:  NN_Test
ifeq ($(APP_MODE), 3)
	IS_SFU=0 
	IS_INPUT_STFT=1
	DISABLE_NN_INFERENCE=0
	STFT_FRAMES=1
	io=host
	CHECKSUM=1
	DEMO=0
endif


############################################## 
FLASH_TYPE ?= DEFAULT
RAM_TYPE   ?= DEFAULT
#############################################
### 	External Mem Settings
#############################################
EXEC_FROM_FLASH ?= false
ifeq '$(FLASH_TYPE)' 'HYPER'
    MODEL_L3_FLASH=AT_MEM_L3_HFLASH
else ifeq '$(FLASH_TYPE)' 'MRAM'
    MODEL_L3_FLASH=AT_MEM_L3_MRAMFLASH
    READFS_FLASH = target/chip/soc/mram
    EXEC_FROM_FLASH=true
else ifeq '$(FLASH_TYPE)' 'QSPI'
    MODEL_L3_FLASH=AT_MEM_L3_QSPIFLASH
    READFS_FLASH = target/board/devices/spiflash
else ifeq '$(FLASH_TYPE)' 'OSPI'
    MODEL_L3_FLASH=AT_MEM_L3_OSPIFLASH
else ifeq '$(FLASH_TYPE)' 'DEFAULT'
    MODEL_L3_FLASH=AT_MEM_L3_DEFAULTFLASH
endif

ifeq '$(RAM_TYPE)' 'HYPER'
    MODEL_L3_RAM=AT_MEM_L3_HRAM
else ifeq '$(RAM_TYPE)' 'QSPI'
    MODEL_L3_RAM=AT_MEM_L3_QSPIRAM
else ifeq '$(RAM_TYPE)' 'OSPI'
    MODEL_L3_RAM=AT_MEM_L3_OSPIRAM
else ifeq '$(RAM_TYPE)' 'DEFAULT'
    MODEL_L3_RAM=AT_MEM_L3_DEFAULTRAM
endif

#quantization dependent features

# Quantization Mode
# FP16=float16
QUANT_BITS?=FP16
H_STATE_LEN?=256

SILENT?=1
CHECKSUM?=0
DEBUG?=0
DEBUG_STFT?=0


FREQ_CL?=370
FREQ_FC?=370
VOLTAGE?=800



#############################################
### 		Demo Settings
#############################################

QUANT_BITS?=FP16MIXED
MODEL_PREFIX=denoiser_dns
MODEL_FP16=1
MODEL_SQ8=1

NNTOOL_EXTRA_FLAGS=--use_lut_sigmoid --use_lut_tanh
NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_demo
GRU?=1
#endif 
DEMO?=0


ifeq ($(APP_MODE), 0)	
	DEMO 		= 0
	FLASH_TYPE 	= MRAM
	RAM_TYPE   	= DEFAULT
	FREQ_CL		= 240
	FREQ_FC		= 240
	VOLTAGE		= 650
endif

#############################################
### NN experiment setup
#############################################
#ifeq ($(APP_MODE), 3) 
ifeq ($(shell expr $(APP_MODE) \>= 2), 1)
	# select model
	ifeq ($(GRU), 0)
		MODEL_PREFIX = denoiser
	else
		MODEL_PREFIX = denoiser_GRU
	endif

	# select quantization level 
	ifeq 	'$(QUANT_BITS)' 'FP16'
		MODEL_FP16=1
		MODEL_SQ8=0
		ifeq ($(GRU), 0)
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16
		else
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16_gru
		endif

	else ifeq 	'$(QUANT_BITS)' 'FP16MIXED'
		MODEL_FP16=1
		MODEL_SQ8=1
		ifeq ($(GRU), 0)
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16_mixed
		else
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_fp16_gru_mixed
		endif

	else ifeq 	'$(QUANT_BITS)' '8'
		MODEL_SQ8=1
		MODEL_FP16=1
		ifeq ($(GRU), 0)
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_int8
		else
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_int8_gru 
		endif

	else ifeq 	'$(QUANT_BITS)' 'NE16'
		$(error NE16 Quantization mode is not yet fully supported)
		MODEL_NE16=1
		MODEL_SQ8=1
		ifeq ($(GRU), 0)
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_ne16
		else
			NNTOOL_SCRIPT=model/nntool_scripts/nntool_script_ne16_gru 
		endif

	else
		$(error Quantization mode is not recognized. Choose among 8, 16, FP16 or NE16)
	endif
endif

## Model Definition Parameters ##
BUILD_DIR?=BUILD
MODEL_SUFFIX = _$(QUANT_BITS)BIT
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)
TRAINED_MODEL_PATH=model
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
	FREQ_SFU?=370

	CLUSTER_STACK_SIZE=4096
	CLUSTER_SLAVE_STACK_SIZE=2048
	CLUSTER_NUM_CORES=8
	TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES))
	MODEL_L1_MEMORY?=$(shell expr 120000 \- $(TOTAL_STACK_SIZE))
	MODEL_L2_MEMORY?=1000000
	MODEL_L3_MEMORY?=8000000

else
	ifeq '$(TARGET_CHIP)' 'GAP9'
		FREQ_CL?=50
		FREQ_FC?=50

		CLUSTER_STACK_SIZE=4096
		CLUSTER_SLAVE_STACK_SIZE=2048
		CLUSTER_NUM_CORES=8
		TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES))
		MODEL_L1_MEMORY=$(shell expr 120000 \- $(TOTAL_STACK_SIZE))
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


PMSIS_OS=freertos



## File Definition ##
APP_SRCS += denoiser.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB) 
APP_SRCS += $(GAP_LIB_PATH)/wav_io/wavIO.c
APP_SRCS += BUILD_MODEL_STFT/RFFTKernels.c  

#C flags
APP_CFLAGS += -O2 -s -mno-memcpy -fno-tree-loop-distribute-patterns 

#include paths
APP_CFLAGS += -Icommon -I$(GAP_SDK_HOME)/libs/gap_lib/include/gaplib/
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) -I$(MODEL_BUILD) $(CNN_LIB_INCLUDE)
APP_CFLAGS += -I$(MFCC_GENERATOR) -I$(TILER_DSP_KERNEL_PATH) -I$(TILER_DSP_KERNEL_PATH)/LUT_Tables
APP_CFLAGS += -IBUILD_MODEL_STFT
APP_CFLAGS += -Isamples

#defines
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE) 
APP_CFLAGS += -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DFREQ_SFU=$(FREQ_SFU) -DVOLTAGE=$(VOLTAGE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DWAV_FILE=$(WAV_FILE) #-DWRITE_WAV #-DPRINT_AT_INPUT #-DPRINT_WAV 

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
APP_CFLAGS += -DDEMO=$(DEMO)
APP_CFLAGS += -DH_STATE_LEN=$(H_STATE_LEN)



APP_LDFLAGS		+= -lm


ifeq 	'$(QUANT_BITS)' 'FP16'
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	 '$(QUANT_BITS)' 'FP16MIXED'
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	'$(QUANT_BITS)' '8'
	APP_CFLAGS += -DSTD_FLOAT

else ifeq 	'$(QUANT_BITS)' 'NE16'
	APP_CFLAGS += -DSTD_FLOAT

else

endif

ifeq ($(platform), gvsoc)
	APP_CFLAGS += -DPERF
else
	ifeq ($(APP_MODE), 0)
	APP_CFLAGS += -DAUDIO_EVK 
	else
	APP_CFLAGS += -DPERF -DAUDIO_EVK 
	endif
endif

ifeq ($(SILENT), 1)
	APP_CFLAGS += -DSILENT
endif

ifeq ($(CHECKSUM), 1)
	APP_CFLAGS += -DCHECKSUM
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



$(TARGET_BUILD_DIR)/GraphINOUT_L2_Descr.c: $(CURDIR)/Graph.src
	mkdir -p $(@D)
	cd $(@D) && SFU -i $(CURDIR)/Graph.src -C

graph: $(TARGET_BUILD_DIR)/GraphINOUT_L2_Descr.c
	

# all depends on the model
all:: | model gen_fft_code graph

clean:: clean_model clean_fft_code
	rm -rf BUILD*

include common/model_rules.mk

# $(info APP_SRCS... $(APP_SRCS))
# $(info APP_CFLAGS... $(APP_CFLAGS))

include $(RULES_DIR)/pmsis_rules.mk
