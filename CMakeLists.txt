# Copyright (c) 2022 GreenWaves Technologies SAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of GreenWaves Technologies SAS nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.16)

###############################################################################
# Application related definitions
###############################################################################

set(TARGET_NAME         "denoiser")
set(MAIN                "denoiser.c")

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
setup_cmake()

###############################################################################
# Application's kconfig options
###############################################################################

set(WAV_FILE samples/sample_0000.wav)
set(APP_MSG_PREFIX "[${BoldRed}${TARGET_NAME}${ColourReset}]")
if (CONFIG_DEMO)
        set(DEMO                        1)
        set(IS_SFU                      1)
        set(IS_INPUT_STFT               0)
        set(DISABLE_NN_INFERENCE        0)
        set(STFT_FRAMES                 1)
        set(CONFIG_FREQUENCY_CLUSTER  200000000)
        set(CONFIG_FREQUENCY_CLUSTER  200000000)
        set(CONFIG_FREQUENCY_SFU      200000000)
        set(CONFIG_VOLTAGE            650)

        set(CONFIG_IO_TYPE_UART         y)

elseif(CONFIG_DENOISE_WAV)
        set(DEMO                        1)
        set(IS_SFU                      0)
        set(IS_INPUT_STFT               0)
        set(DISABLE_NN_INFERENCE        0)
        set(STFT_FRAMES                 1)

        set(CONFIG_IO_TYPE_SEMIHOSTING  y)
        set(WAV_FILE samples/real_samples/phone_call.wav)

elseif(CONFIG_DSP_WAV)
        set(DEMO                        0)
        set(IS_SFU                      0)
        set(IS_INPUT_STFT               0)
        set(DISABLE_NN_INFERENCE        1)
        set(STFT_FRAMES                 1)

        set(CONFIG_IO_TYPE_SEMIHOSTING  y)
        set(WAV_FILE samples/dataset/noisy/p232_050.wav)

elseif(CONFIG_NN_TEST)
        set(DEMO                        0)
        set(IS_SFU                      0)
        set(IS_INPUT_STFT               1)
        set(DISABLE_NN_INFERENCE        0)
        set(STFT_FRAMES                 1)

        set(CONFIG_IO_TYPE_SEMIHOSTING  y)

endif()
set(H_STATE_LEN          256)
set(FRAME_SIZE           400)
set(FRAME_STEP           100)
set(FRAME_NFFT           512)
set(NUM_FRAME_OVERLAP      3)
set(SAMPLING_FREQ      16000)
set(AT_INPUT_WIDTH       257)
set(AT_INPUT_HEIGHT        1)


if (CONFIG_FP16MIXED)
        message(STATUS "${APP_MSG_PREFIX} Quantization Type: FP16MIXED")
        set(CONFIG_NNTOOL_SCRIPT_PATH model/nntool_scripts/nntool_script_fp16_mixed)

elseif (CONFIG_FP16)
        message(STATUS "${APP_MSG_PREFIX} Quantization Type: FP16")
        set(CONFIG_NNTOOL_SCRIPT_PATH model/nntool_scripts/nntool_script_fp16)

elseif (CONFIG_8)
        message(STATUS "${APP_MSG_PREFIX} Quantization Type: SQ8")
        set(CONFIG_NNTOOL_SCRIPT_PATH model/nntool_scripts/nntool_script_int8)

elseif (CONFIG_NE16)
        message(STATUS "${APP_MSG_PREFIX} Quantization Type: NE16")
        set(CONFIG_NNTOOL_SCRIPT_PATH model/nntool_scripts/nntool_script_ne16)

endif()

if (${DEMO})
        message(STATUS "${APP_MSG_PREFIX} Model Type: DEMO")
        set(TARGET_NAME "denoiser_dns")
        set(CONFIG_NNTOOL_SCRIPT_PATH ${CONFIG_NNTOOL_SCRIPT_PATH}_demo)
        set(GRU_FLAG 1)

elseif (${CONFIG_GRU})
        message(STATUS "${APP_MSG_PREFIX} Model Type: GRU")
        set(TARGET_NAME "denoiser_GRU")
        set(CONFIG_NNTOOL_SCRIPT_PATH ${CONFIG_NNTOOL_SCRIPT_PATH}_gru)
        set(GRU_FLAG 1)
else()
        message(STATUS "${APP_MSG_PREFIX} Model Type: LSTM")
        set(GRU_FLAG 0)
endif()
message(STATUS "${APP_MSG_PREFIX} ${CONFIG_NNTOOL_SCRIPT_PATH}")

set(COMMON_MODEL_BUILDDIR ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR})
set(CONFIG_NNTOOL_MODEL_PATH model/${TARGET_NAME}.onnx)

if (${CONFIG_MODEL_FLASH_TYPE_MRAM})
        set(EXEC_FROM_FLASH true)
else()
        set(EXEC_FROM_FLASH false)
endif()

list(APPEND NNTOOLSCRIPT_FLAGS "-DCOLLECT_STATS_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/model/nntool_scripts/collect_stats.py;-DSAMPLES_QUANT=${CMAKE_CURRENT_SOURCE_DIR}/samples/quant/;-DEXEC_FROM_FLASH=${EXEC_FROM_FLASH};-DMODEL_BUILD=${COMMON_MODEL_BUILDDIR};-DGRU=${GRU_FLAG};-DH_STATE_LEN=${H_STATE_LEN};")

###############################################################################
# NNTool Variables 
###############################################################################

autotiler_define_gen_names(${TARGET_NAME})
nntool_setup(${TARGET_NAME})
autotiler_setup(${TARGET_NAME})

###############################################################################
# Custom Commands to build STFT
###############################################################################

set(FFT_BUILD_DIR       ${CMAKE_BINARY_DIR}/BUILD_MODEL_STFT)
set(FFT_KERNEL_C        ${FFT_BUILD_DIR}/RFFTKernels.c)
set(WIN_LUT             ${FFT_BUILD_DIR}/WinLUT_f16.def)
set(FFT_MODEL_GEN       ${FFT_BUILD_DIR}/GenSTFT)
set(FFT_SRCG            $ENV{TILER_DSP_GENERATOR_PATH}/DSP_Generators.c)

set(FRAME_SIZE          400)
set(FRAME_STEP          100)
set(FRAME_NFFT          512)

message(STATUS "${APP_MSG_PREFIX} STFT: FRAME_SIZE=${FRAME_SIZE} FRAME_STEP=${FRAME_STEP} FRAME_NFFT=${FRAME_NFFT}")
add_custom_command(
        COMMENT             "Creating ${FFT_BUILD_DIR} directory"
        OUTPUT              ${FFT_BUILD_DIR}
        COMMAND             ${CMAKE_COMMAND} -E make_directory ${FFT_BUILD_DIR})
add_custom_target(stft_buildir DEPENDS ${FFT_BUILD_DIR})

add_custom_command(
        COMMENT             "Generating LUT in ${WIN_LUT}"
        OUTPUT              ${WIN_LUT}
        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR}
        DEPENDS             ${FFT_BUILD_DIR}
        COMMAND             python $ENV{TILER_MFCC_GEN_LUT_SCRIPT} --fft_lut_file ${WIN_LUT} --win_func "hanning" --dtype "float16" --frame_size ${FRAME_SIZE} --frame_step ${FRAME_STEP} --n_fft ${FRAME_NFFT} --gen_inv
)

add_custom_command(
        OUTPUT              ${FFT_MODEL_GEN}
        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR}
        DEPENDS             ${WIN_LUT}
        COMMAND             gcc -g -o ${FFT_MODEL_GEN} -I. -I$ENV{TILER_DSP_GENERATOR_PATH} -I${CMAKE_BINARY_DIR} ${MODEL_GEN_INCS} ${CMAKE_SOURCE_DIR}/model/STFTModel.c ${FFT_SRCG} $ENV{TILER_LIB} -DFRAME_SIZE=${FRAME_SIZE} -DFRAME_STEP=${FRAME_STEP} -DN_FFT=${FRAME_NFFT} -lSDL2 -lSDL2_ttf
)

add_custom_command(
        COMMENT             "Generating SRCS in ${FFT_KERNEL_C}"
        OUTPUT              ${FFT_KERNEL_C}
        WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
        DEPENDS             ${FFT_MODEL_GEN}
        COMMAND             ${FFT_MODEL_GEN} -o ${FFT_BUILD_DIR} -c ${FFT_BUILD_DIR} ${MODEL_GEN_EXTRA_FLAGS}
)
add_custom_target(gen_stft_code DEPENDS ${FFT_KERNEL_C})
add_custom_target(clean_stft_code COMMAND rm -rf ${FFT_KERNEL_C} ${FFT_MODEL_GEN})

###############################################################################
# Custom Commands to build SFU Graph
###############################################################################

set(SFU_BUILDDIR        ${CMAKE_BINARY_DIR}/BUILD_SFU)
set(SFU_KERNEL_C        ${SFU_BUILDDIR}/GraphINOUT_L2_Descr.c)
set(SFU_GRAPH           ${CMAKE_CURRENT_SOURCE_DIR}/Graph.src)

add_custom_command(
        COMMENT             "Creating ${SFU_BUILDDIR} directory"
        OUTPUT              ${SFU_BUILDDIR}
        COMMAND             ${CMAKE_COMMAND} -E make_directory ${SFU_BUILDDIR})
add_custom_target(sfu_buildir DEPENDS ${SFU_BUILDDIR})

add_custom_command(
        COMMENT                 "Generating SFU Graph ${SFU_KERNEL_C}"
        OUTPUT                  ${SFU_KERNEL_C}
        WORKING_DIRECTORY       ${SFU_BUILDDIR}
        DEPENDS                 ${SFU_GRAPH} ${SFU_BUILDDIR}
        COMMAND                 SFU -i ${CMAKE_SOURCE_DIR}/Graph.src -C
)
add_custom_target(sfu_graph ${SFU_KERNEL_C})

list(APPEND TARGET_SRCS ${SFU_KERNEL_C}
                        $ENV{SFU_RUNTIME}/SFU_RT.c
                        ${CMAKE_SOURCE_DIR}/dac.c)
list(APPEND TARGET_INCS -I${SFU_BUILDDIR}
                        -I$ENV{SFU_RUNTIME}/include)


###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${COMMON_MODEL_BUILDDIR}/${KERNEL_C}
                        ${FFT_KERNEL_C}
)

list(APPEND TARGET_INCS -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I${GAP_SDK_HOME}/libs/gap_lib/include
                          ${CNN_LIB_INCLUDE}
                        -I${COMMON_MODEL_BUILDDIR}
                        -I${FFT_BUILD_DIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR}
                        -I${GAP_SDK_HOME}/utils/power_meas_utils
                        -I${CMAKE_CURRENT_SOURCE_DIR}/samples
)

list(APPEND TARGET_CFLAGS   -gdwarf-2
                            -gstrict-dwarf
                            -O3
                            -mno-memcpy
                            -fno-tree-loop-distribute-patterns)

math(EXPR FREQ_CL_MHZ "${CONFIG_FREQUENCY_CLUSTER}/1000/1000")
math(EXPR FREQ_FC_MHZ "${CONFIG_FREQUENCY_FC}/1000/1000")
math(EXPR FREQ_SFU_MHZ "${CONFIG_FREQUENCY_SFU}/1000/1000")
list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DAT_MODEL_PREFIX=${MODEL_NAME}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ_CL_MHZ}
                                -DFREQ_FC=${FREQ_FC_MHZ}
                                -DFREQ_SFU=${FREQ_SFU_MHZ}
                                -DVOLTAGE=${CONFIG_VOLTAGE}
                                -DWAV_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${WAV_FILE}
                                -DIS_SFU=${IS_SFU}
                                -DIS_INPUT_STFT=${IS_INPUT_STFT}
                                -DSTFT_FRAMES=${STFT_FRAMES}
                                -DFRAME_SIZE=${FRAME_SIZE}
                                -DFRAME_STEP=${FRAME_STEP}
                                -DFRAME_NFFT=${FRAME_NFFT}
                                -DNUM_FRAME_OVERLAP=${NUM_FRAME_OVERLAP}
                                -DSAMPLING_FREQ=${SAMPLING_FREQ}
                                -DAT_INPUT_WIDTH=${AT_INPUT_WIDTH}
                                -DAT_INPUT_HEIGHT=${AT_INPUT_HEIGHT}
                                -DMAX_L2_BUFFER=${CONFIG_MODEL_L2_MEMORY}
                                -DDEMO=${DEMO}
                                -DH_STATE_LEN=${H_STATE_LEN}
                                -DBUILD_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
                                -DSTD_FLOAT
)
if (${GRU_FLAG} EQUAL 1)
list(APPEND TARGET_PREPROCESSOR -DGRU)
endif()
if (${DISABLE_NN_INFERENCE})
list(APPEND TARGET_PREPROCESSOR -DDISABLE_NN_INFERENCE)
endif()
if (${CONFIG_SILENT})
list(APPEND TARGET_PREPROCESSOR -DSILENT)
endif()
if (${CONFIG_CHECKSUM})
list(APPEND TARGET_PREPROCESSOR -DCHECKSUM)
endif()


project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})

target_compile_options(${TARGET_NAME} PUBLIC    ${TARGET_CFLAGS} 
                                                ${TARGET_INCS}
                                                ${TARGET_PREPROCESSOR})
message(STATUS ${TARGET_PREPROCESSOR})

###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})

