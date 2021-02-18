#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "RNN_Generators_SQ8.h"
#include "ResizeGenerator.h"

#include "nntool_extra_generators.h"





void denoiserModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 4, "nntool_extra_kernels.h", "CNN_BasicKernels_SQ8.h", "denoiser.h", "ResizeBasicKernels.h");
    SetGeneratedFilesNames("denoiserKernels.c", "denoiserKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "denoiser_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "denoiser_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "denoiser_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "denoiser_L3_Flash", "denoiser_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    Load_RNN_SQ8_Library();
    LoadResizeLibrary();

    LoadNNTools_Extra_Library();

    // generator for Conv_0_fusion
    CNN_ConvolutionPoolAct_SQ8("S4_Conv2d_16x1x1x4_Relu", 0, 4, 1, 1, 16, 1088, 1,
        KOP_CONV, 4, 1, 1, 1, 4, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Conv_2_fusion
    CNN_GenControl_T gen_ctrl_S9_Conv2d_16x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S9_Conv2d_16x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_16x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S9_Conv2d_16x16x1x1_Relu", &gen_ctrl_S9_Conv2d_16x16x1x1_Relu, 4, 1, 16, 16, 272, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Slice_409
    CNN_Copy("S12_Op_Slice_409", 0, 4096);
    // generator for Conv_4_fusion
    CNN_ConvolutionPoolAct_SQ8("S15_Conv2d_32x16x1x4_Relu", 0, 4, 1, 16, 32, 272, 1,
        KOP_CONV, 4, 1, 1, 1, 4, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Conv_6_fusion
    CNN_GenControl_T gen_ctrl_S20_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S20_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S20_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S20_Conv2d_32x32x1x1_Relu", &gen_ctrl_S20_Conv2d_32x32x1x1_Relu, 4, 1, 32, 32, 68, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Slice_350
    CNN_Copy("S23_Op_Slice_350", 0, 2048);
    // generator for Conv_8_fusion
    CNN_ConvolutionPoolAct_SQ8("S26_Conv2d_64x32x1x4_Relu", 0, 4, 1, 32, 64, 68, 1,
        KOP_CONV, 4, 1, 1, 1, 4, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Conv_10_fusion
    CNN_GenControl_T gen_ctrl_S31_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S31_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S31_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S31_Conv2d_64x64x1x1_Relu", &gen_ctrl_S31_Conv2d_64x64x1x1_Relu, 4, 1, 64, 64, 17, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Slice_291
    CNN_Copy("S34_Op_Slice_291", 0, 1024);
    // generator for Conv_12_fusion
    CNN_ConvolutionPoolAct_SQ8("S37_Conv2d_128x64x1x4_Relu", 0, 4, 1, 64, 128, 17, 1,
        KOP_CONV, 4, 1, 1, 1, 4, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Conv_14_fusion
    CNN_GenControl_T gen_ctrl_S42_Conv2d_128x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S42_Conv2d_128x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S42_Conv2d_128x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S42_Conv2d_128x128x1x1_Relu", &gen_ctrl_S42_Conv2d_128x128x1x1_Relu, 4, 1, 128, 128, 4, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Slice_232
    CNN_Copy("S45_Op_Slice_232", 0, 512);
    // generator for Conv_16_fusion
    CNN_ConvolutionPoolAct_SQ8("S48_Conv2d_256x128x1x4_Relu", 0, 4, 1, 128, 256, 4, 1,
        KOP_CONV, 4, 1, 1, 1, 4, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Conv_18_fusion
    CNN_GenControl_T gen_ctrl_S53_Conv2d_256x256x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S53_Conv2d_256x256x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S53_Conv2d_256x256x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S53_Conv2d_256x256x1x1_Relu", &gen_ctrl_S53_Conv2d_256x256x1x1_Relu, 4, 1, 256, 256, 1, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Slice_173
    CNN_Copy("S55_Op_Slice_173", 0, 256);
    // generator for LSTM_95
    CNN_GenControl_T gen_ctrl_S71_Op_LSTM_95;
    CNN_InitGenCtrl(&gen_ctrl_S71_Op_LSTM_95);
    CNN_SetGenCtrl(&gen_ctrl_S71_Op_LSTM_95, "RNN_USE_HARDACT", AT_OPT_VAL(0));
    CNN_SetGenCtrl(&gen_ctrl_S71_Op_LSTM_95, "RNN_SAME_INOUT_SCALE", AT_OPT_VAL(0));
    LSTM_Stack_SQ8("S71_Op_LSTM_95", &gen_ctrl_S71_Op_LSTM_95, 4, 1, 1, 1, 1, 256, 256, 0, 0);
    // generator for LSTM_161
    CNN_GenControl_T gen_ctrl_S88_Op_LSTM_161;
    CNN_InitGenCtrl(&gen_ctrl_S88_Op_LSTM_161);
    CNN_SetGenCtrl(&gen_ctrl_S88_Op_LSTM_161, "RNN_USE_HARDACT", AT_OPT_VAL(0));
    CNN_SetGenCtrl(&gen_ctrl_S88_Op_LSTM_161, "RNN_SAME_INOUT_SCALE", AT_OPT_VAL(1));
    LSTM_Stack_SQ8("S88_Op_LSTM_161", &gen_ctrl_S88_Op_LSTM_161, 4, 1, 1, 1, 1, 256, 256, 0, 0);
    // generator for Add_174
    CNN_MatAddAct_SQ8("S92_MatAdd_256x1", 0, 1, 256, 1, KOP_MATADD, KOP_NONE);
    // generator for Conv_175_fusion
    CNN_GenControl_T gen_ctrl_S96_Conv2d_256x256x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S96_Conv2d_256x256x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S96_Conv2d_256x256x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S96_Conv2d_256x256x1x1_Relu", &gen_ctrl_S96_Conv2d_256x256x1x1_Relu, 4, 1, 256, 256, 1, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Resize_196
    GenerateResizeMultiChannel("S99_Op_Resize_196", 1, 1, 4, 1, 256, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    // generator for Conv_221_fusion
    CNN_GenControl_T gen_ctrl_S104_Conv2d_128x256x1x4_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S104_Conv2d_128x256x1x4_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S104_Conv2d_128x256x1x4_Relu, "PADTYPE", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S104_Conv2d_128x256x1x4_Relu", &gen_ctrl_S104_Conv2d_128x256x1x4_Relu, 4, 1, 256, 128, 4, 1,
        KOP_CONV, 4, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Add_233
    CNN_MatAddAct_SQ8("S106_MatAdd_128x4", 0, 1, 128, 4, KOP_MATADD, KOP_NONE);
    // generator for Conv_234_fusion
    CNN_GenControl_T gen_ctrl_S110_Conv2d_128x128x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S110_Conv2d_128x128x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S110_Conv2d_128x128x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S110_Conv2d_128x128x1x1_Relu", &gen_ctrl_S110_Conv2d_128x128x1x1_Relu, 4, 1, 128, 128, 4, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Resize_255
    GenerateResizeMultiChannel("S113_Op_Resize_255", 4, 1, 16, 1, 128, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    // generator for Conv_280_fusion
    CNN_GenControl_T gen_ctrl_S118_Conv2d_64x128x1x4_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S118_Conv2d_64x128x1x4_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S118_Conv2d_64x128x1x4_Relu, "PADTYPE", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S118_Conv2d_64x128x1x4_Relu", &gen_ctrl_S118_Conv2d_64x128x1x4_Relu, 4, 1, 128, 64, 16, 1,
        KOP_CONV, 4, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Add_292
    CNN_MatAddAct_SQ8("S120_MatAdd_64x16", 0, 1, 64, 16, KOP_MATADD, KOP_NONE);
    // generator for Conv_293_fusion
    CNN_GenControl_T gen_ctrl_S124_Conv2d_64x64x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S124_Conv2d_64x64x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S124_Conv2d_64x64x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S124_Conv2d_64x64x1x1_Relu", &gen_ctrl_S124_Conv2d_64x64x1x1_Relu, 4, 1, 64, 64, 16, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Resize_314
    GenerateResizeMultiChannel("S127_Op_Resize_314", 16, 1, 64, 1, 64, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    // generator for Conv_339_fusion
    CNN_GenControl_T gen_ctrl_S132_Conv2d_32x64x1x4_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S132_Conv2d_32x64x1x4_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S132_Conv2d_32x64x1x4_Relu, "PADTYPE", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S132_Conv2d_32x64x1x4_Relu", &gen_ctrl_S132_Conv2d_32x64x1x4_Relu, 4, 1, 64, 32, 64, 1,
        KOP_CONV, 4, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Add_351
    CNN_MatAddAct_SQ8("S134_MatAdd_32x64", 0, 1, 32, 64, KOP_MATADD, KOP_NONE);
    // generator for Conv_352_fusion
    CNN_GenControl_T gen_ctrl_S138_Conv2d_32x32x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S138_Conv2d_32x32x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S138_Conv2d_32x32x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S138_Conv2d_32x32x1x1_Relu", &gen_ctrl_S138_Conv2d_32x32x1x1_Relu, 4, 1, 32, 32, 64, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Resize_373
    GenerateResizeMultiChannel("S141_Op_Resize_373", 64, 1, 256, 1, 32, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    // generator for Conv_398_fusion
    CNN_GenControl_T gen_ctrl_S146_Conv2d_16x32x1x4_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S146_Conv2d_16x32x1x4_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S146_Conv2d_16x32x1x4_Relu, "PADTYPE", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S146_Conv2d_16x32x1x4_Relu", &gen_ctrl_S146_Conv2d_16x32x1x4_Relu, 4, 1, 32, 16, 256, 1,
        KOP_CONV, 4, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Add_410
    CNN_MatAddAct_SQ8("S148_MatAdd_16x256", 0, 1, 16, 256, KOP_MATADD, KOP_NONE);
    // generator for Conv_411_fusion
    CNN_GenControl_T gen_ctrl_S152_Conv2d_16x16x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S152_Conv2d_16x16x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S152_Conv2d_16x16x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S152_Conv2d_16x16x1x1_Relu", &gen_ctrl_S152_Conv2d_16x16x1x1_Relu, 4, 1, 16, 16, 256, 1,
        KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for Resize_432
    GenerateResizeMultiChannel("S155_Op_Resize_432", 256, 1, 1024, 1, 16, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    // generator for Conv_457
    CNN_GenControl_T gen_ctrl_S160_Conv2d_1x16x1x4;
    CNN_InitGenCtrl(&gen_ctrl_S160_Conv2d_1x16x1x4);
    CNN_SetGenCtrl(&gen_ctrl_S160_Conv2d_1x16x1x4, "PADTYPE", AT_OPT_VAL(1));
    CNN_ConvolutionPoolAct_SQ8("S160_Conv2d_1x16x1x4", &gen_ctrl_S160_Conv2d_1x16x1x4, 4, 1, 16, 1, 1024, 1,
        KOP_CONV, 4, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_NONE);

#define GRAPH
#ifdef GRAPH
    CreateGraph("denoiserCNN",
        /* Arguments either passed or globals */
            CArgs(130,
                TCArgInfo("signed char * __restrict__", "S0_Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "S2_Conv_0_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S2_Conv_0_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S3_Constant_encoder.0.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S3_Constant_encoder.0.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S4_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S4_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S4_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Conv_2_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S7_Conv_2_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S8_Constant_encoder.0.2.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S8_Constant_encoder.0.2.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S9_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S9_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S9_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Conv_4_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S13_Conv_4_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S14_Constant_encoder.1.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S14_Constant_encoder.1.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S15_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S15_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S15_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S15_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S15_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S15_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S18_Conv_6_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S18_Conv_6_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S19_Constant_encoder.1.2.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S19_Constant_encoder.1.2.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S20_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S20_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S20_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S20_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S20_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S24_Conv_8_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S24_Conv_8_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S25_Constant_encoder.2.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S25_Constant_encoder.2.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S26_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S26_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S26_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S26_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S26_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S29_Conv_10_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S29_Conv_10_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S30_Constant_encoder.2.2.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S30_Constant_encoder.2.2.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S31_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S31_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S31_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S31_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S31_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S31_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S35_Conv_12_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S35_Conv_12_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S36_Constant_encoder.3.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S36_Constant_encoder.3.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S37_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S37_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S37_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S37_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S37_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S37_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S40_Conv_14_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S40_Conv_14_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S41_Constant_encoder.3.2.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S41_Constant_encoder.3.2.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S42_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S42_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S42_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S42_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S42_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S42_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S46_Conv_16_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S46_Conv_16_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S47_Constant_encoder.4.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S47_Constant_encoder.4.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S48_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S48_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S48_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S48_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S48_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S48_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Conv_18_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S51_Conv_18_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S52_Constant_encoder.4.2.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S52_Constant_encoder.4.2.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S53_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S53_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S53_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S53_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S53_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S53_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S61_Lstm_95_r_2_i_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S61_Lstm_95_r_2_i_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S62_Lstm_95_r_2_f_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S62_Lstm_95_r_2_f_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S63_Lstm_95_r_2_c_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S63_Lstm_95_r_2_c_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S64_Lstm_95_r_2_o_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S64_Lstm_95_r_2_o_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S65_Lstm_95_i_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S65_Lstm_95_i_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S66_Lstm_95_f_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S66_Lstm_95_f_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S67_Lstm_95_c_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S67_Lstm_95_c_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S68_Lstm_95_o_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S68_Lstm_95_o_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("char", "Reset", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_UNDEF, 0),
                TCArgInfo("signed char * __restrict__", "S69_Lstm_95_i_state", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "S70_Lstm_95_c_state", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                // f_scale: 73 f_scale_n: 13 i_scale: 73 i_scale_n: 13 c_scale: 73 c_scale_n: 13 o_scale: 73 o_scale_n: 13 cin_scale: 235 cin_scale_n: 18 cout_scale: 35 cout_scale_n: 10 out_scale: 7 out_scale_n: 10 int_q: 12 A0: 24576 B0: 12288 C0: 683 f_scale: 255 f_scale_n: 4 i_scale: 255 i_scale_n: 4 c_scale: 255 c_scale_n: 4 o_scale: 255 o_scale_n: 4
                TCArgInfo("signed char * __restrict__", "S71_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S71_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S78_Lstm_161_r_2_i_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S78_Lstm_161_r_2_i_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S79_Lstm_161_r_2_f_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S79_Lstm_161_r_2_f_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S80_Lstm_161_r_2_c_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S80_Lstm_161_r_2_c_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S81_Lstm_161_r_2_o_w", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S81_Lstm_161_r_2_o_w.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S82_Lstm_161_i_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S82_Lstm_161_i_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S83_Lstm_161_f_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S83_Lstm_161_f_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S84_Lstm_161_c_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S84_Lstm_161_c_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed int * __restrict__", "S85_Lstm_161_o_b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S85_Lstm_161_o_b.tensor", 1, 1, 4, 0)),
                TCArgInfo("signed char * __restrict__", "S86_Lstm_161_i_state", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "S87_Lstm_161_c_state", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                // f_scale: 73 f_scale_n: 13 i_scale: 73 i_scale_n: 13 c_scale: 73 c_scale_n: 13 o_scale: 73 o_scale_n: 13 cin_scale: 69 cin_scale_n: 19 cout_scale: 119 cout_scale_n: 9 out_scale: 7 out_scale_n: 10 int_q: 12 A0: 24576 B0: 12288 C0: 683 f_scale: 1 f_scale_n: 0 i_scale: 1 i_scale_n: 0 c_scale: 1 c_scale_n: 0 o_scale: 1 o_scale_n: 0
                TCArgInfo("signed char * __restrict__", "S88_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S88_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 255 In1ScaleN: 4 OutScale: 129 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S92_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S92_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S94_Conv_175_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S94_Conv_175_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S95_Constant_decoder.0.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S95_Constant_decoder.0.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S96_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S96_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S96_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S96_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S96_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S96_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S102_Conv_221_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S102_Conv_221_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S103_Constant_decoder.0.4.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S103_Constant_decoder.0.4.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S104_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S104_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S104_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S104_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S104_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S104_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 165 In1ScaleN: 6 OutScale: 193 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S106_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S106_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S108_Conv_234_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S108_Conv_234_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S109_Constant_decoder.1.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S109_Constant_decoder.1.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S110_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S110_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S110_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S110_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S110_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S110_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S116_Conv_280_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S116_Conv_280_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S117_Constant_decoder.1.4.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S117_Constant_decoder.1.4.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S118_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S118_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S118_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S118_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S118_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S118_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 147 In1ScaleN: 4 OutScale: 221 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S120_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S120_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S122_Conv_293_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S122_Conv_293_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S123_Constant_decoder.2.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S123_Constant_decoder.2.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S124_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S124_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S124_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S124_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S124_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S124_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S130_Conv_339_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S130_Conv_339_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S131_Constant_decoder.2.4.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S131_Constant_decoder.2.4.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S132_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S132_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S132_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S132_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S132_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 65 In1ScaleN: 2 OutScale: 63 OutScaleN: 10
                TCArgInfo("signed char * __restrict__", "S134_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S134_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S136_Conv_352_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S136_Conv_352_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S137_Constant_decoder.3.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S137_Constant_decoder.3.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S138_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S138_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S138_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S138_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S138_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S138_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S144_Conv_398_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S144_Conv_398_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S145_Constant_decoder.3.4.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S145_Constant_decoder.3.4.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S146_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S146_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S146_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S146_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S146_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S146_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 177 In1ScaleN: 3 OutScale: 185 OutScaleN: 12
                TCArgInfo("signed char * __restrict__", "S148_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S148_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S150_Conv_411_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S150_Conv_411_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S151_Constant_decoder.4.0.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S151_Constant_decoder.4.0.bias.tensor", 1, 1, 4, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S152_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S152_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S152_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S152_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S152_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S152_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S158_Conv_457_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S158_Conv_457_weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S159_Constant_decoder.4.4.bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S159_Constant_decoder.4.4.bias.tensor", 1, 1, 4, 0)),
                // BiasQ: 0
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S160_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S160_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_8BIT/tensors/S160_Mul_shift.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(36,
            TCArgInfo("signed char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S15_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S20_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S23_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S26_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S31_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S34_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S42_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S45_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S48_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S53_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S55_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S71_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S88_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S92_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S96_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S99_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S104_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S106_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S110_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S113_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S118_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S120_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S124_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S127_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S132_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S134_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S138_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S146_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S148_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S152_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S155_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S4_Conv2d_16x1x1x4_Relu inq -4912.00<(i8-0.00)*38.37500000<4873.62 weightsq chan<(i8-0.00)*chan<chan outq -3518.92<(i8-0.00)*27.49154472<3491.43 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S4_Conv2d_16x1x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S0_Input_1", 0), GNodeArg(GNA_IN, "S2_Conv_0_weights", 0), GNodeArg(GNA_IN, "S3_Constant_encoder.0.0.bias", 0), GNodeArg(GNA_OUT, "S4_Output", 0), GNodeArg(GNA_IN, "S4_Mul_scale", 0), GNodeArg(GNA_IN, "S4_Mul_shift", 0), GNodeArg(GNA_IN, "S4_Infos", 0)));
    // Node S9_Conv2d_16x16x1x1_Relu inq -3518.92<(i8-0.00)*27.49154472<3491.43 weightsq chan<(i8-0.00)*chan<chan outq -1796.44<(i8-0.00)*14.03465652<1782.40 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_16x16x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S4_Output", 0), GNodeArg(GNA_IN, "S7_Conv_2_weights", 0), GNodeArg(GNA_IN, "S8_Constant_encoder.0.2.bias", 0), GNodeArg(GNA_OUT, "S9_Output", 0), GNodeArg(GNA_IN, "S9_Mul_scale", 0), GNodeArg(GNA_IN, "S9_Mul_shift", 0), GNodeArg(GNA_IN, "S9_Infos", 0)));
    // Node S15_Conv2d_32x16x1x4_Relu inq -1796.44<(i8-0.00)*14.03465652<1782.40 weightsq chan<(i8-0.00)*chan<chan outq -819.93<(i8-0.00)*6.40572929<813.53 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S15_Conv2d_32x16x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S9_Output", 0), GNodeArg(GNA_IN, "S13_Conv_4_weights", 0), GNodeArg(GNA_IN, "S14_Constant_encoder.1.0.bias", 0), GNodeArg(GNA_OUT, "S15_Output", 0), GNodeArg(GNA_IN, "S15_Mul_scale", 0), GNodeArg(GNA_IN, "S15_Mul_shift", 0), GNodeArg(GNA_IN, "S15_Infos", 0)));
    // Node S20_Conv2d_32x32x1x1_Relu inq -819.93<(i8-0.00)*6.40572929<813.53 weightsq chan<(i8-0.00)*chan<chan outq -252.85<(i8-0.00)*1.97536814<250.87 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S20_Conv2d_32x32x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S15_Output", 0), GNodeArg(GNA_IN, "S18_Conv_6_weights", 0), GNodeArg(GNA_IN, "S19_Constant_encoder.1.2.bias", 0), GNodeArg(GNA_OUT, "S20_Output", 0), GNodeArg(GNA_IN, "S20_Mul_scale", 0), GNodeArg(GNA_IN, "S20_Mul_shift", 0), GNodeArg(GNA_IN, "S20_Infos", 0)));
    // Node S26_Conv2d_64x32x1x4_Relu inq -252.85<(i8-0.00)*1.97536814<250.87 weightsq chan<(i8-0.00)*chan<chan outq -134.40<(i8-0.00)*1.05000162<133.35 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S26_Conv2d_64x32x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S20_Output", 0), GNodeArg(GNA_IN, "S24_Conv_8_weights", 0), GNodeArg(GNA_IN, "S25_Constant_encoder.2.0.bias", 0), GNodeArg(GNA_OUT, "S26_Output", 0), GNodeArg(GNA_IN, "S26_Mul_scale", 0), GNodeArg(GNA_IN, "S26_Mul_shift", 0), GNodeArg(GNA_IN, "S26_Infos", 0)));
    // Node S31_Conv2d_64x64x1x1_Relu inq -134.40<(i8-0.00)*1.05000162<133.35 weightsq chan<(i8-0.00)*chan<chan outq -66.33<(i8-0.00)*0.51818442<65.81 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S31_Conv2d_64x64x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S26_Output", 0), GNodeArg(GNA_IN, "S29_Conv_10_weights", 0), GNodeArg(GNA_IN, "S30_Constant_encoder.2.2.bias", 0), GNodeArg(GNA_OUT, "S31_Output", 0), GNodeArg(GNA_IN, "S31_Mul_scale", 0), GNodeArg(GNA_IN, "S31_Mul_shift", 0), GNodeArg(GNA_IN, "S31_Infos", 0)));
    // Node S37_Conv2d_128x64x1x4_Relu inq -66.33<(i8-0.00)*0.51818442<65.81 weightsq chan<(i8-0.00)*chan<chan outq -39.64<(i8-0.00)*0.30966252<39.33 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S37_Conv2d_128x64x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S31_Output", 0), GNodeArg(GNA_IN, "S35_Conv_12_weights", 0), GNodeArg(GNA_IN, "S36_Constant_encoder.3.0.bias", 0), GNodeArg(GNA_OUT, "S37_Output", 0), GNodeArg(GNA_IN, "S37_Mul_scale", 0), GNodeArg(GNA_IN, "S37_Mul_shift", 0), GNodeArg(GNA_IN, "S37_Infos", 0)));
    // Node S42_Conv2d_128x128x1x1_Relu inq -39.64<(i8-0.00)*0.30966252<39.33 weightsq chan<(i8-0.00)*chan<chan outq -18.08<(i8-0.00)*0.14123094<17.94 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S42_Conv2d_128x128x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S37_Output", 0), GNodeArg(GNA_IN, "S40_Conv_14_weights", 0), GNodeArg(GNA_IN, "S41_Constant_encoder.3.2.bias", 0), GNodeArg(GNA_OUT, "S42_Output", 0), GNodeArg(GNA_IN, "S42_Mul_scale", 0), GNodeArg(GNA_IN, "S42_Mul_shift", 0), GNodeArg(GNA_IN, "S42_Infos", 0)));
    // Node S48_Conv2d_256x128x1x4_Relu inq -18.08<(i8-0.00)*0.14123094<17.94 weightsq chan<(i8-0.00)*chan<chan outq -15.05<(i8-0.00)*0.11759508<14.93 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S48_Conv2d_256x128x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S42_Output", 0), GNodeArg(GNA_IN, "S46_Conv_16_weights", 0), GNodeArg(GNA_IN, "S47_Constant_encoder.4.0.bias", 0), GNodeArg(GNA_OUT, "S48_Output", 0), GNodeArg(GNA_IN, "S48_Mul_scale", 0), GNodeArg(GNA_IN, "S48_Mul_shift", 0), GNodeArg(GNA_IN, "S48_Infos", 0)));
    // Node S53_Conv2d_256x256x1x1_Relu inq -15.05<(i8-0.00)*0.11759508<14.93 weightsq chan<(i8-0.00)*chan<chan outq -9.09<(i8-0.00)*0.07102456<9.02 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S53_Conv2d_256x256x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S48_Output", 0), GNodeArg(GNA_IN, "S51_Conv_18_weights", 0), GNodeArg(GNA_IN, "S52_Constant_encoder.4.2.bias", 0), GNodeArg(GNA_OUT, "S53_Output", 0), GNodeArg(GNA_IN, "S53_Mul_scale", 0), GNodeArg(GNA_IN, "S53_Mul_shift", 0), GNodeArg(GNA_IN, "S53_Infos", 0)));
    // Node S71_Op_LSTM_95 inq -9.09<(i8-0.00)*0.07102456<9.02 outq -0.57<(i8-0.00)*0.00445706<0.57
    AddNode("S71_Op_LSTM_95", Bindings(18, GNodeArg(GNA_INOUT, "S70_Lstm_95_c_state", "S70_Lstm_95_c_state"), GNodeArg(GNA_INOUT, "S69_Lstm_95_i_state", "S69_Lstm_95_i_state"), AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, GNodeArg(GNA_IN, "S53_Output", 0), GNodeArg(GNA_IN, "S62_Lstm_95_r_2_f_w", 0), GNodeArg(GNA_IN, "S66_Lstm_95_f_b", 0), GNodeArg(GNA_IN, "S61_Lstm_95_r_2_i_w", 0), GNodeArg(GNA_IN, "S65_Lstm_95_i_b", 0), GNodeArg(GNA_IN, "S63_Lstm_95_r_2_c_w", 0), GNodeArg(GNA_IN, "S67_Lstm_95_c_b", 0), GNodeArg(GNA_IN, "S64_Lstm_95_r_2_o_w", 0), GNodeArg(GNA_IN, "S68_Lstm_95_o_b", 0), GNodeArg(GNA_OUT, "S71_Output", 0), GNodeArg(GNA_IN, "S71_Infos", 0), GNodeCArg("Reset")));
    // Node S88_Op_LSTM_161 inq -0.57<(i8-0.00)*0.00445706<0.57 outq -0.57<(i8-0.00)*0.00445706<0.57
    AddNode("S88_Op_LSTM_161", Bindings(18, GNodeArg(GNA_INOUT, "S87_Lstm_161_c_state", "S87_Lstm_161_c_state"), GNodeArg(GNA_INOUT, "S86_Lstm_161_i_state", "S86_Lstm_161_i_state"), AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, AT_NO_ARG_BINDING, GNodeArg(GNA_IN, "S71_Output", 0), GNodeArg(GNA_IN, "S79_Lstm_161_r_2_f_w", 0), GNodeArg(GNA_IN, "S83_Lstm_161_f_b", 0), GNodeArg(GNA_IN, "S78_Lstm_161_r_2_i_w", 0), GNodeArg(GNA_IN, "S82_Lstm_161_i_b", 0), GNodeArg(GNA_IN, "S80_Lstm_161_r_2_c_w", 0), GNodeArg(GNA_IN, "S84_Lstm_161_c_b", 0), GNodeArg(GNA_IN, "S81_Lstm_161_r_2_o_w", 0), GNodeArg(GNA_IN, "S85_Lstm_161_o_b", 0), GNodeArg(GNA_OUT, "S88_Output", 0), GNodeArg(GNA_IN, "S88_Infos", 0), GNodeCArg("Reset")));
    // Node S92_MatAdd_256x1 in1q -9.09<(i8-0.00)*0.07102456<9.02 in2q -0.57<(i8-0.00)*0.00445706<0.57 outq -9.06<(i8-0.00)*0.07079369<8.99
    AddNode("S92_MatAdd_256x1", Bindings(4, GNodeArg(GNA_IN, "S55_Output", 0), GNodeArg(GNA_IN, "S88_Output", 0), GNodeArg(GNA_OUT, "S92_Output", 0), GNodeArg(GNA_IN, "S92_Infos", 0)));
    // Node S96_Conv2d_256x256x1x1_Relu inq -9.06<(i8-0.00)*0.07079369<8.99 weightsq chan<(i8-0.00)*chan<chan outq -5.57<(i8-0.00)*0.04354671<5.53 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S96_Conv2d_256x256x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S92_Output", 0), GNodeArg(GNA_IN, "S94_Conv_175_weights", 0), GNodeArg(GNA_IN, "S95_Constant_decoder.0.0.bias", 0), GNodeArg(GNA_OUT, "S96_Output", 0), GNodeArg(GNA_IN, "S96_Mul_scale", 0), GNodeArg(GNA_IN, "S96_Mul_shift", 0), GNodeArg(GNA_IN, "S96_Infos", 0)));
    // Node Resize_196 inq -5.57<(i8-0.00)*0.04354671<5.53 outq -5.57<(i8-0.00)*0.04354671<5.53
    AddNode("S99_Op_Resize_196", Bindings(2, GNodeArg(GNA_IN, "S96_Output", 0), GNodeArg(GNA_OUT, "S99_Output", 0)));
    // Node S104_Conv2d_128x256x1x4_Relu inq -5.57<(i8-0.00)*0.04354671<5.53 weightsq chan<(i8-0.00)*chan<chan outq -7.02<(i8-0.00)*0.05484007<6.96 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S104_Conv2d_128x256x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S99_Output", 0), GNodeArg(GNA_IN, "S102_Conv_221_weights", 0), GNodeArg(GNA_IN, "S103_Constant_decoder.0.4.bias", 0), GNodeArg(GNA_OUT, "S104_Output", 0), GNodeArg(GNA_IN, "S104_Mul_scale", 0), GNodeArg(GNA_IN, "S104_Mul_shift", 0), GNodeArg(GNA_IN, "S104_Infos", 0)));
    // Node S106_MatAdd_128x4 in1q -18.08<(i8-0.00)*0.14123094<17.94 in2q -7.02<(i8-0.00)*0.05484007<6.96 outq -18.66<(i8-0.00)*0.14574265<18.51
    AddNode("S106_MatAdd_128x4", Bindings(4, GNodeArg(GNA_IN, "S45_Output", 0), GNodeArg(GNA_IN, "S104_Output", 0), GNodeArg(GNA_OUT, "S106_Output", 0), GNodeArg(GNA_IN, "S106_Infos", 0)));
    // Node S110_Conv2d_128x128x1x1_Relu inq -18.66<(i8-0.00)*0.14574265<18.51 weightsq chan<(i8-0.00)*chan<chan outq -11.64<(i8-0.00)*0.09097423<11.55 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S110_Conv2d_128x128x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S106_Output", 0), GNodeArg(GNA_IN, "S108_Conv_234_weights", 0), GNodeArg(GNA_IN, "S109_Constant_decoder.1.0.bias", 0), GNodeArg(GNA_OUT, "S110_Output", 0), GNodeArg(GNA_IN, "S110_Mul_scale", 0), GNodeArg(GNA_IN, "S110_Mul_shift", 0), GNodeArg(GNA_IN, "S110_Infos", 0)));
    // Node Resize_255 inq -11.64<(i8-0.00)*0.09097423<11.55 outq -11.64<(i8-0.00)*0.09097423<11.55
    AddNode("S113_Op_Resize_255", Bindings(2, GNodeArg(GNA_IN, "S110_Output", 0), GNodeArg(GNA_OUT, "S113_Output", 0)));
    // Node S118_Conv2d_64x128x1x4_Relu inq -11.64<(i8-0.00)*0.09097423<11.55 weightsq chan<(i8-0.00)*chan<chan outq -7.21<(i8-0.00)*0.05629586<7.15 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S118_Conv2d_64x128x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S113_Output", 0), GNodeArg(GNA_IN, "S116_Conv_280_weights", 0), GNodeArg(GNA_IN, "S117_Constant_decoder.1.4.bias", 0), GNodeArg(GNA_OUT, "S118_Output", 0), GNodeArg(GNA_IN, "S118_Mul_scale", 0), GNodeArg(GNA_IN, "S118_Mul_shift", 0), GNodeArg(GNA_IN, "S118_Infos", 0)));
    // Node S120_MatAdd_64x16 in1q -66.33<(i8-0.00)*0.51818442<65.81 in2q -7.21<(i8-0.00)*0.05629586<7.15 outq -66.75<(i8-0.00)*0.52145571<66.22
    AddNode("S120_MatAdd_64x16", Bindings(4, GNodeArg(GNA_IN, "S34_Output", 0), GNodeArg(GNA_IN, "S118_Output", 0), GNodeArg(GNA_OUT, "S120_Output", 0), GNodeArg(GNA_IN, "S120_Infos", 0)));
    // Node S124_Conv2d_64x64x1x1_Relu inq -66.75<(i8-0.00)*0.52145571<66.22 weightsq chan<(i8-0.00)*chan<chan outq -21.47<(i8-0.00)*0.16777076<21.31 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S124_Conv2d_64x64x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S120_Output", 0), GNodeArg(GNA_IN, "S122_Conv_293_weights", 0), GNodeArg(GNA_IN, "S123_Constant_decoder.2.0.bias", 0), GNodeArg(GNA_OUT, "S124_Output", 0), GNodeArg(GNA_IN, "S124_Mul_scale", 0), GNodeArg(GNA_IN, "S124_Mul_shift", 0), GNodeArg(GNA_IN, "S124_Infos", 0)));
    // Node Resize_314 inq -21.47<(i8-0.00)*0.16777076<21.31 outq -21.47<(i8-0.00)*0.16777076<21.31
    AddNode("S127_Op_Resize_314", Bindings(2, GNodeArg(GNA_IN, "S124_Output", 0), GNodeArg(GNA_OUT, "S127_Output", 0)));
    // Node S132_Conv2d_32x64x1x4_Relu inq -21.47<(i8-0.00)*0.16777076<21.31 weightsq chan<(i8-0.00)*chan<chan outq -15.54<(i8-0.00)*0.12140365<15.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S132_Conv2d_32x64x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S127_Output", 0), GNodeArg(GNA_IN, "S130_Conv_339_weights", 0), GNodeArg(GNA_IN, "S131_Constant_decoder.2.4.bias", 0), GNodeArg(GNA_OUT, "S132_Output", 0), GNodeArg(GNA_IN, "S132_Mul_scale", 0), GNodeArg(GNA_IN, "S132_Mul_shift", 0), GNodeArg(GNA_IN, "S132_Infos", 0)));
    // Node S134_MatAdd_32x64 in1q -252.85<(i8-0.00)*1.97536814<250.87 in2q -15.54<(i8-0.00)*0.12140365<15.42 outq -252.85<(i8-0.00)*1.97536814<250.87
    AddNode("S134_MatAdd_32x64", Bindings(4, GNodeArg(GNA_IN, "S23_Output", 0), GNodeArg(GNA_IN, "S132_Output", 0), GNodeArg(GNA_OUT, "S134_Output", 0), GNodeArg(GNA_IN, "S134_Infos", 0)));
    // Node S138_Conv2d_32x32x1x1_Relu inq -252.85<(i8-0.00)*1.97536814<250.87 weightsq chan<(i8-0.00)*chan<chan outq -189.32<(i8-0.00)*1.47904742<187.84 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S138_Conv2d_32x32x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S134_Output", 0), GNodeArg(GNA_IN, "S136_Conv_352_weights", 0), GNodeArg(GNA_IN, "S137_Constant_decoder.3.0.bias", 0), GNodeArg(GNA_OUT, "S138_Output", 0), GNodeArg(GNA_IN, "S138_Mul_scale", 0), GNodeArg(GNA_IN, "S138_Mul_shift", 0), GNodeArg(GNA_IN, "S138_Infos", 0)));
    // Node Resize_373 inq -189.32<(i8-0.00)*1.47904742<187.84 outq -189.32<(i8-0.00)*1.47904742<187.84
    AddNode("S141_Op_Resize_373", Bindings(2, GNodeArg(GNA_IN, "S138_Output", 0), GNodeArg(GNA_OUT, "S141_Output", 0)));
    // Node S146_Conv2d_16x32x1x4_Relu inq -189.32<(i8-0.00)*1.47904742<187.84 weightsq chan<(i8-0.00)*chan<chan outq -81.07<(i8-0.00)*0.63337237<80.44 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S146_Conv2d_16x32x1x4_Relu", Bindings(7, GNodeArg(GNA_IN, "S141_Output", 0), GNodeArg(GNA_IN, "S144_Conv_398_weights", 0), GNodeArg(GNA_IN, "S145_Constant_decoder.3.4.bias", 0), GNodeArg(GNA_OUT, "S146_Output", 0), GNodeArg(GNA_IN, "S146_Mul_scale", 0), GNodeArg(GNA_IN, "S146_Mul_shift", 0), GNodeArg(GNA_IN, "S146_Infos", 0)));
    // Node S148_MatAdd_16x256 in1q -1796.44<(i8-0.00)*14.03465652<1782.40 in2q -81.07<(i8-0.00)*0.63337237<80.44 outq -1796.44<(i8-0.00)*14.03465652<1782.40
    AddNode("S148_MatAdd_16x256", Bindings(4, GNodeArg(GNA_IN, "S12_Output", 0), GNodeArg(GNA_IN, "S146_Output", 0), GNodeArg(GNA_OUT, "S148_Output", 0), GNodeArg(GNA_IN, "S148_Infos", 0)));
    // Node S152_Conv2d_16x16x1x1_Relu inq -1796.44<(i8-0.00)*14.03465652<1782.40 weightsq chan<(i8-0.00)*chan<chan outq -662.68<(i8-0.00)*5.17715073<657.50 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S152_Conv2d_16x16x1x1_Relu", Bindings(7, GNodeArg(GNA_IN, "S148_Output", 0), GNodeArg(GNA_IN, "S150_Conv_411_weights", 0), GNodeArg(GNA_IN, "S151_Constant_decoder.4.0.bias", 0), GNodeArg(GNA_OUT, "S152_Output", 0), GNodeArg(GNA_IN, "S152_Mul_scale", 0), GNodeArg(GNA_IN, "S152_Mul_shift", 0), GNodeArg(GNA_IN, "S152_Infos", 0)));
    // Node Resize_432 inq -662.68<(i8-0.00)*5.17715073<657.50 outq -662.68<(i8-0.00)*5.17715073<657.50
    AddNode("S155_Op_Resize_432", Bindings(2, GNodeArg(GNA_IN, "S152_Output", 0), GNodeArg(GNA_OUT, "S155_Output", 0)));
    // Node S160_Conv2d_1x16x1x4 inq -662.68<(i8-0.00)*5.17715073<657.50 weightsq -0.14<(i8-0.00)*0.00113291<0.14 outq -75.75<(i8-0.00)*0.59183520<75.16 biasesq -12595496.00<(i32-0.00)*0.00586523<12595495.99
    AddNode("S160_Conv2d_1x16x1x4", Bindings(7, GNodeArg(GNA_IN, "S155_Output", 0), GNodeArg(GNA_IN, "S158_Conv_457_weights", 0), GNodeArg(GNA_IN, "S159_Constant_decoder.4.4.bias", 0), GNodeArg(GNA_OUT, "Output_1", 0), GNodeArg(GNA_IN, "S160_Mul_scale", 0), GNodeArg(GNA_IN, "S160_Mul_shift", 0), GNodeArg(GNA_IN, "S160_Infos", 0)));
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    denoiserModel(52000, 300*1024, 8*1024*1024, 20*1024*1024);
    GenerateTilingCode();
    return 0;
}
