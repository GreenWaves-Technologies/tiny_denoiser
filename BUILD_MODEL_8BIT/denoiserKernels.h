#ifndef __DENOISERKERNEL_H__
#define __DENOISERKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "nntool_extra_kernels.h"
#include "CNN_BasicKernels_SQ8.h"
#include "denoiser.h"
#include "ResizeBasicKernels.h"
#define _denoiser_L1_Memory_SIZE 47000
#define _denoiser_L2_Memory_SIZE 0
extern char *denoiser_L1_Memory; /* Size given for generation: 48736 bytes, used: 47000 bytes */
extern char *denoiser_L2_Memory; /* Size used for generation: 0 bytes */
extern void S4_Conv2d_16x1x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S9_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S12_Op_Slice_409(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S15_Conv2d_32x16x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S20_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S23_Op_Slice_350(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S26_Conv2d_64x32x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S31_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S34_Op_Slice_291(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S37_Conv2d_128x64x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S42_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S45_Op_Slice_232(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S48_Conv2d_256x128x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S53_Conv2d_256x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S55_Op_Slice_173(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S71_Op_LSTM_95G2(
		signed char * __restrict__ SCin,
		signed char * __restrict__ SHin,
		signed char * __restrict__ SCout,
		signed char * __restrict__ SHout,
		signed char * __restrict__ Xin,
		signed char * __restrict__ Wf,
		int * __restrict__ Bf,
		signed char * __restrict__ Wi,
		int * __restrict__ Bi,
		signed char * __restrict__ Wg,
		int * __restrict__ Bg,
		signed char * __restrict__ Wo,
		int * __restrict__ Bo,
		signed char * __restrict__ Hout,
		signed char * __restrict__ Infos,
		signed char  Reset);
extern void S88_Op_LSTM_161G2(
		signed char * __restrict__ SCin,
		signed char * __restrict__ SHin,
		signed char * __restrict__ SCout,
		signed char * __restrict__ SHout,
		signed char * __restrict__ Xin,
		signed char * __restrict__ Wf,
		int * __restrict__ Bf,
		signed char * __restrict__ Wi,
		int * __restrict__ Bi,
		signed char * __restrict__ Wg,
		int * __restrict__ Bg,
		signed char * __restrict__ Wo,
		int * __restrict__ Bo,
		signed char * __restrict__ Hout,
		signed char * __restrict__ Infos,
		signed char  Reset);
extern void S92_MatAdd_256x1(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S96_Conv2d_256x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S99_Op_Resize_196(
		signed char * In,
		signed char * Out);
extern void S104_Conv2d_128x256x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S106_MatAdd_128x4(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S110_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S113_Op_Resize_255(
		signed char * In,
		signed char * Out);
extern void S118_Conv2d_64x128x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S120_MatAdd_64x16(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S124_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S127_Op_Resize_314(
		signed char * In,
		signed char * Out);
extern void S132_Conv2d_32x64x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S134_MatAdd_32x64(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S138_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S141_Op_Resize_373(
		signed char * In,
		signed char * Out);
extern void S146_Conv2d_16x32x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S148_MatAdd_16x256(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S152_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S155_Op_Resize_432(
		signed char * In,
		signed char * Out);
extern void S160_Conv2d_1x16x1x4(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S71_Op_LSTM_95(
		signed char * __restrict__ Cinout,
		signed char * __restrict__ Hinout,
		signed char * __restrict__ Xin,
		signed char * __restrict__ Wf,
		int * __restrict__ Bf,
		signed char * __restrict__ Wi,
		int * __restrict__ Bi,
		signed char * __restrict__ Wg,
		int * __restrict__ Bg,
		signed char * __restrict__ Wo,
		int * __restrict__ Bo,
		signed char * __restrict__ Hout,
		signed char * __restrict__ Infos,
		signed char  Reset);
extern void S88_Op_LSTM_161(
		signed char * __restrict__ Cinout,
		signed char * __restrict__ Hinout,
		signed char * __restrict__ Xin,
		signed char * __restrict__ Wf,
		int * __restrict__ Bf,
		signed char * __restrict__ Wi,
		int * __restrict__ Bi,
		signed char * __restrict__ Wg,
		int * __restrict__ Bg,
		signed char * __restrict__ Wo,
		int * __restrict__ Bo,
		signed char * __restrict__ Hout,
		signed char * __restrict__ Infos,
		signed char  Reset);
#endif
