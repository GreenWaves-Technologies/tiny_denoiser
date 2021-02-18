#include "denoiserKernels.h"
L1_CL_MEM AT_L1_POINTER denoiser_L1_Memory;
L2_MEM AT_L2_POINTER denoiser_L2_Memory;
static AT_HYPERFLASH_FS_T HyperFlash;
void S4_Conv2d_16x1x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 23020 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 1, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 17408 [D1, [0 x 17408, 17408]][Tile0, 1:[272x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 17408, 17408]][Tile0, 1:[272x1], 4]
		Tile0: [0, 17408, 1088], Tile1: [0, 17408, 1088], Tile2; [0, 17408, 1088]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]][D0, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]][D0, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [D1, [0 x 4352, 4352]][Tile0, 1:[272x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4352, 4352]][Tile0, 1:[272x1], 1]
		Tile0: [0, 4352, 4352], Tile1: [0, 4352, 4352], Tile2; [0, 4352, 4352]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1088 [D0, [0 x 1088, 1088]][Tile0, 1:[1088x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1088x1], 1][D0, [0 x 1088, 1088]]
		Tile0: [0, 1088, 1088], Tile1: [0, 1088, 1088], Tile2; [0, 1088, 1088]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+5600);
	KerArg0->W = (unsigned short int) (272);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+1088);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (1088);
	KerArg1->UsedW = (unsigned short int) (1088);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (1);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+1184);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+5600);
	KerArg1->Pad = (v4s) 0;
	KerArg1->N = (unsigned short) (4);
	KerArg1->S = (unsigned char) (4);
	KerArg1->Ny = (unsigned short) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+5600);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+1248);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (272);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+1152);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+1168);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+23008);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1088), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1152), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1168), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1184), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1088, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+23008), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+23008))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1248), 4352, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S9_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 13420 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [Tile0, 1:[16x272], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x272], 1]
		Tile0: [0, 4352, 272], Tile1: [0, 4352, 272], Tile2; [0, 4352, 272]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [Tile0, 1:[16x272], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x272], 1]
		Tile0: [0, 4352, 4352], Tile1: [0, 4352, 4352], Tile2; [0, 4352, 4352]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[16x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[16x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 4]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [Tile0, 1:[16x272], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x272], 1]
		Tile0: [0, 4352, 4352], Tile1: [0, 4352, 4352], Tile2; [0, 4352, 4352]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+256);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+4608);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (272);
	KerArg0->H = (unsigned short int) (16);
	KerArg1->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (16);
	KerArg1->H_In1 = (unsigned short int) (16);
	KerArg1->In2 = (signed char * __restrict__) (denoiser_L1_Memory+4608);
	KerArg1->W_In2 = (unsigned short int) (272);
	KerArg1->Bias = (void * __restrict__) (denoiser_L1_Memory+8960);
	KerArg1->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+13376);
	KerArg1->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+13392);
	KerArg1->Out = (signed char * __restrict__) (denoiser_L1_Memory+9024);
	KerArg1->Infos = (signed char *__restrict__) (denoiser_L1_Memory+13408);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+256), 4352, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8960), 64, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+13376), 16, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+13392), 16, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+13408), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+13408))[5]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+9024), 4352, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S12_Op_Slice_409(
		signed char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 8192 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerCopy_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[4096x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4096x1], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[4096x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4096x1], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+4096);
	KerArg0->W = (unsigned int) (4096);
	KerArg0->H = (unsigned int) (1);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Copy_fps, (void *) KerArg0);
		__CALL(CNN_Copy_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 4096, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S15_Conv2d_32x16x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 17484 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8704 [D1, [0 x 8704, 8704]][Tile0, 1:[68x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 8704, 8704]][Tile0, 1:[68x1], 4]
		Tile0: [0, 8704, 272], Tile1: [0, 8704, 272], Tile2; [0, 8704, 272]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D1, [0 x 2048, 2048]][D0, [0 x 2048, 2048]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2048, 2048]][D0, [0 x 2048, 2048]]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2176 [D1, [0 x 2176, 2176]][Tile0, 1:[68x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2176, 2176]][Tile0, 1:[68x1], 1]
		Tile0: [0, 2176, 2176], Tile1: [0, 2176, 2176], Tile2; [0, 2176, 2176]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [D0, [0 x 4352, 4352]][Tile0, 1:[272x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[272x1], 1][D0, [0 x 4352, 4352]]
		Tile0: [0, 4352, 4352], Tile1: [0, 4352, 4352], Tile2; [0, 4352, 4352]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+8768);
	KerArg0->W = (unsigned short int) (68);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+4352);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (272);
	KerArg1->UsedW = (unsigned short int) (272);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+4544);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+8768);
	KerArg1->Pad = (v4s) 0;
	KerArg1->N = (unsigned short) (4);
	KerArg1->S = (unsigned char) (4);
	KerArg1->Ny = (unsigned short) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+8768);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+6592);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->W = (unsigned short int) (68);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+4480);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+4512);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+17472);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4352), 128, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4480), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4512), 32, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4544), 2048, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4352, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+17472), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+17472))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+6592), 2176, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S20_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 7756 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2176 [Tile0, 1:[32x68], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x68], 1]
		Tile0: [0, 2176, 68], Tile1: [0, 2176, 68], Tile2; [0, 2176, 68]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2176 [Tile0, 1:[32x68], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x68], 1]
		Tile0: [0, 2176, 2176], Tile1: [0, 2176, 2176], Tile2; [0, 2176, 2176]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[32x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x32], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[32x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 4]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2176 [Tile0, 1:[32x68], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x68], 1]
		Tile0: [0, 2176, 2176], Tile1: [0, 2176, 2176], Tile2; [0, 2176, 2176]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[32x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[32x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+1024);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+3200);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (68);
	KerArg0->H = (unsigned short int) (32);
	KerArg1->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (32);
	KerArg1->H_In1 = (unsigned short int) (32);
	KerArg1->In2 = (signed char * __restrict__) (denoiser_L1_Memory+3200);
	KerArg1->W_In2 = (unsigned short int) (68);
	KerArg1->Bias = (void * __restrict__) (denoiser_L1_Memory+5376);
	KerArg1->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+7680);
	KerArg1->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+7712);
	KerArg1->Out = (signed char * __restrict__) (denoiser_L1_Memory+5504);
	KerArg1->Infos = (signed char *__restrict__) (denoiser_L1_Memory+7744);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 2176, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5376), 128, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7680), 32, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7712), 32, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7744), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+7744))[5]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5504), 2176, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S23_Op_Slice_350(
		signed char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 4096 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerCopy_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[2048x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[2048x1], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[2048x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[2048x1], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+2048);
	KerArg0->W = (unsigned int) (2048);
	KerArg0->H = (unsigned int) (1);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 2048, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Copy_fps, (void *) KerArg0);
		__CALL(CNN_Copy_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2048), 2048, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S26_Conv2d_64x32x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 16204 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 32, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4352 [D1, [0 x 4352, 4352]][Tile0, 1:[17x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4352, 4352]][Tile0, 1:[17x1], 4]
		Tile0: [0, 4352, 68], Tile1: [0, 4352, 68], Tile2; [0, 4352, 68]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8192 [D1, [0 x 8192, 8192]][D0, [0 x 8192, 8192]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 8192, 8192]][D0, [0 x 8192, 8192]]
		Tile0: [0, 8192, 8192], Tile1: [0, 8192, 8192], Tile2; [0, 8192, 8192]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1088 [D1, [0 x 1088, 1088]][Tile0, 1:[17x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1088, 1088]][Tile0, 1:[17x1], 1]
		Tile0: [0, 1088, 1088], Tile1: [0, 1088, 1088], Tile2; [0, 1088, 1088]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2176 [D0, [0 x 2176, 2176]][Tile0, 1:[68x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[68x1], 1][D0, [0 x 2176, 2176]]
		Tile0: [0, 2176, 2176], Tile1: [0, 2176, 2176], Tile2; [0, 2176, 2176]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+11840);
	KerArg0->W = (unsigned short int) (17);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (64);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+2176);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (68);
	KerArg1->UsedW = (unsigned short int) (68);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->TotalInFeatures = (unsigned short int) (32);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+2560);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+11840);
	KerArg1->Pad = (v4s) 0;
	KerArg1->N = (unsigned short) (4);
	KerArg1->S = (unsigned char) (4);
	KerArg1->Ny = (unsigned short) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+11840);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+10752);
	KerArg2->Feat = (unsigned short int) (64);
	KerArg2->W = (unsigned short int) (17);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+2432);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+2496);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+16192);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2176), 256, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2432), 64, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2496), 64, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2560), 8192, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 2176, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16192), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+16192))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+10752), 1088, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S31_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 6924 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile1, 1:[64x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[64x64], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1088 [Tile0, 1:[64x17], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x17], 1]
		Tile0: [0, 1088, 1088], Tile1: [0, 1088, 1088], Tile2; [0, 1088, 1088]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 1:[1x64], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 4]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1088 [Tile1, 1:[17x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[17x64], 1]
		Tile0: [0, 1088, 1088], Tile1: [0, 1088, 1088], Tile2; [0, 1088, 1088]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[1x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[1x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (64);
	KerArg0->H_In1 = (unsigned short int) (64);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+5824);
	KerArg0->W_In2 = (unsigned short int) (17);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+4096);
	KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+5440);
	KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+5504);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+4352);
	KerArg0->W_Out = (unsigned short int) (17);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+5568);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+6912);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5824), 1088, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5440), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5504), 64, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+6912), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*17);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+6912))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4352), 1088, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S34_Op_Slice_291(
		signed char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 2048 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerCopy_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+1024);
	KerArg0->W = (unsigned int) (1024);
	KerArg0->H = (unsigned int) (1);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1024, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Copy_fps, (void *) KerArg0);
		__CALL(CNN_Copy_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S37_Conv2d_128x64x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 37196 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 128, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 64, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D1, [0 x 2048, 2048]][Tile0, 1:[4x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2048, 2048]][Tile0, 1:[4x1], 4]
		Tile0: [0, 2048, 16], Tile1: [0, 2048, 16], Tile2; [0, 2048, 16]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D1, [0 x 512, 512]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 512, 512]]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32768 [D1, [0 x 32768, 32768]][D0, [0 x 32768, 32768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32768, 32768]][D0, [0 x 32768, 32768]]
		Tile0: [0, 32768, 32768], Tile1: [0, 32768, 32768], Tile2; [0, 32768, 32768]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D1, [0 x 512, 512]][Tile0, 1:[4x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 512, 512]][Tile0, 1:[4x1], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1088 [D0, [0 x 1088, 1088]][Tile0, 1:[17x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[17x1], 1][D0, [0 x 1088, 1088]]
		Tile0: [0, 1088, 1088], Tile1: [0, 1088, 1088], Tile2; [0, 1088, 1088]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+35136);
	KerArg0->W = (unsigned short int) (4);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (128);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+1088);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (17);
	KerArg1->UsedW = (unsigned short int) (16);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (128);
	KerArg1->TotalInFeatures = (unsigned short int) (64);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+1856);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+35136);
	KerArg1->Pad = (v4s) 0;
	KerArg1->N = (unsigned short) (4);
	KerArg1->S = (unsigned char) (4);
	KerArg1->Ny = (unsigned short) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+35136);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+34624);
	KerArg2->Feat = (unsigned short int) (128);
	KerArg2->W = (unsigned short int) (4);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+1600);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+1728);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+37184);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1088), 512, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1600), 128, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1728), 128, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1856), 32768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1088, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+37184), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+37184))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+34624), 512, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S42_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18700 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[512x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[512x1], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16384 [Tile1, 1:[128x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[128x128], 1]
		Tile0: [0, 16384, 16384], Tile1: [0, 16384, 16384], Tile2; [0, 16384, 16384]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[128x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[128x4], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile1, 1:[1x128], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 4]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile1, 1:[4x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[4x128], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile1, 1:[1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile1, 1:[1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (128);
	KerArg0->H_In1 = (unsigned short int) (128);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+18176);
	KerArg0->W_In2 = (unsigned short int) (4);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+16384);
	KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+17408);
	KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+17536);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+16896);
	KerArg0->W_Out = (unsigned short int) (4);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+17664);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+18688);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 16384, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+18176), 512, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16384), 512, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+17408), 128, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+17536), 128, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+18688), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*4);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+18688))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16896), 512, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S45_Op_Slice_232(
		signed char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 1024 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerCopy_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[512x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[512x1], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[512x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[512x1], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+512);
	KerArg0->W = (unsigned int) (512);
	KerArg0->H = (unsigned int) (1);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 512, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Copy_fps, (void *) KerArg0);
		__CALL(CNN_Copy_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+512), 512, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S48_Conv2d_256x128x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 44204 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _LN_Filter;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 256, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 128, Tiled: 7]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 512 [D0, [6 x 80, 32]][Tile0, 1:[4x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4x1], 1][D0, [6 x 80, 32]]
		Tile0: [0, 80, 4], Tile1: [80, 80, 4], Tile2; [160, 80, 4]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D1, [0 x 1024, 1024]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1024, 1024]]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 131072 [D1, [0 x 131072, 131072]][D0, [6 x 20480, 8192]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 131072, 131072]][D0, [6 x 20480, 8192]]
		Tile0: [0, 20480, 80], Tile1: [80, 20480, 80], Tile2; [160, 20480, 80]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]][Tile0, 1:[1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]][Tile0, 1:[1x1], 1]
		Tile0: [0, 256, 1], Tile1: [0, 256, 1], Tile2; [0, 256, 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D1, [0 x 1024, 1024]][Tile0, 1:[1x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1024, 1024]][Tile0, 1:[1x1], 4]
		Tile0: [0, 1024, 4], Tile1: [0, 1024, 4], Tile2; [0, 1024, 4]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+43168);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (256);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+160);
	KerArg1->W = (unsigned short int) (4);
	KerArg1->UsedW = (unsigned short int) (4);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->UsedH = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (256);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+43168);
	KerArg1->Pad = (v4s) 0;
	KerArg1->N = (unsigned short) (4);
	KerArg1->S = (unsigned char) (4);
	KerArg1->Ny = (unsigned short) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+43168);
	KerArg2->Feat = (unsigned short int) (256);
	KerArg2->W = (unsigned short int) (1);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+1184);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+1440);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+44192);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+0), 80, 4, 4, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+160), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1184), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1440), 256, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1696+0), 20480, 512, 80, 0, &DmaR_Evt5);
	_N_Filter=0;
	_C_Out=0; _SC_Out=256; _LC_Out=1;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+44192), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+44192))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<7; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==6), D0Ind_NextLast = ((D0Ind+1)==6);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (80); _LN_In = (4); _SN_In = (((D0Ind_NextLast)?8:20)*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-480); _LN_In = (4); _SN_In = (20*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + (80); _LN_Filter = ((D0Ind_NextLast)?32:80); _SN_Filter = (256*_LN_Filter); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + (-480); _LN_Filter = (80); _SN_Filter = (256*_LN_Filter); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+80*((D0Ind_Total+1)%2)),
							_SN_In, 4, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1696+20480*((D0Ind_Total+1)%2)),
							_SN_Filter, 512, _LN_Filter, 0, &DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0+80*((D0Ind_Total)%2));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?8:20);
				KerArg1->TotalInFeatures = (unsigned short int) (D0Ind_Last?8:20);
				KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+1696+20480*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConvNxMStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMStrideSxSy_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+42656+256*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42656+256*((T0Ind_Total)%2)),
					_SC_Out, 1, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			
			/*============================= End Prepare Tiles ===================================*/
			T0Ind_Total++;
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S53_Conv2d_256x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 43948 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Total=0, T1Ind_Last, T1Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 4][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In1, Tiled Space: Tile1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 65536 [Tile1, 4:[256x80, 2:256x80, 256x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[256x80, 2:256x80, 256x16], 1]
		Tile0: [0, 20480, 20480], Tile1: [20480, 20480, 20480], Tile2; [40960, 20480, 20480]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile1, 4:[1x80, 2:1x80, 1x16], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 4]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Out, Tiled Space: Tile1
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 80, 80], Tile1: [80, 80, 80], Tile2; [160, 80, 80]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W_In1 = (unsigned short int) (256);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+43680);
	KerArg0->W_In2 = (unsigned short int) (1);
	KerArg0->W_Out = (unsigned short int) (1);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+42656);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+43936);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+0), 20480, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+43680), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+40960), 1024, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=80;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42144), 256, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42400), 256, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+43936), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T1Ind=0; T1Ind<4; T1Ind++, T1Ind_Total++) { /* Iteration on Tile1 */
		int T1Ind_Last = (T1Ind==3), T1Ind_NextLast = ((T1Ind+1)==3);
		/*================================= Prepare Tiles ===================================*/
		_SN_In1 = 0;
		if (!(T1Ind_Last)) {
			_N_In1 = _N_In1 + (20480); _SN_In1 = ((T1Ind_NextLast)?4096:20480); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
		if (_SN_In1) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+20480*((T1Ind_Total+1)%2)),
					_SN_In1, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0+20480*((T1Ind_Total)%2));
			KerArg0->H_In1 = (unsigned short int) (T1Ind_Last?16:80);
			KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+40960+(320*(T1Ind)));
			KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+42144+(80*(T1Ind)));
			KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+42400+(80*(T1Ind)));
			KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+41984+80*((T1Ind_Total)%2));
			KerArg0->OutFirstCol = (unsigned short int) ((0)*1);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+43936))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+41984+80*((T1Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T1Ind_Last)) {
			_C_Out = _C_Out + (80); _SC_Out = ((T1Ind_NextLast)?16:80); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S55_Op_Slice_173(
		signed char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 512 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerCopy_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+256);
	KerArg0->W = (unsigned int) (256);
	KerArg0->H = (unsigned int) (1);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 256, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Copy_fps, (void *) KerArg0);
		__CALL(CNN_Copy_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+256), 256, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S71_Op_LSTM_95G2(
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
		signed char  Reset)

{
	/* Shared L1: 47000 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaR_Evt7;
	AT_L2_EVENT DmaR_Evt8;
	AT_L2_EVENT DmaR_Evt9;
	AT_L2_EVENT DmaR_Evt10;
	AT_L2_EVENT DmaR_Evt11;
	AT_L2_EVENT DmaR_Evt12;
	AT_L2_EVENT DmaW_Evt1;
	AT_L2_EVENT DmaW_Evt2;
	AT_L2_EVENT DmaW_Evt3;
	KerLSTM_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_Wf;
	unsigned int _SN_Wf;
	unsigned int _N_Bf;
	unsigned int _SN_Bf;
	unsigned int _N_Wi;
	unsigned int _SN_Wi;
	unsigned int _N_Bi;
	unsigned int _SN_Bi;
	unsigned int _N_Wg;
	unsigned int _SN_Wg;
	unsigned int _N_Bg;
	unsigned int _SN_Bg;
	unsigned int _N_Wo;
	unsigned int _SN_Wo;
	unsigned int _N_Bo;
	unsigned int _SN_Bo;
	unsigned int _C_Hout;
	unsigned int _SP_Hout, _SC_Hout;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: 1][Tile0 Dim: 24]
	Ker Arg: Wf, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: SCin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SHin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SCout, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SHout, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Xin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: State, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 768 [Tile0, 24:[768x1, 22:768x1, 768x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[768x1, 22:768x1, 768x1], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Bf, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wi, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bi, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wg, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bg, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wo, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bo, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Hout, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]][Tile0, 24:[1x11, 22:1x11, 1x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]][Tile0, 24:[1x11, 22:1x11, 1x3], 1]
		Tile0: [0, 11, 11], Tile1: [11, 11, 11], Tile2; [22, 11, 11]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 29 [Tile0, 24:[29x1, 22:29x1, 29x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[29x1, 22:29x1, 29x1], 1]
		Tile0: [0, 29, 29], Tile1: [0, 29, 29], Tile2; [0, 29, 29]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->StateInOut = (signed char *__restrict__) (denoiser_L1_Memory+768);
	KerArg0->State = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Xin = (signed char *__restrict__) (denoiser_L1_Memory+1280);
	KerArg0->DimState = (unsigned short int) (256);
	KerArg0->DimIn = (unsigned short int) (256);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+46968);
	KerArg0->FirstCell = (char) ((1));
	KerArg0->Reset = (char) (Reset);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wf+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1536+0), 5632, 0, &DmaR_Evt1);
	_N_Wf=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SCin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+768), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read SCin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SHin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read SHin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Xin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1280), 256, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Xin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bf+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12800+0), 44, 0, &DmaR_Evt5);
	_N_Bf=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wi+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12888+0), 5632, 0, &DmaR_Evt6);
	_N_Wi=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bi+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24152+0), 44, 0, &DmaR_Evt7);
	_N_Bi=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wg+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24240+0), 5632, 0, &DmaR_Evt8);
	_N_Wg=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bg+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35504+0), 44, 0, &DmaR_Evt9);
	_N_Bg=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wo+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35592+0), 5632, 0, &DmaR_Evt10);
	_N_Wo=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bo+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46856+0), 44, 0, &DmaR_Evt11);
	_N_Bo=0;
	_C_Hout=0; _SC_Hout=11;
	_SP_Hout=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46968), 29, 0, &DmaR_Evt12);
	AT_L2_WAIT(0, &DmaR_Evt12); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		for (T0Ind=0; T0Ind<24; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==23), T0Ind_NextLast = ((T0Ind+1)==23);
			/*================================= Prepare Tiles ===================================*/
			_SN_Wf = 0;
			if (!(T0Ind_Last)) {
				_N_Wf = _N_Wf + (5632); _SN_Wf = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wf = _N_Wf + (-129536); _SN_Wf = (5632); 
			}
			_SN_Bf = 0;
			if (!(T0Ind_Last)) {
				_N_Bf = _N_Bf + (44); _SN_Bf = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bf = _N_Bf + (-1012); _SN_Bf = (44); 
			}
			_SN_Wi = 0;
			if (!(T0Ind_Last)) {
				_N_Wi = _N_Wi + (5632); _SN_Wi = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wi = _N_Wi + (-129536); _SN_Wi = (5632); 
			}
			_SN_Bi = 0;
			if (!(T0Ind_Last)) {
				_N_Bi = _N_Bi + (44); _SN_Bi = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bi = _N_Bi + (-1012); _SN_Bi = (44); 
			}
			_SN_Wg = 0;
			if (!(T0Ind_Last)) {
				_N_Wg = _N_Wg + (5632); _SN_Wg = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wg = _N_Wg + (-129536); _SN_Wg = (5632); 
			}
			_SN_Bg = 0;
			if (!(T0Ind_Last)) {
				_N_Bg = _N_Bg + (44); _SN_Bg = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bg = _N_Bg + (-1012); _SN_Bg = (44); 
			}
			_SN_Wo = 0;
			if (!(T0Ind_Last)) {
				_N_Wo = _N_Wo + (5632); _SN_Wo = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wo = _N_Wo + (-129536); _SN_Wo = (5632); 
			}
			_SN_Bo = 0;
			if (!(T0Ind_Last)) {
				_N_Bo = _N_Bo + (44); _SN_Bo = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bo = _N_Bo + (-1012); _SN_Bo = (44); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Wf */
			if (_SN_Wf) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wf+_N_Wf), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1536+5632*((T0Ind_Total+1)%2)),
						_SN_Wf, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read Bf */
			if (_SN_Bf) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bf+_N_Bf), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12800+44*((T0Ind_Total+1)%2)),
						_SN_Bf, 0, &DmaR_Evt5);
			}
			AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Wi */
			if (_SN_Wi) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wi+_N_Wi), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12888+5632*((T0Ind_Total+1)%2)),
						_SN_Wi, 0, &DmaR_Evt6);
			}
			AT_L2_WAIT(0, &DmaR_Evt7); /* Wait previous DMA read Bi */
			if (_SN_Bi) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bi+_N_Bi), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24152+44*((T0Ind_Total+1)%2)),
						_SN_Bi, 0, &DmaR_Evt7);
			}
			AT_L2_WAIT(0, &DmaR_Evt8); /* Wait previous DMA read Wg */
			if (_SN_Wg) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wg+_N_Wg), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24240+5632*((T0Ind_Total+1)%2)),
						_SN_Wg, 0, &DmaR_Evt8);
			}
			AT_L2_WAIT(0, &DmaR_Evt9); /* Wait previous DMA read Bg */
			if (_SN_Bg) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bg+_N_Bg), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35504+44*((T0Ind_Total+1)%2)),
						_SN_Bg, 0, &DmaR_Evt9);
			}
			AT_L2_WAIT(0, &DmaR_Evt10); /* Wait previous DMA read Wo */
			if (_SN_Wo) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wo+_N_Wo), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35592+5632*((T0Ind_Total+1)%2)),
						_SN_Wo, 0, &DmaR_Evt10);
			}
			AT_L2_WAIT(0, &DmaR_Evt11); /* Wait previous DMA read Bo */
			if (_SN_Bo) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bo+_N_Bo), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46856+44*((T0Ind_Total+1)%2)),
						_SN_Bo, 0, &DmaR_Evt11);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->Wf = (signed char *__restrict__) (denoiser_L1_Memory+1536+5632*((T0Ind_Total)%2));
			KerArg0->Bf = (void * __restrict__) (denoiser_L1_Memory+12800+44*((T0Ind_Total)%2));
			KerArg0->Wi = (signed char *__restrict__) (denoiser_L1_Memory+12888+5632*((T0Ind_Total)%2));
			KerArg0->Bi = (void * __restrict__) (denoiser_L1_Memory+24152+44*((T0Ind_Total)%2));
			KerArg0->Wg = (signed char *__restrict__) (denoiser_L1_Memory+24240+5632*((T0Ind_Total)%2));
			KerArg0->Bg = (void * __restrict__) (denoiser_L1_Memory+35504+44*((T0Ind_Total)%2));
			KerArg0->Wo = (signed char *__restrict__) (denoiser_L1_Memory+35592+5632*((T0Ind_Total)%2));
			KerArg0->Bo = (void * __restrict__) (denoiser_L1_Memory+46856+44*((T0Ind_Total)%2));
			KerArg0->Hout = (signed char *__restrict__) (denoiser_L1_Memory+46944+12*((T0Ind_Total)%2));
			KerArg0->Nout = (unsigned short int) (T0Ind_Last?3:11);
			KerArg0->FirstOut = (char) ((T0Ind==0));
			KerArg0->TileOffset = (int) ((T0Ind)*11);
			AT_FORK(gap_ncore(), (void *) LSTM_ParKerB32_SQ8, (void *) KerArg0);
			__CALL(LSTM_ParKerB32_SQ8, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Hout) AT_L2_WAIT(0, &DmaW_Evt3); /* Wait previous DMA write Hout */
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Hout+_C_Hout), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46944+12*((T0Ind_Total)%2)),
					_SC_Hout, 1, &DmaW_Evt3);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Hout = _SC_Hout;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Hout = 0;
			if (!(T0Ind_Last)) {
				_C_Hout = _C_Hout + (11); _SC_Hout = ((T0Ind_NextLast)?3:11); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SCout+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+768), 256, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write SCout */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SHout+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 256, 1, &DmaW_Evt2);
	AT_L2_WAIT(0, &DmaW_Evt2); /* Wait DMA write SHout */
	AT_L2_WAIT(0, &DmaW_Evt3); /* Wait previous DMA write Hout */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S88_Op_LSTM_161G2(
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
		signed char  Reset)

{
	/* Shared L1: 47000 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaR_Evt7;
	AT_L2_EVENT DmaR_Evt8;
	AT_L2_EVENT DmaR_Evt9;
	AT_L2_EVENT DmaR_Evt10;
	AT_L2_EVENT DmaR_Evt11;
	AT_L2_EVENT DmaR_Evt12;
	AT_L2_EVENT DmaW_Evt1;
	AT_L2_EVENT DmaW_Evt2;
	AT_L2_EVENT DmaW_Evt3;
	KerLSTM_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_Wf;
	unsigned int _SN_Wf;
	unsigned int _N_Bf;
	unsigned int _SN_Bf;
	unsigned int _N_Wi;
	unsigned int _SN_Wi;
	unsigned int _N_Bi;
	unsigned int _SN_Bi;
	unsigned int _N_Wg;
	unsigned int _SN_Wg;
	unsigned int _N_Bg;
	unsigned int _SN_Bg;
	unsigned int _N_Wo;
	unsigned int _SN_Wo;
	unsigned int _N_Bo;
	unsigned int _SN_Bo;
	unsigned int _C_Hout;
	unsigned int _SP_Hout, _SC_Hout;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: 1][Tile0 Dim: 24]
	Ker Arg: Wf, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: SCin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SHin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SCout, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: SHout, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[256x1, 22:256x1, 256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Xin, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: State, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 768 [Tile0, 24:[768x1, 22:768x1, 768x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[768x1, 22:768x1, 768x1], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Bf, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wi, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bi, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wg, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bg, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Wo, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 131072 [Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[512x11, 22:512x11, 512x3], 1]
		Tile0: [0, 5632, 5632], Tile1: [5632, 5632, 5632], Tile2; [11264, 5632, 5632]
	Ker Arg: Bo, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 1024 [Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[1x11, 22:1x11, 1x3], 4]
		Tile0: [0, 44, 44], Tile1: [44, 44, 44], Tile2; [88, 44, 44]
	Ker Arg: Hout, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 24 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]][Tile0, 24:[1x11, 22:1x11, 1x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]][Tile0, 24:[1x11, 22:1x11, 1x3], 1]
		Tile0: [0, 11, 11], Tile1: [11, 11, 11], Tile2; [22, 11, 11]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 24 logical tiles, 1 physical tiles
			Total Size: 29 [Tile0, 24:[29x1, 22:29x1, 29x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 24:[29x1, 22:29x1, 29x1], 1]
		Tile0: [0, 29, 29], Tile1: [0, 29, 29], Tile2; [0, 29, 29]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->StateInOut = (signed char *__restrict__) (denoiser_L1_Memory+768);
	KerArg0->State = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->Xin = (signed char *__restrict__) (denoiser_L1_Memory+1280);
	KerArg0->DimState = (unsigned short int) (256);
	KerArg0->DimIn = (unsigned short int) (256);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+46968);
	KerArg0->FirstCell = (char) ((1));
	KerArg0->Reset = (char) (Reset);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wf+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1536+0), 5632, 0, &DmaR_Evt1);
	_N_Wf=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SCin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+768), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read SCin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SHin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read SHin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Xin+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1280), 256, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Xin */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bf+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12800+0), 44, 0, &DmaR_Evt5);
	_N_Bf=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wi+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12888+0), 5632, 0, &DmaR_Evt6);
	_N_Wi=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bi+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24152+0), 44, 0, &DmaR_Evt7);
	_N_Bi=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wg+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24240+0), 5632, 0, &DmaR_Evt8);
	_N_Wg=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bg+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35504+0), 44, 0, &DmaR_Evt9);
	_N_Bg=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wo+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35592+0), 5632, 0, &DmaR_Evt10);
	_N_Wo=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bo+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46856+0), 44, 0, &DmaR_Evt11);
	_N_Bo=0;
	_C_Hout=0; _SC_Hout=11;
	_SP_Hout=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46968), 29, 0, &DmaR_Evt12);
	AT_L2_WAIT(0, &DmaR_Evt12); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		for (T0Ind=0; T0Ind<24; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==23), T0Ind_NextLast = ((T0Ind+1)==23);
			/*================================= Prepare Tiles ===================================*/
			_SN_Wf = 0;
			if (!(T0Ind_Last)) {
				_N_Wf = _N_Wf + (5632); _SN_Wf = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wf = _N_Wf + (-129536); _SN_Wf = (5632); 
			}
			_SN_Bf = 0;
			if (!(T0Ind_Last)) {
				_N_Bf = _N_Bf + (44); _SN_Bf = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bf = _N_Bf + (-1012); _SN_Bf = (44); 
			}
			_SN_Wi = 0;
			if (!(T0Ind_Last)) {
				_N_Wi = _N_Wi + (5632); _SN_Wi = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wi = _N_Wi + (-129536); _SN_Wi = (5632); 
			}
			_SN_Bi = 0;
			if (!(T0Ind_Last)) {
				_N_Bi = _N_Bi + (44); _SN_Bi = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bi = _N_Bi + (-1012); _SN_Bi = (44); 
			}
			_SN_Wg = 0;
			if (!(T0Ind_Last)) {
				_N_Wg = _N_Wg + (5632); _SN_Wg = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wg = _N_Wg + (-129536); _SN_Wg = (5632); 
			}
			_SN_Bg = 0;
			if (!(T0Ind_Last)) {
				_N_Bg = _N_Bg + (44); _SN_Bg = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bg = _N_Bg + (-1012); _SN_Bg = (44); 
			}
			_SN_Wo = 0;
			if (!(T0Ind_Last)) {
				_N_Wo = _N_Wo + (5632); _SN_Wo = ((T0Ind_NextLast)?1536:5632); 
			} else if (!(1)) {
				_N_Wo = _N_Wo + (-129536); _SN_Wo = (5632); 
			}
			_SN_Bo = 0;
			if (!(T0Ind_Last)) {
				_N_Bo = _N_Bo + (44); _SN_Bo = ((T0Ind_NextLast)?12:44); 
			} else if (!(1)) {
				_N_Bo = _N_Bo + (-1012); _SN_Bo = (44); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Wf */
			if (_SN_Wf) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wf+_N_Wf), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1536+5632*((T0Ind_Total+1)%2)),
						_SN_Wf, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read Bf */
			if (_SN_Bf) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bf+_N_Bf), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12800+44*((T0Ind_Total+1)%2)),
						_SN_Bf, 0, &DmaR_Evt5);
			}
			AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Wi */
			if (_SN_Wi) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wi+_N_Wi), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12888+5632*((T0Ind_Total+1)%2)),
						_SN_Wi, 0, &DmaR_Evt6);
			}
			AT_L2_WAIT(0, &DmaR_Evt7); /* Wait previous DMA read Bi */
			if (_SN_Bi) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bi+_N_Bi), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24152+44*((T0Ind_Total+1)%2)),
						_SN_Bi, 0, &DmaR_Evt7);
			}
			AT_L2_WAIT(0, &DmaR_Evt8); /* Wait previous DMA read Wg */
			if (_SN_Wg) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wg+_N_Wg), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+24240+5632*((T0Ind_Total+1)%2)),
						_SN_Wg, 0, &DmaR_Evt8);
			}
			AT_L2_WAIT(0, &DmaR_Evt9); /* Wait previous DMA read Bg */
			if (_SN_Bg) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bg+_N_Bg), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35504+44*((T0Ind_Total+1)%2)),
						_SN_Bg, 0, &DmaR_Evt9);
			}
			AT_L2_WAIT(0, &DmaR_Evt10); /* Wait previous DMA read Wo */
			if (_SN_Wo) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Wo+_N_Wo), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35592+5632*((T0Ind_Total+1)%2)),
						_SN_Wo, 0, &DmaR_Evt10);
			}
			AT_L2_WAIT(0, &DmaR_Evt11); /* Wait previous DMA read Bo */
			if (_SN_Bo) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bo+_N_Bo), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46856+44*((T0Ind_Total+1)%2)),
						_SN_Bo, 0, &DmaR_Evt11);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->Wf = (signed char *__restrict__) (denoiser_L1_Memory+1536+5632*((T0Ind_Total)%2));
			KerArg0->Bf = (void * __restrict__) (denoiser_L1_Memory+12800+44*((T0Ind_Total)%2));
			KerArg0->Wi = (signed char *__restrict__) (denoiser_L1_Memory+12888+5632*((T0Ind_Total)%2));
			KerArg0->Bi = (void * __restrict__) (denoiser_L1_Memory+24152+44*((T0Ind_Total)%2));
			KerArg0->Wg = (signed char *__restrict__) (denoiser_L1_Memory+24240+5632*((T0Ind_Total)%2));
			KerArg0->Bg = (void * __restrict__) (denoiser_L1_Memory+35504+44*((T0Ind_Total)%2));
			KerArg0->Wo = (signed char *__restrict__) (denoiser_L1_Memory+35592+5632*((T0Ind_Total)%2));
			KerArg0->Bo = (void * __restrict__) (denoiser_L1_Memory+46856+44*((T0Ind_Total)%2));
			KerArg0->Hout = (signed char *__restrict__) (denoiser_L1_Memory+46944+12*((T0Ind_Total)%2));
			KerArg0->Nout = (unsigned short int) (T0Ind_Last?3:11);
			KerArg0->FirstOut = (char) ((T0Ind==0));
			KerArg0->TileOffset = (int) ((T0Ind)*11);
			AT_FORK(gap_ncore(), (void *) LSTM_ParKerB32_SameInStateScale_SQ8, (void *) KerArg0);
			__CALL(LSTM_ParKerB32_SameInStateScale_SQ8, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Hout) AT_L2_WAIT(0, &DmaW_Evt3); /* Wait previous DMA write Hout */
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Hout+_C_Hout), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+46944+12*((T0Ind_Total)%2)),
					_SC_Hout, 1, &DmaW_Evt3);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Hout = _SC_Hout;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Hout = 0;
			if (!(T0Ind_Last)) {
				_C_Hout = _C_Hout + (11); _SC_Hout = ((T0Ind_NextLast)?3:11); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SCout+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+768), 256, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write SCout */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SHout+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 256, 1, &DmaW_Evt2);
	AT_L2_WAIT(0, &DmaW_Evt2); /* Wait DMA write SHout */
	AT_L2_WAIT(0, &DmaW_Evt3); /* Wait previous DMA write Hout */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S92_MatAdd_256x1(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 780 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 256, 256]][Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (denoiser_L1_Memory+256);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+512);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (256);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+768);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 256, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+256), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+768), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+512), 256, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S96_Conv2d_256x256x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 43948 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Total=0, T1Ind_Last, T1Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 4][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In1, Tiled Space: Tile1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 65536 [Tile1, 4:[256x80, 2:256x80, 256x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[256x80, 2:256x80, 256x16], 1]
		Tile0: [0, 20480, 20480], Tile1: [20480, 20480, 20480], Tile2; [40960, 20480, 20480]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile1, 4:[1x80, 2:1x80, 1x16], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 4]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Out, Tiled Space: Tile1
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 80, 80], Tile1: [80, 80, 80], Tile2; [160, 80, 80]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 4:[1x80, 2:1x80, 1x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W_In1 = (unsigned short int) (256);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+43680);
	KerArg0->W_In2 = (unsigned short int) (1);
	KerArg0->W_Out = (unsigned short int) (1);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+42656);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+43936);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+0), 20480, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+43680), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+40960), 1024, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=80;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42144), 256, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42400), 256, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+43936), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T1Ind=0; T1Ind<4; T1Ind++, T1Ind_Total++) { /* Iteration on Tile1 */
		int T1Ind_Last = (T1Ind==3), T1Ind_NextLast = ((T1Ind+1)==3);
		/*================================= Prepare Tiles ===================================*/
		_SN_In1 = 0;
		if (!(T1Ind_Last)) {
			_N_In1 = _N_In1 + (20480); _SN_In1 = ((T1Ind_NextLast)?4096:20480); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
		if (_SN_In1) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+20480*((T1Ind_Total+1)%2)),
					_SN_In1, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0+20480*((T1Ind_Total)%2));
			KerArg0->H_In1 = (unsigned short int) (T1Ind_Last?16:80);
			KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+40960+(320*(T1Ind)));
			KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+42144+(80*(T1Ind)));
			KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+42400+(80*(T1Ind)));
			KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+41984+80*((T1Ind_Total)%2));
			KerArg0->OutFirstCol = (unsigned short int) ((0)*1);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+43936))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+41984+80*((T1Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T1Ind_Last)) {
			_C_Out = _C_Out + (80); _SC_Out = ((T1Ind_NextLast)?16:80); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S99_Op_Resize_196(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 1280 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeNearestNeighborSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[1x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x256], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[4x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4x256], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->Win = (unsigned int) (1);
	KerArg0->Hin = (unsigned int) (256);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+256);
	KerArg0->Wout = (unsigned int) (4);
	KerArg0->Hout = (unsigned int) (256);
	KerArg0->HTileOut = (unsigned int) (4);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 256, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->FirstLineIndex = (unsigned int) ((65280*(0)*255)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeNearestNeighborSigned, (void *) KerArg0);
		__CALL(KerResizeNearestNeighborSigned, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+256), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S104_Conv2d_128x256x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 45132 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _LN_Filter;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 128, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 256, Tiled: 7]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 1024 [D0, [6 x 160, 64]][Tile0, 1:[4x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4x1], 1][D0, [6 x 160, 64]]
		Tile0: [0, 160, 4], Tile1: [160, 160, 4], Tile2; [320, 160, 4]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D1, [0 x 512, 512]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 512, 512]]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 131072 [D1, [0 x 131072, 131072]][D0, [6 x 20480, 8192]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 131072, 131072]][D0, [6 x 20480, 8192]]
		Tile0: [0, 20480, 160], Tile1: [160, 20480, 160], Tile2; [320, 20480, 160]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D1, [0 x 512, 512]][Tile0, 1:[4x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 512, 512]][Tile0, 1:[4x1], 1]
		Tile0: [0, 512, 4], Tile1: [0, 512, 4], Tile2; [0, 512, 4]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D1, [0 x 2048, 2048]][Tile0, 1:[4x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2048, 2048]][Tile0, 1:[4x1], 4]
		Tile0: [0, 2048, 16], Tile1: [0, 2048, 16], Tile2; [0, 2048, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+43072);
	KerArg0->W = (unsigned short int) (4);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (128);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+320);
	KerArg1->W = (unsigned short int) (4);
	KerArg1->UsedW = (unsigned short int) (4);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->UsedH = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (128);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+43072);
	KerArg1->Pad = (v4s) ((v4s){0,3,0,0});
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+43072);
	KerArg2->Feat = (unsigned short int) (128);
	KerArg2->W = (unsigned short int) (4);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+832);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+960);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+45120);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+0), 160, 4, 4, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+320), 512, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+832), 128, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+960), 128, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1088+0), 20480, 1024, 160, 0, &DmaR_Evt5);
	_N_Filter=0;
	_C_Out=0; _SC_Out=512; _LC_Out=4;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+45120), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+45120))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<7; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==6), D0Ind_NextLast = ((D0Ind+1)==6);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (160); _LN_In = (4); _SN_In = (((D0Ind_NextLast)?16:40)*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-960); _LN_In = (4); _SN_In = (40*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + (160); _LN_Filter = ((D0Ind_NextLast)?64:160); _SN_Filter = (128*_LN_Filter); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + (-960); _LN_Filter = (160); _SN_Filter = (128*_LN_Filter); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0+160*((D0Ind_Total+1)%2)),
							_SN_In, 4, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1088+20480*((D0Ind_Total+1)%2)),
							_SN_Filter, 1024, _LN_Filter, 0, &DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0+160*((D0Ind_Total)%2));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?16:40);
				KerArg1->TotalInFeatures = (unsigned short int) (D0Ind_Last?16:40);
				KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+1088+20480*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv4x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv4x1Stride1x1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+42048+512*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+42048+512*((T0Ind_Total)%2)),
					_SC_Out, 4, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			
			/*============================= End Prepare Tiles ===================================*/
			T0Ind_Total++;
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S106_MatAdd_128x4(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 1548 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 512, 512]][Tile0, 1:[128x4], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (denoiser_L1_Memory+512);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+1024);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (128);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+1536);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 512, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+512), 512, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1536), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 512, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S110_Conv2d_128x128x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18700 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[512x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[512x1], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16384 [Tile1, 1:[128x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[128x128], 1]
		Tile0: [0, 16384, 16384], Tile1: [0, 16384, 16384], Tile2; [0, 16384, 16384]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[128x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[128x4], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile1, 1:[1x128], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 4]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile1, 1:[4x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[4x128], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile1, 1:[1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile1, 1:[1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x128], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (128);
	KerArg0->H_In1 = (unsigned short int) (128);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+18176);
	KerArg0->W_In2 = (unsigned short int) (4);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+16384);
	KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+17408);
	KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+17536);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+16896);
	KerArg0->W_Out = (unsigned short int) (4);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+17664);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+18688);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 16384, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+18176), 512, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16384), 512, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+17408), 128, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+17536), 128, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+18688), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*4);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+18688))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16896), 512, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S113_Op_Resize_255(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 2560 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeNearestNeighborSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 512 [Tile0, 1:[4x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[4x128], 1]
		Tile0: [0, 512, 512], Tile1: [0, 512, 512], Tile2; [0, 512, 512]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[16x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x128], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->Win = (unsigned int) (4);
	KerArg0->Hin = (unsigned int) (128);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+512);
	KerArg0->Wout = (unsigned int) (16);
	KerArg0->Hout = (unsigned int) (128);
	KerArg0->HTileOut = (unsigned int) (16);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 512, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->FirstLineIndex = (unsigned int) ((65024*(0)*127)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeNearestNeighborSigned, (void *) KerArg0);
		__CALL(KerResizeNearestNeighborSigned, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+512), 2048, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S118_Conv2d_64x128x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 40332 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 128, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D1, [0 x 4096, 4096]][Tile0, 1:[16x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4096, 4096]][Tile0, 1:[16x1], 4]
		Tile0: [0, 4096, 64], Tile1: [0, 4096, 64], Tile2; [0, 4096, 64]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32768 [D1, [0 x 32768, 32768]][D0, [0 x 32768, 32768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32768, 32768]][D0, [0 x 32768, 32768]]
		Tile0: [0, 32768, 32768], Tile1: [0, 32768, 32768], Tile2; [0, 32768, 32768]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D1, [0 x 1024, 1024]][Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1024, 1024]][Tile0, 1:[16x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D0, [0 x 2048, 2048]][Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1][D0, [0 x 2048, 2048]]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+36224);
	KerArg0->W = (unsigned short int) (16);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (64);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+2048);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (16);
	KerArg1->UsedW = (unsigned short int) (16);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (128);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->TotalInFeatures = (unsigned short int) (128);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+2432);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+36224);
	KerArg1->Pad = (v4s) ((v4s){0,3,0,0});
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+36224);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+35200);
	KerArg2->Feat = (unsigned short int) (64);
	KerArg2->W = (unsigned short int) (16);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+2304);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+2368);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+40320);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2048), 256, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2304), 64, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2368), 64, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2432), 32768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 2048, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+40320), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+40320))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv4x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv4x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+35200), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S120_MatAdd_64x16(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 3084 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1024, 1024]][Tile0, 1:[64x16], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (denoiser_L1_Memory+1024);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+2048);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (64);
	KerArg0->H = (unsigned short int) (16);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+3072);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1024, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+3072), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2048), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S124_Conv2d_64x64x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 6796 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile1, 1:[64x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[64x64], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[64x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x16], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile1, 1:[1x64], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 4]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile1, 1:[16x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x64], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[1x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[1x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x64], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (64);
	KerArg0->H_In1 = (unsigned short int) (64);
	KerArg0->In2 = (signed char * __restrict__) (denoiser_L1_Memory+5760);
	KerArg0->W_In2 = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+4096);
	KerArg0->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+5376);
	KerArg0->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+5440);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+4352);
	KerArg0->W_Out = (unsigned short int) (16);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (denoiser_L1_Memory+5504);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+6784);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5760), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5376), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5440), 64, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+6784), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*16);
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+6784))[5]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4352), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S127_Op_Resize_314(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 5120 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeNearestNeighborSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[16x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x64], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[64x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x64], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->Win = (unsigned int) (16);
	KerArg0->Hin = (unsigned int) (64);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+1024);
	KerArg0->Wout = (unsigned int) (64);
	KerArg0->Hout = (unsigned int) (64);
	KerArg0->HTileOut = (unsigned int) (64);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1024, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->FirstLineIndex = (unsigned int) ((64512*(0)*63)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeNearestNeighborSigned, (void *) KerArg0);
		__CALL(KerResizeNearestNeighborSigned, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 4096, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S132_Conv2d_32x64x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 22732 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 64, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8192 [D1, [0 x 8192, 8192]][Tile0, 1:[64x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 8192, 8192]][Tile0, 1:[64x1], 4]
		Tile0: [0, 8192, 256], Tile1: [0, 8192, 256], Tile2; [0, 8192, 256]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8192 [D1, [0 x 8192, 8192]][D0, [0 x 8192, 8192]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 8192, 8192]][D0, [0 x 8192, 8192]]
		Tile0: [0, 8192, 8192], Tile1: [0, 8192, 8192], Tile2; [0, 8192, 8192]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D1, [0 x 2048, 2048]][Tile0, 1:[64x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2048, 2048]][Tile0, 1:[64x1], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D0, [0 x 4096, 4096]][Tile0, 1:[64x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x1], 1][D0, [0 x 4096, 4096]]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+14528);
	KerArg0->W = (unsigned short int) (64);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+4096);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (64);
	KerArg1->UsedW = (unsigned short int) (64);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (64);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+4288);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+14528);
	KerArg1->Pad = (v4s) ((v4s){0,3,0,0});
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+14528);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+12480);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->W = (unsigned short int) (64);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+4224);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+4256);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+22720);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 128, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4224), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4256), 32, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4288), 8192, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+22720), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+22720))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv4x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv4x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12480), 2048, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S134_MatAdd_32x64(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 6156 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2048, 2048]][Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (denoiser_L1_Memory+2048);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+4096);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (32);
	KerArg0->H = (unsigned short int) (64);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+6144);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 2048, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2048), 2048, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+6144), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 2048, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S138_Conv2d_32x32x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 7372 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 64], Tile1: [0, 2048, 64], Tile2; [0, 2048, 64]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [Tile0, 1:[32x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x32], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[32x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 4]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[32x64], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x64], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[32x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[32x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x1], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+1024);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+3072);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (64);
	KerArg0->H = (unsigned short int) (32);
	KerArg1->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (32);
	KerArg1->H_In1 = (unsigned short int) (32);
	KerArg1->In2 = (signed char * __restrict__) (denoiser_L1_Memory+3072);
	KerArg1->W_In2 = (unsigned short int) (64);
	KerArg1->Bias = (void * __restrict__) (denoiser_L1_Memory+5120);
	KerArg1->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+7296);
	KerArg1->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+7328);
	KerArg1->Out = (signed char * __restrict__) (denoiser_L1_Memory+5248);
	KerArg1->Infos = (signed char *__restrict__) (denoiser_L1_Memory+7360);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+1024), 2048, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5120), 128, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7296), 32, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7328), 32, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+7360), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+7360))[5]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+5248), 2048, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S141_Op_Resize_373(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 10240 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeNearestNeighborSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [Tile0, 1:[64x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x32], 1]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8192 [Tile0, 1:[256x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x32], 1]
		Tile0: [0, 8192, 8192], Tile1: [0, 8192, 8192], Tile2; [0, 8192, 8192]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->Win = (unsigned int) (64);
	KerArg0->Hin = (unsigned int) (32);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+2048);
	KerArg0->Wout = (unsigned int) (256);
	KerArg0->Hout = (unsigned int) (32);
	KerArg0->HTileOut = (unsigned int) (256);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 2048, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->FirstLineIndex = (unsigned int) ((63488*(0)*31)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeNearestNeighborSigned, (void *) KerArg0);
		__CALL(KerResizeNearestNeighborSigned, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+2048), 8192, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S146_Conv2d_16x32x1x4_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 30828 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 32, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16384 [D1, [0 x 16384, 16384]][Tile0, 1:[256x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16384, 16384]][Tile0, 1:[256x1], 4]
		Tile0: [0, 16384, 1024], Tile1: [0, 16384, 1024], Tile2; [0, 16384, 1024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2048 [D1, [0 x 2048, 2048]][D0, [0 x 2048, 2048]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2048, 2048]][D0, [0 x 2048, 2048]]
		Tile0: [0, 2048, 2048], Tile1: [0, 2048, 2048], Tile2; [0, 2048, 2048]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D1, [0 x 4096, 4096]][Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4096, 4096]][Tile0, 1:[256x1], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8192 [D0, [0 x 8192, 8192]][Tile0, 1:[256x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x1], 1][D0, [0 x 8192, 8192]]
		Tile0: [0, 8192, 8192], Tile1: [0, 8192, 8192], Tile2; [0, 8192, 8192]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+14432);
	KerArg0->W = (unsigned short int) (256);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+8192);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (256);
	KerArg1->UsedW = (unsigned short int) (256);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (32);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+8288);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+14432);
	KerArg1->Pad = (v4s) ((v4s){0,3,0,0});
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+14432);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+10336);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (256);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+8256);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+8272);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+30816);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8192), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8256), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8272), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8288), 2048, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 8192, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+30816), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+30816))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv4x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv4x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+10336), 4096, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S148_MatAdd_16x256(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12300 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4096, 4096]][Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (denoiser_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (denoiser_L1_Memory+4096);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+8192);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (16);
	KerArg0->H = (unsigned short int) (256);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (denoiser_L1_Memory+12288);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 4096, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12288), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8192), 4096, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S152_Conv2d_16x16x1x1_Relu(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12652 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 256], Tile1: [0, 4096, 256], Tile2; [0, 4096, 256]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[16x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x16], 1]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[16x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 4]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[16x256], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x256], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (denoiser_L1_Memory+256);
	KerArg0->Out = (signed char *__restrict__) (denoiser_L1_Memory+4352);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (256);
	KerArg0->H = (unsigned short int) (16);
	KerArg1->In1 = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (16);
	KerArg1->H_In1 = (unsigned short int) (16);
	KerArg1->In2 = (signed char * __restrict__) (denoiser_L1_Memory+4352);
	KerArg1->W_In2 = (unsigned short int) (256);
	KerArg1->Bias = (void * __restrict__) (denoiser_L1_Memory+8448);
	KerArg1->Scale = (unsigned char * __restrict__) (denoiser_L1_Memory+12608);
	KerArg1->ScaleN = (unsigned char * __restrict__) (denoiser_L1_Memory+12624);
	KerArg1->Out = (signed char * __restrict__) (denoiser_L1_Memory+8512);
	KerArg1->Infos = (signed char *__restrict__) (denoiser_L1_Memory+12640);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+256), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8448), 64, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12608), 16, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12624), 16, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+12640), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+12640))[5]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+8512), 4096, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S155_Op_Resize_432(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 20480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeNearestNeighborSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [Tile0, 1:[256x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[256x16], 1]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16384 [Tile0, 1:[1024x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x16], 1]
		Tile0: [0, 16384, 16384], Tile1: [0, 16384, 16384], Tile2; [0, 16384, 16384]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg0->Win = (unsigned int) (256);
	KerArg0->Hin = (unsigned int) (16);
	KerArg0->Out = (signed char * __restrict__) (denoiser_L1_Memory+4096);
	KerArg0->Wout = (unsigned int) (1024);
	KerArg0->Hout = (unsigned int) (16);
	KerArg0->HTileOut = (unsigned int) (1024);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 4096, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->FirstLineIndex = (unsigned int) ((61440*(0)*15)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeNearestNeighborSigned, (void *) KerArg0);
		__CALL(KerResizeNearestNeighborSigned, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+4096), 16384, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S160_Conv2d_1x16x1x4(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 21592 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 1, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D1, [0 x 4096, 4096]][Tile0, 1:[1024x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4096, 4096]][Tile0, 1:[1024x1], 4]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4 [D1, [0 x 4, 4]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4, 4]]
		Tile0: [0, 4, 4], Tile1: [0, 4, 4], Tile2; [0, 4, 4]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [D1, [0 x 1, 1]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1, 1]]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [D1, [0 x 1, 1]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1, 1]]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]][D0, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]][D0, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D1, [0 x 1024, 1024]][Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1024, 1024]][Tile0, 1:[1024x1], 1]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16384 [D0, [0 x 16384, 16384]][Tile0, 1:[1024x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1024x1], 1][D0, [0 x 16384, 16384]]
		Tile0: [0, 16384, 16384], Tile1: [0, 16384, 16384], Tile2; [0, 16384, 16384]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (denoiser_L1_Memory+17484);
	KerArg0->W = (unsigned short int) (1024);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->Bias = (void * __restrict__) (denoiser_L1_Memory+16384);
	KerArg1->In = (signed char * __restrict__) (denoiser_L1_Memory+0);
	KerArg1->W = (unsigned short int) (1024);
	KerArg1->UsedW = (unsigned short int) (1024);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (1);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (denoiser_L1_Memory+16396);
	KerArg1->Out = (int * __restrict__) (denoiser_L1_Memory+17484);
	KerArg1->Pad = (v4s) ((v4s){0,3,0,0});
	KerArg2->In = (int *__restrict__) (denoiser_L1_Memory+17484);
	KerArg2->Out = (void *__restrict__) (denoiser_L1_Memory+16460);
	KerArg2->Feat = (unsigned short int) (1);
	KerArg2->W = (unsigned short int) (1024);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (denoiser_L1_Memory+16388);
	KerArg2->ScaleN = (unsigned char *__restrict__) (denoiser_L1_Memory+16392);
	KerArg2->Infos = (signed char *__restrict__) (denoiser_L1_Memory+21580);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16384), 4, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16388), 1, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16392), 1, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16396), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+0), 16384, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+21580), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(denoiser_L1_Memory+21580))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv4x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv4x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) denoiser_L1_Memory+16460), 1024, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S71_Op_LSTM_95(
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
		signed char  Reset)


{
/*
	KerArg:                         Cinout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                         Hinout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                            Xin, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                             Wf, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bf, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wi, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bi, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wg, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bg, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wo, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bo, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                           Hout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                          Infos, Total Size:       29, Dimension:   1, Items:       29, ItemSize:  1
	KerArg:                          Reset, Total Size:        1, Dimension:   1, Items:        1, ItemSize:  1
*/
	S71_Op_LSTM_95G2(
		(signed char * __restrict__) (Cinout),
		(signed char * __restrict__) (Hinout),
		(signed char * __restrict__) (Cinout),
		(signed char * __restrict__) (Hinout),
		(signed char * __restrict__) (Xin + 0),
		(signed char * __restrict__) (Wf + 0),
		(int * __restrict__) (Bf + 0),
		(signed char * __restrict__) (Wi + 0),
		(int * __restrict__) (Bi + 0),
		(signed char * __restrict__) (Wg + 0),
		(int * __restrict__) (Bg + 0),
		(signed char * __restrict__) (Wo + 0),
		(int * __restrict__) (Bo + 0),
		(signed char * __restrict__) (Hout),
		(signed char * __restrict__) (Infos + 0),
		(signed char ) (Reset)
	);
}

void S88_Op_LSTM_161(
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
		signed char  Reset)


{
/*
	KerArg:                         Cinout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                         Hinout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                            Xin, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                             Wf, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bf, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wi, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bi, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wg, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bg, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                             Wo, Total Size:   131072, Dimension:   1, Items:   131072, ItemSize:  1
	KerArg:                             Bo, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  4
	KerArg:                           Hout, Total Size:      256, Dimension:   1, Items:      256, ItemSize:  1
	KerArg:                          Infos, Total Size:       29, Dimension:   1, Items:       29, ItemSize:  1
	KerArg:                          Reset, Total Size:        1, Dimension:   1, Items:        1, ItemSize:  1
*/
	S88_Op_LSTM_161G2(
		(signed char * __restrict__) (Cinout),
		(signed char * __restrict__) (Hinout),
		(signed char * __restrict__) (Cinout),
		(signed char * __restrict__) (Hinout),
		(signed char * __restrict__) (Xin + 0),
		(signed char * __restrict__) (Wf + 0),
		(int * __restrict__) (Bf + 0),
		(signed char * __restrict__) (Wi + 0),
		(int * __restrict__) (Bi + 0),
		(signed char * __restrict__) (Wg + 0),
		(int * __restrict__) (Bg + 0),
		(signed char * __restrict__) (Wo + 0),
		(int * __restrict__) (Bo + 0),
		(signed char * __restrict__) (Hout),
		(signed char * __restrict__) (Infos + 0),
		(signed char ) (Reset)
	);
}

