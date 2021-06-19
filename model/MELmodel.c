#include "AutoTilerLib.h"
#include "AutoTilerLibTypes.h"
#include "DSP_Generators.h"
#include "MEL_params.h"


void MFCCConfiguration(unsigned int L1Memory)
{
    SetInlineMode(ALWAYS_INLINE);
    SetSymbolDynamics();
    SetUsedFilesNames(0, 3, "MfccBasicKernels.h", "CmplxFunctions.h", "PreProcessing.h");

    SetGeneratedFilesNames("MFCCKernels.c", "MFCCKernels.h");

    SetL1MemorySize(L1Memory);
}

int main(int argc, char **argv)
{
  	if (TilerParseOptions(argc, argv)) GenTilingError("Failed to initialize or incorrect output arguments directory.\n");
    // Set Auto Tiler configuration, given shared L1 memory is 51200
    MFCCConfiguration(51200);
    // Load FIR basic kernels
    LoadMFCCLibrary();

    CNN_GenControl_T gen_ctrl;
    CNN_InitGenCtrl(&gen_ctrl);
    CNN_SetGenCtrl(&gen_ctrl, "MFCC_COMPUTE_DB", AT_OPT_VAL(1));

    MFCC_Generator("MEL", &gen_ctrl , -1, FRAME_SIZE, FRAME_STEP, N_FFT, MFCC_BANK_CNT ,MFCC_COEFF_CNT, N_DCT, 0, 0, 0, USE_POWER, DATA_TYPE, 2, 0);

    GenerateTilingCode();
}
