#include "AutoTilerLib.h"
#include "AutoTilerLibTypes.h"
#include "DSP_Generators.h"


void FFTConfiguration(unsigned int L1Memory)
{
  SetInlineMode(ALWAYS_INLINE);
	SetSymbolDynamics();

  SetUsedFilesNames(0, 1, "DSP_Lib.h");
  SetGeneratedFilesNames("RFFTKernels.c", "RFFTKernels.h");

  SetL1MemorySize(L1Memory);
}

int main(int argc, char **argv)
{
  	if (TilerParseOptions(argc, argv)) GenTilingError("Failed to initialize or incorrect output arguments directory.\n");

    // Set Auto Tiler configuration, given shared L1 memory is 51200
    FFTConfiguration(51200);
    // Load FIR basic kernels
    LoadMFCCLibrary();
    // Generate code for STFT and iSTFT applied to N_FRAME of size FRAME_SIZE with FRAME_STEP as stride
    RFFT_2D_Generator(
      "STFT",     // name 
      0,          // ctrl
      1,          // all the frames
      FRAME_SIZE, // frame size
      FRAME_STEP, // frame stride
      N_FFT,      // Nfft
      0,          // PreempFactor
      0,          // SkipPreemp
      0,          // NoWindow
      1,          // OutFFT
      0,          // MagSquared
      FLOAT16     // datatype
    );

    IRFFT_2D_Generator(
      "iSTFT",     // name 
      0,          // ctrl
      1,          // all the frames
      N_FFT,      // Nfft
      0,          // InvertWindow, bypassed and manually inserted
      FLOAT16     // datatype
    );

    GenerateTilingCode();
}
