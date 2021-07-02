/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

/* Autotiler includes. */
#include "Gap.h"
#include "denoiser.h"
#include "wavIO.h"
#include "fs_switch.h"

#include "RFFTKernels.h"
#include "denoiserKernels.h"

#include <math.h>



#define  WAV_BUFFER_SIZE    16000 // 1sec@16kHz
#define  NUM_CLASSES        12

// DCT_NORMALIZATION        -> np.sqrt(1/(N_DCT))*0.5
// NNTOOL_INPUT_SCALE_FLOAT -> 1.9372712
// SCALE = NNTOOL_INPUT_SCALE_FLOAT*DCT_NORMALIZATION
#define  INPUT_SCALE        236
#define  INPUT_SCALEN       16

#define NB_ELEM 8000
#define BUFF_SIZE (NB_ELEM*2)
#define ITER    2

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

#ifdef GAPUINO
    struct pi_device gpio;
    #define GPIO_OUT PI_GPIO_A1_PAD_13_B2 
    #define NMAX_ITER 5
    int iter = 0;
#endif



#define DATATYPE_SIGNAL f16

#if IS_INPUT_STFT == 0 
    //load the input audio signal and compute the MFCC
    #include "TwiddlesDef.h"
    #include "RFFTTwiddlesDef.h"
    #include "SwapTablesDef.h"
    #include "WinLUT.def"
    #include "WinLUT_f32.def"
    #ifdef __gap9__
    #include "WinLUT_f16.def"
    #endif

// input oputput signals dynamically allocated
#if IS_FAKE_SIGNAL_IN == 1
    #define TOT_FRAMES 1
#else 
    // allocate space to load the input signal
    #define AUDIO_BUFFER_SIZE ((TOT_FRAMES+NUM_FRAME_OVERLAP)*FRAME_STEP)
    char *WavName = NULL;
    PI_L2 short int inSig[AUDIO_BUFFER_SIZE];
    PI_L2 short int outSigt[AUDIO_BUFFER_SIZE];
#endif

#else
    // here the allocation in caso of stft inputs
    #if IS_FAKE_SIGNAL_IN == 1
        #define TOT_FRAMES 1
    #else 
        // allocate space to load the input signal
        char *WavName = NULL;
    #endif
#endif


// computation buffers
PI_L2 DATATYPE_SIGNAL Audio_Frame[FRAME_NFFT];  // only first FRAME_SIZE samples (<FRAME_NFFT) are valid
PI_L2 DATATYPE_SIGNAL STFT_Spectrogram[AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2];   // FIXME: must be double in case float values are loaded from file
PI_L2 DATATYPE_SIGNAL STFT_Magnitude[AT_INPUT_WIDTH*AT_INPUT_HEIGHT];

//PI_L2 DATATYPE_SIGNAL LSTM_STATE_0_I[257];
//PI_L2 DATATYPE_SIGNAL LSTM_STATE_0_C[257];
//PI_L2 DATATYPE_SIGNAL LSTM_STATE_1_I[257];
//PI_L2 DATATYPE_SIGNAL LSTM_STATE_1_C[257];

#if IS_INPUT_STFT == 0 ///load the input audio signal and compute the MFCC

static void RunSTFT()
{
#ifdef PERF
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#endif
    unsigned int ta = gap_cl_readhwtimer();

//    STFT(
//        Audio_Frame, 
//        STFT_Spectrogram, 
//        R2_Twiddles_fix_256,   
//        RFFT_Twiddles_fix_512,   
//        R2_SwapTable_fix_256, 
//        WinLUT
//    );
//    STFT(Audio_Frame, STFT_Spectrogram, R2_Twiddles_fix_256,   RFFT_Twiddles_fix_512,   R2_SwapTable_fix_256, WinLUT, PreempShift);
//    STFT(Audio_Frame, STFT_Spectrogram, R2_Twiddles_float_256, RFFT_Twiddles_float_512, R2_SwapTable_fix_256, WinLUT_f32);
    STFT(
        Audio_Frame, 
        STFT_Spectrogram, 
        R4_Twiddles_f16_256,   
        RFFT_Twiddles_f16_512,   
        R4_SwapTable_fix_256, 
        WinLUT_f16
    );

    
    unsigned int ti = gap_cl_readhwtimer() - ta;

    PRINTF("%45s: Cycles: %10d\n","STFT: ", ti );

    ta = gap_cl_readhwtimer();
    // compute the magnitude of the STFT components
    for (int i=0; i<AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++){
        f16 STFT_Real_Part = STFT_Spectrogram[2*i];
        f16 STFT_Imag_Part = STFT_Spectrogram[2*i+1];
//        STFT_Magnitude[i] = (f16) (sqrt((float) (STFT_Real_Part*STFT_Real_Part + STFT_Imag_Part*STFT_Imag_Part) ));
        STFT_Magnitude[i] = __builtin_pulp_f16sqrt (STFT_Real_Part*STFT_Real_Part + STFT_Imag_Part*STFT_Imag_Part);
    }
    ti = gap_cl_readhwtimer() - ta;

    PRINTF("%45s: Cycles: %10d\n","Magnitude Compute: ", ti );
}

#include "istft_window.h"

static void RuniSTFT()
{
#ifdef PERF
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#endif
    unsigned int ta, ti;

    ta = gap_cl_readhwtimer();

#ifdef APPLY_DENOISER
    /****
        Apply filtering on the STFT map
    ****/
    ta = gap_cl_readhwtimer();
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * STFT_Magnitude[i];
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * STFT_Magnitude[i];
        }
    ti = gap_cl_readhwtimer() - ta;

    PRINTF("\nSTFT Filtered: ");
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2; i++ ){
        printf("%f, ", STFT_Spectrogram[i]);
    }
    PRINTF("\n");

    PRINTF("%45s: Cycles: %10d\n","iScaling: ", ti );

#endif // apply scaling


    /****
        Inverse iSTFT
    ****/
    ta = gap_cl_readhwtimer();
   
    iSTFT(
        STFT_Spectrogram, 
        STFT_Spectrogram, 
        R2_Twiddles_f16_256,   
        RFFT_Twiddles_f16_512,   
        R2_SwapTable_fix_256
    );

    
    ti = gap_cl_readhwtimer() - ta;
    PRINTF("%45s: Cycles: %10d\n","iSTFT: ", ti );

    /****
        Inverse Hanning Filtering
    ****/
    ta = gap_cl_readhwtimer();
    // applying inverse hanning windowing
    for(int i=0;i<FRAME_SIZE;i++){
//        printf("%f *", STFT_Spectrogram[i]);
        Audio_Frame[i] = hanning_inv[i] * STFT_Spectrogram[i];
//        printf(" %f(%f) --> %f\n", (DATATYPE_SIGNAL) hanning_inv[i],hanning_inv[i],Audio_Frame[i]);
    }
    
    ti = gap_cl_readhwtimer() - ta;
    PRINTF("%45s: Cycles: %10d\n","iHanning: ", ti );

}



#endif

PI_L2 int ResetLSTM;

static void RunDenoiser()
{
// L1_Memory = __PREFIX(_L1_Memory);

  PRINTF("Running on cluster\n");
#ifdef PERF
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
#ifdef GAPUINO
  pi_gpio_pin_write(&gpio, GPIO_OUT, 1 );
#endif

  __PREFIX(CNN)(
        STFT_Magnitude,  
//        LSTM_STATE_0_I,
//        LSTM_STATE_0_C,
//        LSTM_STATE_1_I,
//        LSTM_STATE_1_C,
        ResetLSTM, 
        STFT_Magnitude
    );

#ifdef GAPUINO
  pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
#endif
}



switch_file_t File = (switch_file_t) 0;
switch_fs_t fs;

void denoiser(void)
{
    PRINTF("Entering main controller\n");
#ifdef NN_INF_NOT
    PRINTF("NN_INF_NOT is defined\n");
#endif
#ifdef APPLY_DENOISER
    PRINTF("APPLY_DENOISER is defined\n");
#endif

    // Voltage-Frequency settings
    uint32_t voltage =1200;
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
//    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 300*1000*1000);
    //PMU_set_voltage(voltage, 0);
    PRINTF("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
        (float)voltage/1000, FREQ_FC, FREQ_CL);
//    pulp_write32(0x1A10414C,1);   // what is this?


#ifdef GAPUINO
	//configuring gpio
	struct pi_gpio_conf gpio_conf = {0};
    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio, &gpio_conf);
    int errors = pi_gpio_open(&gpio);
    if (errors)
    {
        PRINTF("Error opening GPIO %d\n", errors);
        pmsis_exit(errors);
    }
    /* Configure gpio input. */
    pi_gpio_pin_configure(&gpio, GPIO_OUT, PI_GPIO_OUTPUT);
    pi_pad_set_function(PI_PAD_13_B2_RF_PACTRL1, PI_PAD_13_B2_GPIO_A1_FUNC1  );

    pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
#endif


    /****
        Configure And open cluster. 
    ****/
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }


    /******
        Setup STFT/ISTF task
    ******/
    struct pi_cluster_task *task_stft = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    if (!task_stft) {
        PRINTF("failed to allocate memory for task\n");
    }

    memset(task_stft, 0, sizeof(struct pi_cluster_task));
    task_stft->stack_size = STACK_SIZE;
    task_stft->slave_stack_size = SLAVE_STACK_SIZE;
    task_stft->arg = NULL;

#ifndef NN_INF_NOT
    /******
        Setup Denoiser NN inference task
    ******/
    printf("Setup Cluster Task for inference!\n");
    struct pi_cluster_task *task_net = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    if(task_net==NULL) {
      PRINTF("pi_cluster_task alloc Error!\n");
      pmsis_exit(-1);
    }
    PRINTF("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
    memset(task_net, 0, sizeof(struct pi_cluster_task));
    task_net->entry = &RunDenoiser;
    task_net->stack_size = STACK_SIZE;
    task_net->slave_stack_size = SLAVE_STACK_SIZE;
    task_net->arg = NULL;
    
    // Reset LSTM
    ResetLSTM = 1;

    /******
        Denoiser NN constructor
    ******/
    PRINTF("\n\nDenoiser Constructor\n");
    int err_construct = __PREFIX(CNN_Construct)();
    if (err_construct)
    {
        PRINTF("Graph constructor exited with error: %d\n", err_construct);
        pmsis_exit(-5);
    }
    printf("Denoiser Contrcuctor OK! The L1 memory base is: %x\n",denoiser_L1_Memory);

#endif // NN_INF_NOT


#if IS_INPUT_STFT == 0 

/****
    load the input audio signal and compute the MFCC
****/

#if IS_FAKE_SIGNAL_IN == 1
    // load fake data into Audio_Frame: a single frame of lenght FRAME_SIZE
    for (int i=0;i<FRAME_SIZE;i++){
        Audio_Frame[i] = 0;
    }
#else
    PRINTF("Reading wav...\n");
    header_struct header_info;
    if (ReadWavFromFile("../../../samples/sample_0000.wav", inSig, AUDIO_BUFFER_SIZE*sizeof(float), &header_info)){
        PRINTF("Error reading wav file\n");
        pmsis_exit(1);
    }
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    PRINTF("Num Samples: %d\n",num_samples);
    PRINTF("BitsPerSample: %d\n",header_info.BitsPerSample);

    PRINTF("Finished Read wav.\n");

    // Cast if needed
    PRINTF("Audio In: ");
    for (int i= 0 ; i<FRAME_SIZE; i++){
//        printf("%f", ((float)inSig[i])/(1<<15) );
        Audio_Frame[i] = ((DATATYPE_SIGNAL) inSig[i] )/(1<<15);
        PRINTF("%f, ", Audio_Frame[i] );
    }
#endif

    /******
        MFCC Task
    ******/
    // compute mfcc if not read from file

    PRINTF("\n\n****** Computing STFT ***** \n");
    task_stft->entry = &RunSTFT;

    L1_Memory = pmsis_l1_malloc(_L1_Memory_SIZE);
    if (L1_Memory==NULL){
        printf("Error allocating L1\n");
        pmsis_exit(-1);
    }

    pi_cluster_send_task_to_cl(&cluster_dev, task_stft);
    pmsis_l1_malloc_free(L1_Memory,_L1_Memory_SIZE);

    /***
        Check the Spectrogram Results
    ***/
    PRINTF("\nSTFT OUT: ");
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2; i++ ){
        printf("%f, ",STFT_Spectrogram[i]);
    }
    PRINTF("\n");

    PRINTF("\nMagnitude OUT: ");
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
        printf("%f, ",STFT_Magnitude[i]);
    }
    PRINTF("\n");

    

#else ///load the STFT

#if IS_FAKE_SIGNAL_IN == 1
    // load fake data into STFT_Spectrogram_in
    PRINTF("Loading a fake zeroed STFT...\n");
    for (int i=0;i<AT_INPUT_WIDTH*AT_INPUT_HEIGHT;i++){
        STFT_Spectrogram[i] = 0;
    }
#else

    // open FS and read the binary files with STFT (flaot values)
    __FS_INIT(fs);

    for(int frame_id = 0; frame_id<TOT_FRAMES; frame_id++){

        PRINTF("Reading STFT file %.4d/%d...\n", frame_id, TOT_FRAMES );
        sprintf(WavName, "../../../samples/mags_%.4d.bin",frame_id);
        printf("File being read is : %s\n", WavName);

        File = __OPEN_READ(fs, WavName);
        if (File == 0) {
            printf("Failed to open file, %s\n", WavName); 
            pmsis_exit(7);
        }
        printf("File %x of size %d\n", File, sizeof(switch_file_t));

        int TotBytes = sizeof(float)*AT_INPUT_WIDTH;
        int len = __READ(File, STFT_Spectrogram, TotBytes);
        if (len != TotBytes){
            printf("Too few bytes in %s\n", WavName); 
            pmsis_exit(8);
        } 
        __CLOSE(File);

        float * spectrogram_fp32 = (float *)STFT_Spectrogram;
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f ",spectrogram_fp32[i]);
            STFT_Spectrogram[i] = (f16) spectrogram_fp32[i];
            PRINTF("(%f), ",STFT_Spectrogram[i]);
        }
#endif // fake data or data from file

#endif // load data STFT or AUDIO


#ifndef NN_INF_NOT
        /******
            NN Denoiser Task
        ******/
        PRINTF("\n\n****** Denoiser ***** \n");

//        PRINTF("\n\nConstructor\n");
//        int err_construct = __PREFIX(CNN_Construct)();
//        if (err_construct)
//        {
//            PRINTF("Graph constructor exited with error: %d\n", err_construct);
//            pmsis_exit(-5);
//        }
//        printf("The memory base is: %x\n",denoiser_L1_Memory);


        PRINTF("Send task to cluster\n");
   	    pi_cluster_send_task_to_cl(&cluster_dev, task_net);

        PRINTF("\n Denoiser Output\n");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f, ",STFT_Magnitude[i]);
        }

        #ifdef PERF
        {
            unsigned int TotalCycles = 0, TotalOper = 0;
            PRINTF("\n");
            for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
                PRINTF("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i],
                       AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
                TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
            }
            PRINTF("\n");
            PRINTF("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
            PRINTF("\n");
        }
        #endif  /* PERF */

//        __PREFIX(CNN_Destruct)();

        // Deassert Reset LSTM
        ResetLSTM = 0;
#endif  // disable nn inference


    /******
        ISTF Task
    ******/
    PRINTF("\n\n****** Computing iSTFT ***** \n");
    // compute mfcc if not read from file
    task_stft->entry = &RuniSTFT;

    L1_Memory = pmsis_l1_malloc(_L1_Memory_SIZE);
    if (L1_Memory==NULL){
        printf("Error allocating L1\n");
        pmsis_exit(-1);
    }

    pi_cluster_send_task_to_cl(&cluster_dev, task_stft);

    pmsis_l1_malloc_free(L1_Memory,_L1_Memory_SIZE);

    // check spectrogram results
    PRINTF("\nAudio Out: ");
    for (int i= 0 ; i<FRAME_SIZE; i++){
//        printf("%f", ((float)inSig[i])/(1<<15) );
        PRINTF("%f, ", Audio_Frame[i] );
    }
    PRINTF("\n");


#if IS_FAKE_SIGNAL_IN == 0
#if IS_INPUT_STFT == 1

   }   // stop looping over frames
#endif
#endif


#ifndef NN_INF_NOT
    __PREFIX(CNN_Destruct)();
#endif

    // Close the cluster
    pi_cluster_close(&cluster_dev);
    PRINTF("Ended\n");
    pmsis_exit(0);
}


int main()
{
	PRINTF("\n\n\t *** Denoiser ***\n\n");

    #define __XSTR(__s) __STR(__s)
    #define __STR(__s) #__s
//#if IS_FAKE_SIGNAL_IN == 0
//    WavName = __XSTR(AT_WAV);
//#endif    
    return pmsis_kickoff((void *) denoiser);
}
