/*
 * Copyright (C) 2021 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

// GAP Libraries and BSP
#include "Gap.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"

// Autotiler NN functions
#include "RFFTKernels.h"
#ifdef GRU
    #include "denoiser_GRU.h"
#else
    #include "denoiser.h"
#endif

// FS and Audio utils
#include "wavIO.h" 
#include "fs_switch.h"


// macros for F16 sqrt
#ifdef F16_DSP_BFLOAT
    #define SqrtF16(a) __builtin_pulp_f16altsqrt(a)
#else
    #define SqrtF16(a) __builtin_pulp_f16sqrt(a)
#endif

// global struct
struct pi_device HyperRam; 
AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

// board-dependent defines
#ifdef GAPUINO
    struct pi_device gpio;
    #define GPIO_OUT PI_GPIO_A1_PAD_13_B2 
    #define NMAX_ITER 5
    int iter = 0;
#endif

// datatype for computation
#if DTYPE == 0
    #define DATATYPE_SIGNAL     float16
    #define DATATYPE_SIGNAL_INF float16
#elif DTYPE == 1
    #define DATATYPE_SIGNAL     float16alt
    #define DATATYPE_SIGNAL_INF float16alt
#elif DTYPE == 2
    #define DATATYPE_SIGNAL     float16
    #define DATATYPE_SIGNAL_INF char
#else
    #define DATATYPE_SIGNAL short
#endif

/*
    Configuration: 
        IS_INPUT_STFT := 0
            IS_FAKE_SIGNAL_IN := 0
                >> load input input wav file and compute STFT
            IS_FAKE_SIGNAL_IN := 1
                >> input wav is a synthetic audio vector
        IS_INPUT_STFT := 1
            IS_FAKE_SIGNAL_IN := 0
                >> skip STFT computation and load STFT matrix
            IS_FAKE_SIGNAL_IN := 1
                >> skip STFT computation and use synthetic STFT matrix
        APPLY_DENOISER
*/
//#define CHECKSUM

#if IS_INPUT_STFT == 0 

    //load the input audio signal and compute the STFT
    #ifdef __gap9__
        #include "WinLUT_f16.def"
    #else

    #endif

    // defines for audio IOs

    // allocate space to load the input signal
    #define AUDIO_BUFFER_SIZE (MAX_L2_BUFFER) // as big as the L2 autotiler
    char *WavName = NULL;

    // L3 arrays to store input and output audio 
    static uint32_t inSig;
    static uint32_t outSig;

    #ifdef CHECKSUM
        #include "golden_sample_0000.h"
        float error;
        PI_L2 float STFT_Mag_Golden[] = GOLDEN_STFT_MAG;
        PI_L2 float Denoiser_Golden[] = GOLDEN_DENOISER;
        float snr = 0.0f;
        float p_err = 0.0f;
        float p_sig = 0.0f;
    #endif


#else
    // here the allocation in case of stft inputs

    // allocate space to load the input signal
    char *WavName = NULL;
    #ifdef CHECKSUM
        #include "golden_sample_0000.h"
        float error;
        PI_L2 float Denoiser_Golden[] = GOLDEN_DENOISER;
        float snr = 0.0f;
        float p_err = 0.0f;
        float p_sig = 0.0f;
    #endif
#endif


/* 
    static allocation of temporary buffers
*/
PI_L2 DATATYPE_SIGNAL Audio_Frame[FRAME_NFFT];  // stores the clip to compute the STFT. only first FRAME_SIZE samples (<FRAME_NFFT) are valid
PI_L2 DATATYPE_SIGNAL STFT_Spectrogram[AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2]; // the 2 is because of complex numbers
PI_L2 DATATYPE_SIGNAL STFT_Magnitude[AT_INPUT_WIDTH*AT_INPUT_HEIGHT];     // magnitude of the precedent vectors, used as denoiser input and output

PI_L2 short int Audio_Frame_temp[FRAME_SIZE];
PI_L2 int ResetLSTM;

// RNN states statically allocated to preserve the values during time
#define RNN_STATE_DIM_0 257 // FIXME: should be replaced with model-dependent defines
#define RNN_STATE_DIM_1 257
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_0_I[RNN_STATE_DIM_0];
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_1_I[RNN_STATE_DIM_1];
#ifndef GRU
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_0_C[RNN_STATE_DIM_0];
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_1_C[RNN_STATE_DIM_1];
#endif

#if IS_INPUT_STFT == 0 ///load the input audio signal and compute the MFCC
    /*
        STFT computation
            argument parameters are manually set based on STFT configuration
    */
    static void RunSTFT()
    {
    #ifdef PERF
        gap_cl_starttimer();
        gap_cl_resethwtimer();
    #endif
        unsigned int ta = gap_cl_readhwtimer();

        // compute the STFT 
        //      input: Audio Frame (FRAME_SIZE): 16 bits from the microphone or file
        //      output: STFT_Spectrogram, DATATYPE_SIGNAL as output (e.g. float16)
        STFT(
            Audio_Frame, 
            STFT_Spectrogram, 
            TwiddlesLUT_f16,
            RFFTTwiddlesLUT_f16,
            SwapTable_f16,
            WindowLUT_f16
        );

        unsigned int ti = gap_cl_readhwtimer() - ta;
        PRINTF("%45s: Cycles: %10d\n","STFT: ", ti );

        ta = gap_cl_readhwtimer();
        // compute the magnitude of the STFT components
        for (int i=0; i<AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++){
            DATATYPE_SIGNAL STFT_Real_Part = STFT_Spectrogram[2*i];
            DATATYPE_SIGNAL STFT_Imag_Part = STFT_Spectrogram[2*i+1];
            DATATYPE_SIGNAL STFT_Squared = STFT_Real_Part*STFT_Real_Part + STFT_Imag_Part*STFT_Imag_Part ;
            STFT_Magnitude[i] = SqrtF16 (STFT_Squared);
        }
        ti = gap_cl_readhwtimer() - ta;

        PRINTF("%45s: Cycles: %10d\n","Magnitude Compute: ", ti );
    }

    /*
        iSTFT computation
            argument parameters are manually set based on STFT configuration
    */
    #include "istft_window.h"   // includes the iSTFT windowing parameters

    static void RuniSTFT()
    {
    #ifdef PERF
        gap_cl_starttimer();
        gap_cl_resethwtimer();
    #endif
        unsigned int ta, ti;


        // compute the iSTFT 
        //      input: STFT_Spectrogram: DATATYPE_SIGNAL
        //      output: STFT_Spectrogram, DATATYPE_SIGNAL - reusing the same buffer
        ta = gap_cl_readhwtimer();
        iSTFT(
            STFT_Spectrogram, 
            STFT_Spectrogram, 
            TwiddlesLUT_f16,   
            RFFTTwiddlesLUT_f16,   
            SwapTable_f16
        );
        ti = gap_cl_readhwtimer() - ta;
        PRINTF("%45s: Cycles: %10d\n","iSTFT: ", ti );

        // Inverse Hanning Windowing
        ta = gap_cl_readhwtimer();
        for(int i=0;i<FRAME_SIZE;i++){
            Audio_Frame[i] = hanning_inv[i] * STFT_Spectrogram[i];
        }
        ti = gap_cl_readhwtimer() - ta;
        PRINTF("%45s: Cycles: %10d\n","iHanning: ", ti );
    }

#endif  // end TF transformation 


/*
    Denoiser Task
*/
static void RunDenoiser()
{

  PRINTF("Running on cluster\n");

#ifdef PERF
  unsigned int ta, ti;
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif

#ifdef GAPUINO
  pi_gpio_pin_write(&gpio, GPIO_OUT, 1 );
#endif

// casting from preprocessing datatype to NN datatype
DATATYPE_SIGNAL_INF * net_in_out = (DATATYPE_SIGNAL_INF * ) STFT_Magnitude;
#if DTYPE == 2
  // scale and quantize
  PRINTF("\nQuantized Inputs: ");
  int temp ; 
  for(int i = 0 ; i<257; i++){
    temp  = (DATATYPE_SIGNAL_INF) ( (DATATYPE_SIGNAL) STFT_Magnitude[i] /  SCALE_IN);
    if (temp > 127) net_in_out[i] = 127;
    else if (temp < -128) net_in_out[i] = -128;
    else net_in_out[i] = temp;
    PRINTF("%d, ", temp);
  }
  PRINTF("\n");
#endif

    /* Denoiser NN computation
          input: STFT_Magnitude: DATATYPE_SIGNAL, 
          output: STFT_Magnitude, DATATYPE_SIGNAL - reusing the same buffer
          states: RNN_STATE_0_I, RNN_STATE_0_C, RNN_STATE_1_I, RNN_STATE_1_C, must be preserved
          reset: only enabled at the start of the application
    */
  __PREFIX(CNN)(
        STFT_Magnitude,  
        RNN_STATE_0_I,
#ifndef GRU
        RNN_STATE_0_C,
#endif
        RNN_STATE_1_I,
#ifndef GRU
        RNN_STATE_1_C,
#endif
        ResetLSTM, 
        ResetLSTM, 
        STFT_Magnitude
    );

  // casting from inference datatype to post-processing datatype
#if DTYPE == 2
  // scale and dequantize the output
  PRINTF("Denoiser Output INT8:\n");
  for(int i = 0 ; i<257; i++){
    PRINTF("%d, ", (char) net_in_out[i]);
    STFT_Magnitude [257-i] = (DATATYPE_SIGNAL) net_in_out[257-i] * (DATATYPE_SIGNAL) SCALE_OUT ;
  }
  PRINTF("\n");

//    #define DATATYPE_SIGNAL     float16
//    #define DATATYPE_SIGNAL_INF char
//  
//  //scale the state
//  for(int i = 0 ; i<257; i++){
//    temp  = (DATATYPE_SIGNAL_INF) ( (DATATYPE_SIGNAL) LSTM_STATE_0_I[i] /  SCALE_IN);
//    if (temp > 127) net_in_out[i] = 127;
//    else if (temp < -128) net_in_out[i] = -128;
//    else net_in_out[i] = temp;
//    PRINTF("%d, ", temp);
//  }

#endif
//#if DTYPE == 1
//  printf("Going to cast the output from bf16 to f16\n");
//  for(int i = 0 ; i<257; i++){
//    STFT_Magnitude[i] = (float16) temp_bfp16[i];
//    printf("%f, ", STFT_Magnitude[i]);
//  }
//#endif


// apply denoising here
#ifdef APPLY_DENOISER
    // if denoiser is enabled, filter the STFT spectrogram with the mask in STFT_Magnitude
    ta = gap_cl_readhwtimer();
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * STFT_Magnitude[i];
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * STFT_Magnitude[i];
        }
    ti = gap_cl_readhwtimer() - ta;

    // debug print
    PRINTF("\nSTFT Filtered: ");
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2; i++ ){
        PRINTF("%f, ", STFT_Spectrogram[i]);
    }
    PRINTF("\n");

    PRINTF("%45s: Cycles: %10d\n","iScaling: ", ti );

#endif // apply scaling





#ifdef GAPUINO
  pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
#endif
}



static switch_file_t File = (switch_file_t) 0;
static switch_fs_t fs;

void denoiser(void)
{
    printf("Entering main controller\n");


    // Voltage-Frequency settings
    uint32_t voltage =1200;
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 300*1000*1000);

    //PMU_set_voltage(voltage, 0);
    printf("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
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
        Configure And Open the Hyperram. 
    ****/
    struct pi_hyperram_conf hyper_conf;
    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);
    if (pi_ram_open(&HyperRam))
    {
        printf("Error ram open !\n");
        pmsis_exit(-3);
    }

    /****
        Configure And open cluster. 
    ****/

    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);

    /******
        Setup STFT/ISTF task
    ******/

    struct pi_cluster_task* task_stft;
    task_stft = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    pi_cluster_task(task_stft,&RunSTFT,NULL);
    if (task_stft == NULL) {
        PRINTF("failed to allocate memory for task\n");
    }
    pi_cluster_task_stacks(task_stft, NULL, SLAVE_STACK_SIZE);

#ifndef DISABLE_NN_INFERENCE
    /******
        Setup Denoiser NN inference task (if enabled)
    ******/
    printf("Setup Cluster Task for inference!\n");
    struct pi_cluster_task* task_net;
    task_net = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    pi_cluster_task(task_net,&RunDenoiser,NULL);
    if(task_net==NULL) {
      PRINTF("pi_cluster_task alloc Error!\n");
      pmsis_exit(-1);
    }
    pi_cluster_task_stacks(task_net, NULL, SLAVE_STACK_SIZE);
    PRINTF("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
    
    // Reset LSTM
    ResetLSTM = 1;
    for(int i=0; i<RNN_STATE_DIM_0; i++){
        RNN_STATE_0_I[i] = (DATATYPE_SIGNAL_INF) 0.0f;
    }
    for(int i=0; i<RNN_STATE_DIM_1; i++){
        RNN_STATE_1_I[i] = (DATATYPE_SIGNAL_INF) 0.0f;
    }


#if IS_INPUT_STFT == 0 
    /****
        Read Audio Data from file using __PREFIX(_L2_Memory) as temporary buffer
        Data are prepared in L3 external memory
    ****/

    // allocate L2 Memory
    __PREFIX(_L2_Memory) = pi_l2_malloc(denoiser_L2_SIZE);
    if (__PREFIX(_L2_Memory) == 0) {
        printf("Error when allocating L2 buffer\n");
        pmsis_exit(18);        
    }

    // Allocate L3 buffers for audio IN/OUT
    if (pi_ram_alloc(&HyperRam, &inSig, (uint32_t) AUDIO_BUFFER_SIZE))
    {
        printf("inSig Ram malloc failed !\n");
        pmsis_exit(-4);
    }
    if (pi_ram_alloc(&HyperRam, &outSig, (uint32_t) AUDIO_BUFFER_SIZE))
    {
        printf("outSig Ram malloc failed !\n");
        pmsis_exit(-5);
    }

    // Read audio from file
    PRINTF("Reading wav from: %s \n", WavName);
    header_struct header_info;
      if (ReadWavFromFile(WavName,
            __PREFIX(_L2_Memory), AUDIO_BUFFER_SIZE*sizeof(short), &header_info)){
        PRINTF("\nError reading wav file\n");
        pmsis_exit(1);
    }
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    PRINTF("Num Samples: %d\n", num_samples);
    PRINTF("BitsPerSample: %d\n", header_info.BitsPerSample);
    printf("Finished Read wav.\n");

    // copy input data to L3
    pi_ram_write(&HyperRam, inSig,   __PREFIX(_L2_Memory), num_samples * sizeof(short));

    // Reset Output Buffer and copy to L3
    short * out_temp_buffer = (short *) __PREFIX(_L2_Memory);
    for(int i=0; i < num_samples; i++){
        out_temp_buffer[i] = 0;
    }
    pi_ram_write(&HyperRam, outSig,   __PREFIX(_L2_Memory), num_samples * sizeof(short));

    // free the temporary input memory
    pi_l2_free(__PREFIX(_L2_Memory),denoiser_L2_SIZE);

#endif

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
    PRINTF("Denoiser Contrcuctor OK! The L1 memory base is: %x\n",__PREFIX(_L1_Memory));

#endif // DISABLE_NN_INFERENCE


#if IS_INPUT_STFT == 0 

/****
    Load the input audio signal and compute the MFCC
    Audio_Frame: includes only a single frame for audio
****/

    int tot_frames = (int) (((float)num_samples / FRAME_STEP) - NUM_FRAME_OVERLAP) ;
//    tot_frames = 10; // debug purpose then remove
    printf("Number of frames to be processed: %d\n", tot_frames);

    for (int frame_id = 0; frame_id < tot_frames; frame_id++)
    {   
        printf("***** Processing Frame %d of %d ***** \n", frame_id+1, tot_frames);
        // Copy Data from L3 to L2
        short * in_temp_buffer = (short *) Audio_Frame;
        pi_ram_read(
            &HyperRam, 
            inSig + frame_id * FRAME_STEP * sizeof(short), 
            in_temp_buffer, 
            (uint32_t) FRAME_SIZE*sizeof(short)
        );

        // cast data from Q16.15 to DATATYPE_SIGNAL (may be float16)
        PRINTF("Audio In: ");
        for (int i= 0 ; i<FRAME_SIZE; i++){
            Audio_Frame[i] = ((DATATYPE_SIGNAL) in_temp_buffer[i] )/(1<<15);
            PRINTF("%f, ", Audio_Frame[i] );
        }

        /******
            MFCC Task
        ******/
        // compute mfcc if not read from file
        PRINTF("\n\n****** Computing STFT ***** \n");
        //task_stft->entry = &RunSTFT;

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
            PRINTF("%f, ",STFT_Spectrogram[i]);
        }
        PRINTF("\n");

        PRINTF("\nMagnitude OUT: ");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f, ",STFT_Magnitude[i]);
        }
        PRINTF("\n");


#ifdef CHECKSUM
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            float err = STFT_Magnitude[i] - STFT_Mag_Golden[i]; 
            p_err += err * err;
            p_sig += STFT_Magnitude[i] * STFT_Magnitude[i];
        }
        snr = p_sig / p_err;
        printf("STFT Signal-to-noise ratio in linear scale: %f\n", snr);
        if (snr > 10000.0f)     // qsnr > 40db
            printf("--> STFT OK!\n");
        else
            printf("--> STFT NOK!\n");
#endif

    

#else ///load the STFT



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
            STFT_Spectrogram[i] = (f16) spectrogram_fp32[i];    // FIXME: this may be removed
            PRINTF("(%f), ",STFT_Spectrogram[i]);
            STFT_Magnitude[i] = (f16) spectrogram_fp32[i];
        }

#endif // load data STFT or AUDIO


#ifndef DISABLE_NN_INFERENCE
        /******
            NN Denoiser Task
                Model already constructed and never destructed
        ******/
        PRINTF("\n\n****** Denoiser ***** \n");

        PRINTF("Send task to cluster\n");
   	    pi_cluster_send_task_to_cl(&cluster_dev, task_net);

        PRINTF("\n Denoiser Output\n");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f, ",STFT_Magnitude[i]);
        }


        #ifdef PERF
        {
            unsigned int TotalCycles = 0, TotalOper = 0;
            printf("\n");
            for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
                printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", 
                    AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], 
                    ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
                TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
            }
            printf("\n");
            printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
            printf("\n");
        }
        #endif  /* PERF */


        // Deassert Reset LSTM
        ResetLSTM = 0;
#endif  // disable nn inference


#if IS_INPUT_STFT == 0 // if not loading the STFT
    #ifdef CHECKSUM
        p_err = 0.0f; p_sig=0.0f;
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            float err = STFT_Magnitude[i] - Denoiser_Golden[i]; 
            p_err += err * err;
            p_sig += STFT_Magnitude[i] * STFT_Magnitude[i];
        }
        snr = p_sig / p_err;
        printf("Denoiser Signal-to-noise ratio in linear scale: %f\n", snr);
        if (snr > 1000.0f)     // qsnr > 30db
            printf("--> Denoiser OK!\n");
        else
            printf("--> Denoiser NOK!\n");
    #endif

    /******
        ISTF Task
    ******/
    PRINTF("\n\n****** Computing iSTFT ***** \n");

    //task_stft->entry = &RuniSTFT;
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
        PRINTF("%f, ", Audio_Frame[i] );
    }
    PRINTF("\n");
#endif


#if IS_FAKE_SIGNAL_IN == 0
#if IS_INPUT_STFT == 0

        // if denoising auio files, outputs are loaded to the L3 output buffer outSig
        PRINTF("Writing Frame %d/%d to the output buffer\n\n", frame_id+1, tot_frames);

        // Cast if needed
        pi_ram_read(&HyperRam,  (short *) outSig + (frame_id*FRAME_STEP), 
            Audio_Frame_temp, FRAME_SIZE * sizeof(short));

        // from DATA_S
        for (int i= 0 ; i<FRAME_SIZE; i++){
            Audio_Frame_temp[i] += (short int)(Audio_Frame[i] * (1<<15));
        }
        pi_ram_write(&HyperRam,  (short *) outSig + (frame_id*FRAME_STEP),   
            Audio_Frame_temp, FRAME_SIZE * sizeof(short));
#endif

   }   // stop looping over frames
    #ifdef CHECKSUM
        p_err = 0.0f; p_sig=0.0f;
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            float err = STFT_Magnitude[i] - Denoiser_Golden[i]; 
            p_err += err * err;
            p_sig += STFT_Magnitude[i] * STFT_Magnitude[i];
        }
        snr = p_sig / p_err;
        printf("Denoiser Signal-to-noise ratio in linear scale: %f\n", snr);
        if (snr > 1000.0f)     // qsnr > 30db
            printf("--> Denoiser OK!\n");
        else
            printf("--> Denoiser NOK!\n");
    #endif
#endif


#ifndef DISABLE_NN_INFERENCE
    __PREFIX(CNN_Destruct)();
#endif



// write reults to file: test_gap.wav
#if IS_INPUT_STFT == 0

    // allocate L2 Memory
    __PREFIX(_L2_Memory) = pi_l2_malloc(denoiser_L2_SIZE);
    if (__PREFIX(_L2_Memory) == 0) {
        printf("Error when allocating L2 buffer\n");
        pmsis_exit(18);        
    }


    // copy input data to L3
    pi_ram_read(&HyperRam, outSig,   __PREFIX(_L2_Memory), num_samples * sizeof(short));
    

    // final sample 
    out_temp_buffer = (short int * ) __PREFIX(_L2_Memory);
    PRINTF("\nAudio Out: ");
    for (int i= 0 ; i<num_samples; i++){
        PRINTF("%f, ", ((float) out_temp_buffer[i] )/(1<<15)  );
    }
    PRINTF("\n");

    WriteWavToFile("test_gap.wav", 16, 16000, 1, 
        (uint32_t *) __PREFIX(_L2_Memory), num_samples* sizeof(short));
    printf("Writing wav file to test_gap.wav completed successfully\n");

    pi_l2_free(__PREFIX(_L2_Memory),denoiser_L2_SIZE);
#endif


    // Close the cluster
    pi_cluster_close(&cluster_dev);
    PRINTF("Ended\n");
    pmsis_exit(0);
}


int main()
{
	PRINTF("\n\n\t *** Denoiser ***\n\n");

#if IS_INPUT_STFT == 0 
#if IS_FAKE_SIGNAL_IN == 0
    #define __XSTR(__s) __STR(__s)
    #define __STR(__s) #__s
    WavName = __XSTR(WAV_FILE);
#endif    
#endif

    return pmsis_kickoff((void *) denoiser);
}
