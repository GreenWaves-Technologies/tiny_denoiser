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
#include "denoiserKernels.h"

#include "RFFTKernels.h"
#include "TwiddlesDef.h"
#include "RFFTTwiddlesDef.h"
#include "SwapTablesDef.h"
#include "WinLUT.def"
#include "WinLUT_f32.def"
#ifdef __gap9__
#include "WinLUT_f16.def"
#endif





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

#define DATATYPE_SIGNAL f16

// computation buffers
PI_L2 DATATYPE_SIGNAL AudioIn[FRAME_SIZE];
PI_L2 DATATYPE_SIGNAL STFT_Spectrogram_in[AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2];
PI_L2 DATATYPE_SIGNAL STFT_Spectrogram_out[AT_INPUT_WIDTH*AT_INPUT_HEIGHT];
PI_L2 DATATYPE_SIGNAL AudioOut[FRAME_SIZE];

static void RunMel()
{
#ifdef PERF
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#endif
    unsigned int ta = gap_cl_readhwtimer();

//    STFT(
//        AudioIn, 
//        STFT_Spectrogram_in, 
//        R2_Twiddles_fix_256,   
//        RFFT_Twiddles_fix_512,   
//        R2_SwapTable_fix_256, 
//        WinLUT
//    );
//    STFT(AudioIn, STFT_Spectrogram_in, R2_Twiddles_fix_256,   RFFT_Twiddles_fix_512,   R2_SwapTable_fix_256, WinLUT, PreempShift);
//    STFT(AudioIn, STFT_Spectrogram_in, R2_Twiddles_float_256, RFFT_Twiddles_float_512, R2_SwapTable_fix_256, WinLUT_f32);
    STFT(
        AudioIn, 
        STFT_Spectrogram_in, 
        R4_Twiddles_f16_256,   
        RFFT_Twiddles_f16_512,   
        R4_SwapTable_fix_256, 
        WinLUT_f16
    );

    
    unsigned int ti = gap_cl_readhwtimer() - ta;

    PRINTF("%45s: Cycles: %10d\n","LOG MEL: ", ti );

}

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
  __PREFIX(CNN)(STFT_Spectrogram_in,  0, STFT_Spectrogram_out);
#ifdef GAPUINO
  pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
#endif
}





void denoiser(void)
{
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


    PRINTF("Entering main controller\n");
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }

#if IS_FAKE_SIGNAL_IN == 1

    // load fake data into AudioIn: a single frame of lenght FRAME_SIZE
    for (int i=0;i<FRAME_SIZE;i++){
        AudioIn[i] = 0;
    }
#else
    printf("Reading wav...\n");
    header_struct header_info;
    if (ReadWavFromFile("../../../samples/sample_0000.wav", inSig, AUDIO_BUFFER_SIZE*sizeof(float), &header_info)){
        printf("Error reading wav file\n");
        pmsis_exit(1);
    }
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    printf("Num Samples: %d\n",num_samples);
    printf("BitsPerSample: %d\n",header_info.BitsPerSample);

    printf("Finished Read wav.\n");

    // Cast if needed
    for (int i= 0 ; i<FRAME_SIZE; i++){
//        printf("%f", ((float)inSig[i])/(1<<15) );
        AudioIn[i] = ((DATATYPE_SIGNAL) inSig[i] )/(1<<15);
//        printf("\t%f\n", AudioIn[i] );
    }
#endif

    /******
        MFCC Task
    ******/
    PRINTF("\n\n****** STFT ***** \n");
    struct pi_cluster_task *task_stft = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    if (!task_stft) {
        PRINTF("failed to allocate memory for task\n");
    }

    memset(task_stft, 0, sizeof(struct pi_cluster_task));
    task_stft->entry = &RunMel;
    task_stft->stack_size = STACK_SIZE;
    task_stft->slave_stack_size = SLAVE_STACK_SIZE;
    task_stft->arg = NULL;

    L1_Memory = pmsis_l1_malloc(_L1_Memory_SIZE);
    if (L1_Memory==NULL){
        printf("Error allocating L1\n");
        pmsis_exit(-1);
    }

    pi_cluster_send_task_to_cl(&cluster_dev, task_stft);
    pmsis_l1_malloc_free(L1_Memory,_L1_Memory_SIZE);

    // check spectrogram results
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2; i++ ){
        printf("%f, ",STFT_Spectrogram_in[i]);
        
    }

    /******
        NN Denoiser Task
    ******/
    PRINTF("\n\n****** Denoiser ***** \n");

    PRINTF("\n\nConstructor\n");
    int err_construct = __PREFIX(CNN_Construct)();
    if (err_construct)
    {
        PRINTF("Graph constructor exited with error: %d\n", err_construct);
        pmsis_exit(-5);
    }

    PRINTF("Call cluster\n");
	struct pi_cluster_task *task_net = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if(task_net==NULL) {
	  PRINTF("pi_cluster_task alloc Error!\n");
	  pmsis_exit(-1);
	}
	//PRINTF("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
	memset(task_net, 0, sizeof(struct pi_cluster_task));
	task_net->entry = &RunDenoiser;
	task_net->stack_size = STACK_SIZE;
	task_net->slave_stack_size = SLAVE_STACK_SIZE;
	task_net->arg = NULL;
	pi_cluster_send_task_to_cl(&cluster_dev, task_net);

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

    __PREFIX(CNN_Destruct)();

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
#if IS_FAKE_SIGNAL_IN == 0
    WavName = __XSTR(AT_WAV);
#endif    
    return pmsis_kickoff((void *) denoiser);
}
