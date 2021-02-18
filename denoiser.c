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

#include "wavIO.h"
#include "denoiserKernels.h"



#define  WAV_BUFFER_SIZE    16000 // 1sec@16kHz
#define  NUM_CLASSES        12

//DCT_NORMALIZATION        -> np.sqrt(1/(N_DCT))*0.5
//NNTOOL_INPUT_SCALE_FLOAT -> 1.9372712
// SCALE = NNTOOL_INPUT_SCALE_FLOAT*DCT_NORMALIZATION
#define  INPUT_SCALE        236
#define  INPUT_SCALEN       16

#define NB_ELEM 8000
#define BUFF_SIZE (NB_ELEM*2)
#define ITER    2

char *WavName = NULL;
char *AudioIn;
char *AudioOut;

 
int off_shift = 0;
AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

int num_samples;
short int *inSig;
short int *outSig;
int count, idx, end1, end2;
int rec_digit;
int prev = -1;

struct pi_device gpio;
#define GPIO_OUT PI_GPIO_A1_PAD_13_B2 
#define NMAX_ITER 5
int iter = 0;





static void RunDenoiser()
{
//        L1_Memory = __PREFIX(_L1_Memory);

  PRINTF("Running on cluster\n");
#ifdef PERF
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
  pi_gpio_pin_write(&gpio, GPIO_OUT, 1 );
  __PREFIX(CNN)(AudioIn,  AudioOut);
  pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
  //Checki Results
//  rec_digit = 0;
//  int highest = ResOut[0];
//  PRINTF("Results: \n");
//  for(int i = 0; i < NUM_CLASSES; i++) {
//    if(ResOut[i] > highest) {
//      highest = ResOut[i];
//      rec_digit = i;
//    }
//    PRINTF("class %d: %d\n", i, ResOut[i]);
//  }
//  if (highest<20000 && rec_digit!=0) rec_digit = 1;
//  if (prev>0 && rec_digit!=prev) rec_digit = 1;
//  prev = rec_digit;
//
//  if(rec_digit>1)
//    printf("Recognized: %s\twith confidence: %d\n", LABELS[rec_digit], highest);
//
//#ifdef PERF
//    if (rec_digit!=8){
//        printf("App didn't recognize ON with %s test sample\n", WavName);
//        pmsis_exit(-1);
//    }
//#endif
}


void denoiser(void)
{
    // Voltage-Frequency settings
    uint32_t voltage =1200;
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    //PMU_set_voltage(voltage, 0);
    printf("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
        (float)voltage/1000, FREQ_FC, FREQ_CL);
    pulp_write32(0x1A10414C,1);

	//configuring gpio
	struct pi_gpio_conf gpio_conf = {0};
    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio, &gpio_conf);
    int errors = pi_gpio_open(&gpio);
    if (errors)
    {
        printf("Error opening GPIO %d\n", errors);
        pmsis_exit(errors);
    }
    /* Configure gpio input. */
    pi_gpio_pin_configure(&gpio, GPIO_OUT, PI_GPIO_OUTPUT);
    pi_pad_set_function(PI_PAD_13_B2_RF_PACTRL1, PI_PAD_13_B2_GPIO_A1_FUNC1  );

    pi_gpio_pin_write(&gpio, GPIO_OUT, 0);

    printf("Entering main controller\n");
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    
    // this are the input output
    AudioIn       = (char *)      pi_l2_malloc(AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char));
    AudioOut      = (char *)      pi_l2_malloc(AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char));
    if (AudioIn==NULL || AudioOut==NULL ){
        printf("Error allocating output\n");
        pmsis_exit(1);
    }


    printf("\n\nConstructor\n");
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    int err_construct = __PREFIX(CNN_Construct)();
    if (err_construct)
    {
        printf("Graph constructor exited with error: %d\n", err_construct);
        pmsis_exit(-5);
    }



    // read data from file
    header_struct header_info;
    printf("Reading the wav file\n");
    if (ReadWavFromFile(WavName, AudioIn, AT_INPUT_WIDTH*sizeof(short int), &header_info)){
        printf("Error reading wav file\n");
        pmsis_exit(1);
    }
    num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    printf("Number of samples: %d\n", num_samples);

    PRINTF("Call cluster\n");
	struct pi_cluster_task *task_net = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if(task_net==NULL) {
	  printf("pi_cluster_task alloc Error!\n");
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
        printf("\n");
        for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
            printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i],
                   AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
            TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
        }
        printf("\n");
        printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
        printf("\n");
        break;
    }
    #endif  /* PERF */


    __PREFIX(CNN_Destruct)();

    pi_l2_free(AudioIn, AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char) );
    pi_l2_free(ResOut,  AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char) );
    // Close the cluster
    pi_cluster_close(&cluster_dev);
    PRINTF("Ended\n");
    pmsis_exit(0);
}


int main()
{
	PRINTF("\n\n\t *** KWS ***\n\n");

    #define __XSTR(__s) __STR(__s)
    #define __STR(__s) #__s
    WavName = __XSTR(AT_WAV);
    return pmsis_kickoff((void *) denoiser);
}
