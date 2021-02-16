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

// FIXME
#include "KWS_ds_cnn_l_quantKernels.h"
#include "MFCC_params_LARGE.h"



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
char *ImageIn;
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

#ifdef FROM_SENSOR
    static uint16_t buff[2][NB_ELEM];
    static struct pi_device i2s;
    static int end = 0;
    static pi_task_t task;
    static short *chunk;
    #define LENGTH_AV 32
    #define SHL 3
    static int av[4][LENGTH_AV] ;
    static int idx_av=0;
    static int av_00=0,av_01=0,av_02=0,av_03=0;

    static void copy_data(uint16_t *dst, uint16_t *src, uint size)
    {
      for (int i=0; i<size; i++) dst[i] = src[i];
    }

    static void shift_copy_data(uint16_t *dst, uint16_t *src, uint size)
    {
      for (int i=0; i<size; i++) dst[i] = src[i]<<SHL;
    }
    // dump one mono channel chunk (NB_ELEM/2 16bits samples) from the i2s0 interface in dump_buff:
    static void my_copy_data(uint16_t *dst, uint16_t *src, uint size)
    {
      int av_0=0,av_1=0,av_2=0,av_3=0;
      unsigned i, j=0;

      // copy 1 mono channel chunk into the dump buffer
      for ( i=0;i<size; i++){
        dst[i] = src[i]<<SHL;
        av_0 += dst[i];
      }

      // assume 256 samples in the buffer
      av_0 /= size;
      
      av_00 -= av[0][idx_av];
      av_00 += av_0;
      av[0][idx_av] = av_0;
      idx_av++;
      if (idx_av==LENGTH_AV) idx_av=0;

      // offset correction (length of buff of avg values is 16)
      for(i = 0; i < size; i++)
        {
          dst[i] -= (av_00>>5) ;
        }

    }

    // This callback is called when a transfer is finished
    // Just reenqueue another buffer unless we are done
    static void end_of_capture(void *arg)
    {
        unsigned int size;

        pi_i2s_read_status(&task, (void **)&chunk, &size);
        copy_data((uint16_t *)(inSig), (uint16_t *) inSig + NB_ELEM, NB_ELEM);
        //my_copy_data((uint16_t *)(inSig + NB_ELEM), (uint16_t *) chunk, NB_ELEM);
        shift_copy_data((uint16_t *)(inSig + NB_ELEM), (uint16_t *) chunk, NB_ELEM);
        idx++;

        if (idx < 2){
          pi_i2s_read_async(&i2s, pi_task_callback(&task, end_of_capture, NULL));
        }
        else {
          end1 = 1;
        }

    }
#endif

static void RunMFCC(){
    L1_Memory = __PREFIX(_L1_Memory);
    PRINTF("Runnning MFCC\n");
    #ifdef PERF
        gap_cl_starttimer();
        gap_cl_resethwtimer();
        int start, elapsed, total_cyc;
        total_cyc = 0;
        start = gap_cl_readhwtimer();
    #endif
    // run inference on inSig[0:WAV_BUFFER_SIZE] and inSig[WAV_BUFFER_SIZE/2:WAV_BUFER_SIZE*3/2] alternately
    pi_gpio_pin_write(&gpio, GPIO_OUT, 1 );
    MFCC(inSig, mfcc_features, 0, TwiddlesLUT, SwapLUT, WindowLUT, MFCC_FilterBank, MFCC_Coeffs, 5, DCT_Coeff);
    pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
    #ifdef PERF
        elapsed = gap_cl_readhwtimer() - start;
        total_cyc += elapsed;
        printf("MFCC Total Cycles: %d\n\n\n", total_cyc);
    #endif
}

static void Runkws()
{
  PRINTF("Running on cluster\n");
#ifdef PERF
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
  pi_gpio_pin_write(&gpio, GPIO_OUT, 1 );
  __PREFIX(CNN)(ImageIn, (short int *) ResOut);
  pi_gpio_pin_write(&gpio, GPIO_OUT, 0);
  //Checki Results
  rec_digit = 0;
  int highest = ResOut[0];
  PRINTF("Results: \n");
  for(int i = 0; i < NUM_CLASSES; i++) {
    if(ResOut[i] > highest) {
      highest = ResOut[i];
      rec_digit = i;
    }
    PRINTF("class %d: %d\n", i, ResOut[i]);
  }
  if (highest<20000 && rec_digit!=0) rec_digit = 1;
  if (prev>0 && rec_digit!=prev) rec_digit = 1;
  prev = rec_digit;

  if(rec_digit>1)
    printf("Recognized: %s\twith confidence: %d\n", LABELS[rec_digit], highest);

#ifdef PERF
    if (rec_digit!=8){
        printf("App didn't recognize ON with %s test sample\n", WavName);
        pmsis_exit(-1);
    }
#endif
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
    ResOut        = (unsigned short int *) pi_l2_malloc(NUM_CLASSES             * sizeof(short int));
    ImageIn       = (char *)      pi_l2_malloc(AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char));
    mfcc_features = (short int *) pi_l2_malloc(N_FRAME * N_DCT                  * sizeof(short int));
    inSig         = (short int *) pi_l2_malloc(NB_ELEM * (ITER + ITER/2)        * sizeof(short int));
    if (mfcc_features==NULL || ImageIn==NULL || inSig==NULL || ResOut==NULL){
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

    #ifdef FROM_SENSOR
        // Get default I2S interface config
        struct pi_i2s_conf i2s_conf;
        pi_i2s_conf_init(&i2s_conf);

        // Configure first interface for PDM 44100KHz DDR
        // Also gives the 2 buffers for double-buffering the sampling
        i2s_conf.options = PI_I2S_OPT_EXT_CLK | PI_I2S_OPT_EXT_WS;
        i2s_conf.pingpong_buffers[0] = buff[0];
        i2s_conf.pingpong_buffers[1] = buff[1];
        i2s_conf.block_size = NB_ELEM*sizeof(short);
        i2s_conf.frame_clk_freq = 16000;
        i2s_conf.itf = 0;
        i2s_conf.channels = 1;
        i2s_conf.format = PI_I2S_FMT_DATA_FORMAT_PDM;
        i2s_conf.word_size = 16;
        i2s_conf.pdm_decimation=128;

        pi_open_from_conf(&i2s, &i2s_conf);

        // Open the driver
        if (pi_i2s_open(&i2s))
          pmsis_exit(1);  

        // Start sampling, the driver will use the double-buffers we provided to store
        // the incoming samples
        if (pi_i2s_ioctl(&i2s, PI_I2S_IOCTL_START, NULL))
          pmsis_exit(1);
        pi_time_wait_us(1000);
    #endif

count=0;
idx = 0;
end1 = end2 = 0;
printf("Waiting for command... [yes, no, up, down, left, right, on, off, stop, go]\n");
while(1)
{
    #ifndef FROM_SENSOR
        header_struct header_info;
        printf("Reading the wav file\n");
        //if (ReadWavFromFile(WavName, inSig, WAV_BUFFER_SIZE*sizeof(short int), &header_info)){
        //    printf("Error reading wav file\n");
        //    pmsis_exit(1);
        //}
        num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    #else
        unsigned int size;

        // Once it returns, chunk will point to the next available buffer
        // containing samples.
        pi_i2s_read_async(&i2s, pi_task_callback(&task, end_of_capture, NULL));
        // Wait until acquisition is finished
        while(idx<2 || (end1==0 && end2==0))
          {
            pi_yield();
          }
        count++;
        if (end1) end1 = 0;
        if (end2) end2 = 0;

        #ifdef WRITE_WAV
            char FileName[100];
            sprintf(FileName, "../../../from_gap_%d_%s.wav", count, LABELS[rec_digit]);
            // run inference on inSig[0:WAV_BUFFER_SIZE] and inSig[WAV_BUFFER_SIZE/2:WAV_BUFER_SIZE*3/2] alternately
            WriteWavToFile(FileName, i2s_conf.word_size, i2s_conf.frame_clk_freq, i2s_conf.channels, 
                           (void *)inSig, WAV_BUFFER_SIZE * sizeof(short int));
        #endif
    #endif

    struct pi_cluster_task task_mfcc = {0};
    task_mfcc.entry = RunMFCC;
    task_mfcc.arg = NULL;
    task_mfcc.stack_size = (unsigned int) STACK_SIZE;
    task_mfcc.slave_stack_size = SLAVE_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &task_mfcc);

    for (int i=0; i<N_FRAME; i++) {
        // Take only the first N_CEP that you need (in this case 10)
        for (int j=0; j<AT_INPUT_WIDTH; j++) {
            ImageIn[i*AT_INPUT_WIDTH+j] = (char) gap_roundnorm(mfcc_features[i*N_DCT+j]*INPUT_SCALE, INPUT_SCALEN);
        }
    }

    PRINTF("Call cluster\n");
	struct pi_cluster_task *task_net = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if(task_net==NULL) {
	  printf("pi_cluster_task alloc Error!\n");
	  pmsis_exit(-1);
	}
	//PRINTF("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
	memset(task_net, 0, sizeof(struct pi_cluster_task));
	task_net->entry = &Runkws;
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

}
    #ifdef FROM_SENSOR
        // Now stop sampling
        pi_i2s_ioctl(&i2s, PI_I2S_IOCTL_STOP, NULL);
        // Close the driver
        pi_i2s_close(&i2s);
    #endif
    __PREFIX(CNN_Destruct)();

    pi_l2_free(ImageIn, AT_INPUT_WIDTH * AT_INPUT_HEIGHT * sizeof(char));
    pi_l2_free(ResOut,  NUM_CLASSES*sizeof(short int));
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
