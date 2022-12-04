/*
 * Copyright (C) 2022 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

/* 
    include files
*/
#include "Gap.h"
#include "bsp/ram.h"
#include <bsp/fs/hostfs.h>
#include "gaplib/wavIO.h" 

// Autotiler NN functions
#include "RFFTKernels.h"
#include "WinLUT_f16.def"   //load the input audio signal and compute the STFT


// NN Model Header
#if DEMO == 1 
    #include "denoiser_dns.h"   // demo configuration
#else
    #ifdef GRU
        #include "denoiser_GRU.h"
    #else
        #include "denoiser.h"
    #endif
#endif

/* 
     global variables
*/
struct pi_device DefaultRam; 
struct pi_device* ram = &DefaultRam;

AT_DEFAULTFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

#ifdef AUDIO_EVK
    // GPIO defines
    pi_gpio_e gpio_pin_o; /* PI_GPIO_A02-PI_GPIO_A05 */
    int val_gpio;
#endif

//static struct pi_default_flash_conf flash_conf;
static pi_fs_file_t * file[1];
static struct pi_device fs;
static struct pi_device flash;
pi_device_t* i2c_slider;
static PI_L2 uint16_t slider_value;

// datatype for computation
#define DATATYPE_SIGNAL     float16
#define DATATYPE_SIGNAL_INF float16
#define SqrtF16(a) __builtin_pulp_f16sqrt(a)


#if IS_INPUT_STFT == 0 

    // defines for audio IOs

    // allocate space to load the input signal
    #define AUDIO_BUFFER_SIZE (MAX_L2_BUFFER>>1) // as big as the L2 autotiler
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

#if IS_SFU == 1 
PI_L2 DATATYPE_SIGNAL Audio_Frame_temp[FRAME_SIZE];
#else 
PI_L2 short int Audio_Frame_temp[FRAME_SIZE];
#endif //IS_INPUT_STFT == 0 && IS_SFU == 0

PI_L2 int ResetLSTM;

// RNN states statically allocated to preserve the values during time
// note that, for simplicity we left the rnn states to be 16 bits variables even if quantized to 8 bits
#define RNN_STATE_DIM_0 (H_STATE_LEN) 
#define RNN_STATE_DIM_1 (H_STATE_LEN)
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_0_I[RNN_STATE_DIM_0];
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_1_I[RNN_STATE_DIM_1];
#ifndef GRU
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_0_C[RNN_STATE_DIM_0];
PI_L2 DATATYPE_SIGNAL_INF RNN_STATE_1_C[RNN_STATE_DIM_1];
#endif


static uint16_t ads1014_read(pi_device_t *dev, uint8_t addr)
{
    uint16_t result;
    pi_i2c_write(dev, &addr, 1, PI_I2C_XFER_START | PI_I2C_XFER_STOP);
    pi_i2c_read(dev, (uint8_t *)&result, 2, PI_I2C_XFER_START | PI_I2C_XFER_STOP);
    result = (result << 8) | (result >> 8);
    return result;
}

static int ads1014_write(pi_device_t *dev, uint8_t addr, uint16_t value)
{
    uint8_t buffer[3] = { addr, value >> 8, value & 0xFF };
    return pi_i2c_write(dev, buffer, 3, PI_I2C_XFER_START | PI_I2C_XFER_STOP);
}

int init_ads1014(pi_device_t *i2c)
{
    struct pi_i2c_conf conf;
    pi_i2c_conf_init(&conf);
    conf.itf = 1;
    pi_i2c_conf_set_slave_addr(&conf, 0x90, 0);

    pi_open_from_conf(i2c, &conf);
    if (pi_i2c_open(i2c)) return -1;

    uint16_t expected = (1 << 15) | (0 << 12) | (2 << 9) | (7 << 5) | 3;
    ads1014_write(i2c, 1, expected);

    return 0;
}


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
        TwiddlesLUT,
        RFFTTwiddlesLUT,
        SwapTable,
        WindowLUT
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

static void RuniSTFT()
{
#   ifdef PERF
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#   endif
    unsigned int ta, ti;


    // compute the iSTFT 
    //      input: STFT_Spectrogram: DATATYPE_SIGNAL
    //      output: STFT_Spectrogram, DATATYPE_SIGNAL - reusing the same buffer
    ta = gap_cl_readhwtimer();
    iSTFT(
        STFT_Spectrogram, 
        STFT_Spectrogram, 
        TwiddlesLUT,   
        RFFTTwiddlesLUT,   
        SwapTable
    );
    ti = gap_cl_readhwtimer() - ta;
    PRINTF("%45s: Cycles: %10d\n","iSTFT: ", ti );
}

/*
    Denoiser Task
*/
static void RunDenoiser()
{

    PRINTF("Running on cluster\n");

#   ifdef PERF
    unsigned int ta, ti;
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#   endif

    /* Denoiser NN computation
          input: STFT_Magnitude: DATATYPE_SIGNAL, 
          output: STFT_Magnitude, DATATYPE_SIGNAL - reusing the same buffer
          states: RNN_STATE_0_I, RNN_STATE_0_C, RNN_STATE_1_I, RNN_STATE_1_C, must be preserved
          reset: only enabled at the start of the application
    */
#ifdef AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 1);
#endif
    __PREFIX(CNN)(
#   ifndef GRU
        RNN_STATE_1_C,
        RNN_STATE_0_C,
#   endif
        RNN_STATE_1_I,
        RNN_STATE_0_I,        
        STFT_Magnitude,  
        ResetLSTM, 
        ResetLSTM, 
        STFT_Magnitude
    );
#ifdef AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 0);
#endif

    /* 
        apply denoising here! 
        filter the STFT_Spectrogram with the mask in STFT_Magnitude
        if STFT_Magnitude[i] == 1.0 the filtering does not apply
    */
    #   ifdef PERF
    ta = gap_cl_readhwtimer();
    #endif
    for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
        //#ifdef AUDIO_EVK
        
        if(slider_value>28000){
        //#endif
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * STFT_Magnitude[i];
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * STFT_Magnitude[i];
        //#ifdef AUDIO_EVK
        }else{
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * 1.0f;
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * 1.0f;    
        }
        //#endif
    }    
    #   ifdef PERF
    ti = gap_cl_readhwtimer() - ta;
    PRINTF("%45s: Cycles: %10d\n","Denoising applicatio: ", ti );
    #endif
}


#if IS_SFU == 1

    #include "GraphINOUT_L2_Descr.h"
    #include "SFU_RT.h"

    // FIXME: to tune it!!
    #define Q_BIT_IN 27
    #define Q_BIT_OUT (Q_BIT_IN-3)

    #define BUFF_SIZE (FRAME_STEP*4)
    #define CHUNK_NUM (8)

    //This should be equal to FRAME_SIZE/FRAME_STEP + 1
    #define STRUCT_DELAY (1)

    #define SAI_ITF_IN         (1)
    #define SAI_ITF_OUT        (2)

    #define SAI_ID               (48)
    #define SAI_SCK(itf)         (48+(itf*4)+0)
    #define SAI_WS(itf)          (48+(itf*4)+1)
    #define SAI_SDI(itf)         (48+(itf*4)+2)
    #define SAI_SDO(itf)         (48+(itf*4)+3)

    SFU_uDMA_Channel_T *ChanOutCtxt_0;
    SFU_uDMA_Channel_T *ChanOutCtxt_1;
    SFU_uDMA_Channel_T *ChanInCtxt_0;
    SFU_uDMA_Channel_T *ChanInCtxt_1;

    void ** BufferInList;
    void ** BufferOutList;

    volatile int remaining_size;
    volatile int sent_size;
    volatile int done;
    int nb_transfers;
    int current_size[2];
    static pi_event_t proc_task;


    static int open_i2s_PDM(struct pi_device *i2s, unsigned int SAIn, unsigned int Frequency, unsigned int Direction, unsigned int Diff)
    {
        struct pi_i2s_conf i2s_conf;
        pi_i2s_conf_init(&i2s_conf);

        // polarity: b0: SDI: slave/master, b1:SDO: slave/master    1:RX, 0:TX
        i2s_conf.options = PI_I2S_OPT_REF_CLK_FAST;
        i2s_conf.frame_clk_freq = Frequency;                // In pdm mode, the frame_clk_freq = i2s_clk
        i2s_conf.itf = SAIn;                                // Which sai interface
        i2s_conf.mode = PI_I2S_MODE_PDM;                    // Choose PDM mode
        i2s_conf.pdm_direction = Direction;                 // 2b'11 slave on both SDI and SDO (SDO under test)
        i2s_conf.pdm_diff = Diff;                           // Set differential mode on pairs (TX only)

    //    i2s_conf.options |= PI_I2S_OPT_EXT_CLK;             // Put I2S CLK in input mode for safety

        pi_open_from_conf(i2s, &i2s_conf);

        if (pi_i2s_open(i2s))
            return -1;

        pi_pad_set_function(SAI_SCK(SAIn),PI_PAD_FUNC0);
        pi_pad_set_function(SAI_SDI(SAIn),PI_PAD_FUNC0);
        pi_pad_set_function(SAI_SDO(SAIn),PI_PAD_FUNC0);
        pi_pad_set_function(SAI_WS(SAIn),PI_PAD_FUNC0);

        return 0;
    }

    static int chunk_in_cnt;



    static void handle_sfu_in_0_end(void *arg)
    {
        
        if(chunk_in_cnt==STRUCT_DELAY){
            //pi_time_wait_us(5000);
            SFU_Enqueue_uDMA_Channel_Multi(ChanOutCtxt_0, CHUNK_NUM, BufferOutList, BUFF_SIZE, 0);
            SFU_GraphResetInputs(&SFU_RTD(GraphINOUT));
        }

            pi_evt_push(&proc_task);
    }

#endif // IS_SFU == 1 





int denoiser(void)
{
    printf("Entering main controller\n");

        /****
        Change Frequency if needed
    ****/
 
    // Voltage-Frequency settings
    uint32_t voltage =VOLTAGE;
    pi_freq_set(PI_FREQ_DOMAIN_FC,      FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH,  FREQ_FC*1000*1000);

#ifdef AUDIO_EVK
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
#endif 

    //PMU_set_voltage(voltage, 0);
    printf("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
        (float)voltage/1000, FREQ_FC, FREQ_CL);

#ifdef AUDIO_EVK
    /****
        Configure GPIO Output.
    ****/
    gpio_pin_o = PI_GPIO_A89; /* PI_GPIO_A02-PI_GPIO_A05 */

    pi_pad_set_function(PI_PAD_089, PI_PAD_FUNC1);

    pi_gpio_pin_configure( gpio_pin_o, PI_GPIO_OUTPUT);
#endif


    /****
        Configure And Open the External Ram. 
    ****/
    struct pi_default_ram_conf ram_conf;
    pi_default_ram_conf_init(&ram_conf);
    ram_conf.baudrate = FREQ_FC*1000*1000;
    pi_open_from_conf(&DefaultRam, &ram_conf);
    if (pi_ram_open(&DefaultRam))
    {
        printf("Error ram open !\n");
        pmsis_exit(-3);
    }
    printf("RAM Opened\n");

    /****
        Configure And open cluster. 
    ****/

    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;
        cl_conf.id = 0;                /* Set cluster ID. */
                       // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |   
                       // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                       PI_CLUSTER_ICACHE_PREFETCH_ENABLE |      
                       // Enable the icache for all the cores
                       PI_CLUSTER_ICACHE_ENABLE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    printf("Cluster Opened\n");
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);



#if IS_SFU == 1 
    /****
        Setup the SFU for PDM in/out
    ****/
    struct pi_device i2s_in;
    struct pi_device i2s_out;
    int Status;
    int Trace = 0;
    pi_evt_sig_init(&proc_task);

    // Drive pad with 12 mAP to have less noise
    uint32_t *Magic_Setting_0 = (uint32_t *)0x1A104064;
    *Magic_Setting_0 = 3 << 2 | 3 << 10 | 3 << 18 | 3 << 26;

    // SAI 2 -> Drive pad with 12 mAP to have less noise
    uint32_t *Magic_Setting = (uint32_t *)0x1A104068;
    *Magic_Setting = 3 << 10 | 3 << 18;
    
    // Configure PDM in
    if (open_i2s_PDM(&i2s_in, SAI_ITF_IN,   3072000, 3, 0)) return -1;
    // Configure PDM out
    if (open_i2s_PDM(&i2s_out, SAI_ITF_OUT, 3072000, 0, 0)) return -1;

    StartSFU(FREQ_SFU*1000*1000, 1);

    ChanInCtxt_0   = (SFU_uDMA_Channel_T *) pi_l2_malloc(sizeof(SFU_uDMA_Channel_T));
    ChanOutCtxt_0  = (SFU_uDMA_Channel_T *) pi_l2_malloc(sizeof(SFU_uDMA_Channel_T));
    
    
    BufferInList = (void*) pi_l2_malloc(sizeof(void*)*CHUNK_NUM);
    for(int i=0;i<CHUNK_NUM;i++) BufferInList[i]=pi_l2_malloc(BUFF_SIZE);
    
    BufferOutList = (void*)pi_l2_malloc(sizeof(void*)*CHUNK_NUM);
    for(int i=0;i<CHUNK_NUM;i++) BufferOutList[i]=pi_l2_malloc(BUFF_SIZE);;


    // Get uDMA channels for GraphIN
    SFU_Allocate_uDMA_Channel(ChanInCtxt_0, 0, &SFU_RTD(GraphINOUT));
    SFU_uDMA_Channel_Callback(ChanInCtxt_0, handle_sfu_in_0_end, ChanInCtxt_0);
    
    // Get uDMA channels for GraphOUT
    SFU_Allocate_uDMA_Channel(ChanOutCtxt_0, 0, &SFU_RTD(GraphINOUT));
    //SFU_uDMA_Channel_Callback(ChanOutCtxt_0, handle_sfu_out_0_end, ChanOutCtxt_0);
    
    // Connect Channels to SFU for Mic IN (PDM IN)
    SFU_GraphConnectIO(SFU_Name(GraphINOUT, In_1), SAI_ITF_IN, 2, &SFU_RTD(GraphINOUT));
    SFU_GraphConnectIO(SFU_Name(GraphINOUT, Out_1), ChanInCtxt_0->ChannelId, 0, &SFU_RTD(GraphINOUT));
    

    // Connect Channels to SFU for PDM OUT
    Status =  SFU_GraphConnectIO(SFU_Name(GraphINOUT, In1), ChanOutCtxt_0->ChannelId, 0, &SFU_RTD(GraphINOUT));
    Status =  SFU_GraphConnectIO(SFU_Name(GraphINOUT, Out1), SAI_ITF_OUT, 1, &SFU_RTD(GraphINOUT));

    //Next API will have a value to replace this high number with -1
    //To be able to 
    SFU_Enqueue_uDMA_Channel_Multi(ChanInCtxt_0, CHUNK_NUM, BufferInList, BUFF_SIZE, 0);

            //Starting In and Out Graphs
    pi_i2s_ioctl(&i2s_in, PI_I2S_IOCTL_START, NULL);
    pi_i2s_ioctl(&i2s_out, PI_I2S_IOCTL_START, NULL);

    

    fxl6408_setup();

    // Setup 2 DAC
    if(setup_dac(0) || setup_dac(1))
    {
        printf("Failed to setup DAC\n");
        pmsis_exit(-1);
    }
    pi_time_wait_us(100000);
    //printf("Setup DAC OK\n"); 

    //Enable slicer
    i2c_slider = pi_l2_malloc(sizeof(pi_device_t));
    init_ads1014(i2c_slider);

#else //IS_SFU == 0 

    /****
        Load Audio Wav from file 
        if not testing STFT input
    ****/
    
#if IS_INPUT_STFT == 0 

    // Read Audio Data from file using __PREFIX(_L2_Memory) as temporary buffer
    // Data are prepared in L3 external memory
 
    __PREFIX(_L2_Memory) = pi_l2_malloc(denoiser_L2_SIZE);
    if (__PREFIX(_L2_Memory) == 0) {
        printf("Error when allocating L2 buffer\n");
        pmsis_exit(18);        
    }
    
    // Allocate L3 buffers for audio IN/OUT
    if (pi_ram_alloc(&DefaultRam, &inSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("inSig Ram malloc failed !\n");
        pmsis_exit(-4);
    }
    if (pi_ram_alloc(&DefaultRam, &outSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("outSig Ram malloc failed !\n");
        pmsis_exit(-5);
    }

    // Read audio from file
    printf("Reading wav from: %s \n", WavName);
    header_struct header_info;
      if (ReadWavFromFile(WavName,
            __PREFIX(_L2_Memory), AUDIO_BUFFER_SIZE*sizeof(short), &header_info)){
        printf("\nError reading wav file\n");
        pmsis_exit(1);
    }
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    printf("Num Samples: %d with BitsPerSample: %d\n", num_samples, header_info.BitsPerSample);
    printf("Finished Read wav.\n");


    if(num_samples*sizeof(short) > denoiser_L2_SIZE){
        printf("The size of the audio exceeds the available L2 memory space!\n");
        pmsis_exit(1);
    }

    // copy input data to L3
    pi_ram_write(&DefaultRam, inSig, __PREFIX(_L2_Memory), num_samples * sizeof(short));

    // Reset Output Buffer and copy to L3
    short * out_temp_buffer = (short *) __PREFIX(_L2_Memory);
    for(int i=0; i < num_samples; i++){
        out_temp_buffer[i] = 0;
    }
    pi_ram_write(&DefaultRam, outSig,   __PREFIX(_L2_Memory), num_samples * sizeof(short));

    // free the temporary input memory
    pi_l2_free( __PREFIX(_L2_Memory), denoiser_L2_SIZE);


#endif //IS_INPUT_STFT == 0 
#endif //IS_SFU


    /******
        Setup STFT/ISTF task
    ******/
    printf("Setup STFT task!\n");
    struct pi_cluster_task* task_stft;
    task_stft = pi_l2_malloc(sizeof(struct pi_cluster_task));
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
    task_net = pi_l2_malloc(sizeof(struct pi_cluster_task));
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
    IMP: Audio_Frame includes only a single frame for audio
****/

#if IS_SFU == 0     
    
    // audio from file

    int tot_frames = (int) (((float)num_samples / FRAME_STEP) - NUM_FRAME_OVERLAP) ;
    printf("Number of frames to be processed: %d\n", tot_frames);

    for (int frame_id=0; frame_id < tot_frames; frame_id++)
    {   
        printf("***** Processing Frame %d of %d ***** \n", frame_id+1, tot_frames);
        // Copy Data from L3 to L2
        short * in_temp_buffer = (short *) Audio_Frame;
        pi_ram_read(
            &DefaultRam, 
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
#else   

    // audio from SFU

    chunk_in_cnt=0;
    SFU_StartGraph(&SFU_RTD(GraphINOUT));
    while(1){
        slider_value = ads1014_read(i2c_slider, 0);
        pi_evt_wait_on(&proc_task);

#ifdef AUDIO_EVK
        pi_gpio_pin_write(gpio_pin_o, 1);
#endif

        int round = (chunk_in_cnt%CHUNK_NUM);
        int round_out = (chunk_in_cnt>(STRUCT_DELAY-1))? ((chunk_in_cnt-(STRUCT_DELAY-1))%CHUNK_NUM):0;

        //First Copy previous loop processed frame to output
        for(int i=0;i<BUFF_SIZE/4;i++) {
            ((int32_t*)BufferOutList[round_out])[i]= (int32_t)((float)(Audio_Frame_temp[i])*((int)(1<<Q_BIT_OUT)));
        }


        for(int i=0;i<FRAME_SIZE-FRAME_STEP;i++){
            Audio_Frame[i] = Audio_Frame[i+FRAME_STEP];
            Audio_Frame_temp[i] = Audio_Frame_temp[i+FRAME_STEP];
        }

        for(int i=0;i<FRAME_STEP;i++){
            Audio_Frame[i+FRAME_SIZE-FRAME_STEP] = (DATATYPE_SIGNAL)(((float)((int32_t*)BufferInList[round])[i]) /((int)(1<<Q_BIT_IN)));
            Audio_Frame_temp[i+FRAME_SIZE-FRAME_STEP] = (DATATYPE_SIGNAL) 0.0f;
        }

#endif //IS_SFU == 0     


        /******
            Compute the MFCC
        ******/

        PRINTF("\n\n****** Computing STFT ***** \n");
        pi_cluster_task(task_stft,&RunSTFT,NULL);

        L1_Memory = pi_l1_malloc(&cluster_dev, _L1_Memory_SIZE);
        if (L1_Memory==NULL){
            printf("Error allocating L1\n");
            pmsis_exit(-1);
        }

        pi_cluster_send_task_to_cl(&cluster_dev, task_stft);
        pi_l1_free(&cluster_dev, L1_Memory,_L1_Memory_SIZE);

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
    

#   else // IS_INPUT_STFT == 0  //load the STFT 


    // open FS and read the binary files with STFT (flaot values)
    struct pi_hostfs_conf conf;
    pi_hostfs_conf_init(&conf);
    conf.fs.flash = &flash;
    pi_open_from_conf(&fs, &conf);
    if (pi_fs_mount(&fs))
        return -2;

    for(int frame_id = 0; frame_id<STFT_FRAMES; frame_id++){

        PRINTF("Reading STFT file %.4d/%d...\n", frame_id, STFT_FRAMES );
        sprintf(WavName, "%s/samples/mags_%.4d.bin",BUILD_DIR,frame_id);
        printf("File being read is : %s\n", WavName);

        file[0] = pi_fs_open(&fs, WavName, 0);

        if (file[0] == 0) {
            printf("Failed to open file, %s\n", WavName); 
            pmsis_exit(7);
        }
        printf("File %x of size %d\n", file[0], sizeof(pi_fs_file_t));

        int TotBytes = sizeof(float)*AT_INPUT_WIDTH*AT_INPUT_HEIGHT;
        int len = pi_fs_read(file[0], STFT_Spectrogram, TotBytes);
        printf("Bytes read %d - of %d bytes expected\n", len,TotBytes );
        if (len != TotBytes){
            printf("Too few bytes in %s\n", WavName); 
            pmsis_exit(8);
        } 
        //__CLOSE(File);
        pi_fs_close(file[0]);

        float * spectrogram_fp32 = (float *)STFT_Spectrogram;
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f ",spectrogram_fp32[i]);
            STFT_Magnitude[i] = (DATATYPE_SIGNAL) spectrogram_fp32[i] ;
            PRINTF(" - %f), ",STFT_Magnitude[i]);

        }

#   endif // load data STFT or AUDIO


#   ifndef DISABLE_NN_INFERENCE
        /******
            NN Denoiser Task
                Model already constructed and never destructed
        ******/
        PRINTF("\n\n****** Denoiser ***** \n");

        // Debug PRINT
        PRINTF("\n Denoiser Input\n");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f, ",STFT_Magnitude[i]);
        }

        PRINTF("Send task to cluster\n");
   	    pi_cluster_send_task_to_cl(&cluster_dev, task_net);

        // Debug PRINT
        PRINTF("\n Denoiser Output\n");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
            PRINTF("%f, ",STFT_Magnitude[i]);
        }
        PRINTF("\nSTFT Filtered: ");
        for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT*2; i++ ){
            PRINTF("%f, ", STFT_Spectrogram[i]);
        }

    #ifdef PERF
        {
            unsigned int TotalCycles = 0, TotalOper = 0;
            PRINTF("\n");
            for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
                PRINTF("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", 
                    AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], 
                    ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
                TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
            }
            PRINTF("\n");
            PRINTF("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
            PRINTF("\n");
        }
    #endif  /* PERF */

        // Deassert Reset LSTM
        ResetLSTM = 0;

#ifdef CHECKSUM
        if(frame_id == STFT_FRAMES-1){ // last frame
            p_err = 0.0f; p_sig=0.0f;
            for (int i = 0; i< AT_INPUT_WIDTH*AT_INPUT_HEIGHT; i++ ){
                float err = ((float) STFT_Magnitude[i]) - Denoiser_Golden[i]; 
                p_err += err * err;
                p_sig += STFT_Magnitude[i] * STFT_Magnitude[i];
                PRINTF("[%d] %f vs %f -> %f\n", i, STFT_Magnitude[i], Denoiser_Golden[i], (err * err)/(STFT_Magnitude[i] * STFT_Magnitude[i]));
            }
            snr = p_sig / p_err;
            printf("Denoiser Signal-to-noise ratio in linear scale: %f\n", snr);
            if (snr > 90.0f)     // qsnr >~ 20db
                printf("--> Denoiser OK!\n");
            else{
                printf("--> Denoiser NOK!\n");
                pmsis_exit(-1);
            }            
        }

#endif //CHECKSUM

#ifdef AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 0);
#endif

#endif  // DISABLE_NN_INFERENCE



#if IS_INPUT_STFT == 0 // if not loading the STFT

#ifdef AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 1);
#endif

        /******
            ISTF Task
        ******/
        PRINTF("\n\n****** Computing iSTFT ***** \n");
        pi_cluster_task(task_stft, &RuniSTFT, NULL);
        L1_Memory = pi_l1_malloc(&cluster_dev, _L1_Memory_SIZE);
        if (L1_Memory==NULL){
            printf("Error allocating L1\n");
            pmsis_exit(-1);
        }

        pi_cluster_send_task_to_cl(&cluster_dev, task_stft);

    	pi_l1_free(&cluster_dev, L1_Memory,_L1_Memory_SIZE);

        
        // debug printf
        PRINTF("\nAudio Out: ");
        for (int i= 0 ; i<FRAME_SIZE; i++){
            PRINTF("%f,", Audio_Frame[i] );
        }


        //copy spectrogram into Audio Frames 
        for (int i= 0 ; i<FRAME_SIZE; i++){
#if IS_SFU == 1
            // overlap and add using temporary buffer
            // use Audio_Frame to store an output frame
            //Audio_Frame_temp[i] = DRY * (STFT_Spectrogram[i] / 2 ) + (1-DRY)* Audio_Frame[i];   // FIXME: divide by 2 because of current Hanning windowing
            Audio_Frame_temp[i] += (STFT_Spectrogram[i] / 2 );   // FIXME: divide by 2 because of current Hanning windowing
#else
            // use Audio_Frame to store an output frame
            Audio_Frame[i] = (STFT_Spectrogram[i] / 2 );   // FIXME: divide by 2 because of current Hanning windowing
#endif
        }
        PRINTF("\n");



#if IS_SFU == 1

        // block until next input audio frame is ready
#ifdef AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 0);
#endif
        chunk_in_cnt++;
        pi_evt_sig_init(&proc_task);


#else // audio from file 

        // if denoising auio files, outputs are loaded to the L3 output buffer outSig
        PRINTF("Writing Frame %d/%d to the output buffer\n\n", frame_id+1, tot_frames);

        // Cast if needed
        pi_ram_read(&DefaultRam,  (short *) outSig + (frame_id*FRAME_STEP), 
            Audio_Frame_temp, FRAME_SIZE * sizeof(short));

        // from DATA_S
        for (int i= 0 ; i<FRAME_SIZE; i++){
            Audio_Frame_temp[i] += (short int)(Audio_Frame[i] * (1<<15));
        }
        pi_ram_write(&DefaultRam,  (short *) outSig + (frame_id*FRAME_STEP),   
            Audio_Frame_temp, FRAME_SIZE * sizeof(short));
#endif //IS_SFU == 1

#endif //IS_INPUT_STFT == 0

   }   // stop looping over frames


#ifndef DISABLE_NN_INFERENCE
    __PREFIX(CNN_Destruct)();
#endif



/*
    Exit the real-time mode (only for testing)
    and write clean speech audio to file: test_gap.wav
*/
#if IS_INPUT_STFT == 0 && IS_SFU == 0

    // allocate L2 Memory
    __PREFIX(_L2_Memory) = pi_l2_malloc(denoiser_L2_SIZE);
    if (__PREFIX(_L2_Memory) == 0) {
        printf("Error when allocating L2 buffer\n");
        pmsis_exit(18);        
    }


    // copy input data to L3
    out_temp_buffer = (short int * ) __PREFIX(_L2_Memory); 
    pi_ram_read(&DefaultRam, outSig,   out_temp_buffer, num_samples * sizeof(short));
    
#ifdef CHECKSUM

    short int * in_temp_buffer = ((short int * ) __PREFIX(_L2_Memory)) + num_samples;
    pi_ram_read(&DefaultRam, inSig,   in_temp_buffer, num_samples * sizeof(short));

    p_err = 0.0f; p_sig=0.0f;
    for (int i = 0; i< num_samples; i++ ){   // remove first and last elements from checksum
        float in_signal = ((float) in_temp_buffer[i] )/(1<<15);
        float out_signal = ((float) out_temp_buffer[i] )/(1<<15);
        PRINTF("%f vs %f\n", in_signal, out_signal);
        float err = in_signal - out_signal; 
        p_err += (err * err);
        p_sig += (in_signal * in_signal);

    }
    printf("Completed the checksum check over %d samples\n", num_samples);
    if (p_err == 0.0)
        snr = 1000000.0f;
    else
        snr = p_sig / p_err;
    printf("ISTFT Signal-to-noise ratio in linear scale: %f\n", snr);
    if (snr > 1000.0f)     // qsnr > 30db
        printf("--> STFT+iSTFT OK!\n");
    else{
        printf("--> STFT+iSTFT NOK!\n");
        pmsis_exit(-1);
    }


#else //CHECKSUM
    // final sample
    PRINTF("\nAudio Out: ");
    for (int i= 0 ; i<num_samples; i++){
        PRINTF("%f, ", ((float) out_temp_buffer[i] )/(1<<15)  );
    }
    PRINTF("\n");

    WriteWavToFile("%s/test_gap.wav", BUILD_DIR, 16, 16000, 1, 
        (uint32_t *) __PREFIX(_L2_Memory), num_samples* sizeof(short));
    printf("Writing wav file to test_gap.wav completed successfully\n");
#endif //CHECKSUM

    pi_l2_free(__PREFIX(_L2_Memory),denoiser_L2_SIZE);
#endif //IS_INPUT_STFT == 0 && IS_SFU == 0


    // Close the cluster
    pi_cluster_close(&cluster_dev);
    PRINTF("Ended\n");
    pmsis_exit(0);
    return 0;
}


int main()
{
	PRINTF("\n\n\t *** Denoiser ***\n\n");

#   if IS_SFU == 0 
    #define __XSTR(__s) __STR(__s)
    #define __STR(__s) #__s
    WavName = __XSTR(WAV_FILE);
#   endif    

    return pmsis_kickoff((void *) denoiser);
}
