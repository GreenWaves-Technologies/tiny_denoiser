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
#if AK4332 == 1
    #include "dac.h"
#else
    #include "ssm6515.h"
#endif

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

#if AUDIO_EVK
    // GPIO defines
    pi_gpio_e gpio_pin_o; /* PI_GPIO_A02-PI_GPIO_A05 */
    int val_gpio;
#endif

//static struct pi_default_flash_conf flash_conf;
static pi_fs_file_t * file[1];
static struct pi_device fs;
static struct pi_device flash;
pi_device_t* i2c_slider;
static PI_L2 uint16_t slider_value = 29000;

// datatype for computation
#define DATATYPE_SIGNAL     float16
#if FIX_STATES == 1
    #define DATATYPE_SIGNAL_INF signed char
    #define ZERO 0
#else
    #define DATATYPE_SIGNAL_INF float16
    #define ZERO 0.0f
#endif
#ifndef SqrtF16
#define SqrtF16(a) __builtin_pulp_f16sqrt(a)
#endif


#if IS_INPUT_STFT == 0 

    // defines for audio IOs

    // allocate space to load the input signal
    #define AUDIO_BUFFER_SIZE (denoiser_L2_SIZE>>1) // as big as the L2 autotiler
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
#if AUDIO_EVK
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
#if AUDIO_EVK
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
        #if IS_SFU == 1
        if(slider_value>28000){
        #endif
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * STFT_Magnitude[i];
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * STFT_Magnitude[i];
        #if IS_SFU == 1
        }else{
            STFT_Spectrogram[2*i]    = STFT_Spectrogram[2*i]   * 1.0f;
            STFT_Spectrogram[2*i+1]  = STFT_Spectrogram[2*i+1] * 1.0f;    
        }
        #endif
    }    
    #   ifdef PERF
    ti = gap_cl_readhwtimer() - ta;
    PRINTF("%45s: Cycles: %10d\n","Denoising applicatio: ", ti );
    #endif
}


#if IS_SFU == 1

    #include "GraphINOUT_L2_Descr.h"
    #include "SFU_RT.h"


    #include "sfu_pmsis_runtime.h"

    // FIXME: to tune it!!
    #define Q_BIT_IN 29
    #define Q_BIT_OUT (Q_BIT_IN-2)

    #define BUFF_SIZE (FRAME_STEP*4)
    #define CHUNK_NUM (2)

    //This should be equal to FRAME_SIZE/FRAME_STEP + 1
    #define STRUCT_DELAY (5)

    #define SAI_ITF_IN         (1)
    #define SAI_ITF_OUT        (2)

    #define SAI_ID               (48)
    #define SAI_SCK(itf)         (48+(itf*4)+0)
    #define SAI_WS(itf)          (48+(itf*4)+1)
    #define SAI_SDI(itf)         (48+(itf*4)+2)
    #define SAI_SDO(itf)         (48+(itf*4)+3)

    pi_sfu_graph_t *sfu_graph; // SFU sfu_graph instance.

    pi_sfu_mem_port_t *sfu_memout_port; // Memory port context
    pi_sfu_mem_port_t *sfu_memin_port;

    // Runtime port identifiers
    int sfu_out_port_ids;
    int sfu_in_port_ids;

    // List of buffer lists (a list for each output port)
    pi_sfu_buffer_t sfu_out_buffers[CHUNK_NUM];
    pi_sfu_buffer_t sfu_in_buffers[CHUNK_NUM];

    uint32_t sfu_out_buff_idx;
    uint32_t sfu_in_buff_idx;

    // OS task to register end of chunk transfer callback
    pi_evt_t sfu_mem_out_task;
    pi_evt_t sfu_mem_in_task;

    static pi_evt_t proc_task;

    static int chunk_in_cnt;


    /*******
     *  Functions for the PDM and SFU
     *******/

    static int32_t open_sfu_graph(uint32_t sfu_freq, pi_sfu_graph_desc_t *graph_desc)
    {
        int32_t err = 0;

        pi_sfu_conf_t sfu_conf = {.sfu_frequency = sfu_freq};
        if (PI_OK != pi_sfu_open(&sfu_conf))
        {
            err = -1;
            //printf("SFU open failed\n");
        }
        else
        {
            sfu_graph = pi_sfu_graph_open(graph_desc);
            if(sfu_graph == NULL)
            {
                err = -2;
                //printf("SFU open graph failed\n");
            }
        }

        return err;
    }

    static int open_i2s_PDM(struct pi_device *i2s, unsigned int SAIn, unsigned int Frequency, unsigned int Polarity, unsigned int Diff)
    {
        struct pi_i2s_conf i2s_conf;
        pi_i2s_conf_init(&i2s_conf);

        // polarity: b0: SDI: slave/master, b1:SDO: slave/master    1:RX, 0:TX
        i2s_conf.options = PI_I2S_OPT_REF_CLK_FAST;
        i2s_conf.frame_clk_freq = Frequency;                // In pdm mode, the frame_clk_freq = i2s_clk
        i2s_conf.itf = SAIn;                                // Which sai interface
        i2s_conf.mode = PI_I2S_MODE_PDM;                // Choose PDM mode
        i2s_conf.pdm_direction = Polarity;                   // 2b'11 slave on both SDI and SDO (SDO under test)
        i2s_conf.pdm_diff = Diff;                           // Set differential mode on pairs (TX only)

        pi_open_from_conf(i2s, &i2s_conf);

        if (pi_i2s_open(i2s))
            return -1;

        pi_pad_function_set(SAI_SCK(SAIn),PI_PAD_FUNC0);
        pi_pad_function_set(SAI_SDI(SAIn),PI_PAD_FUNC0);
        pi_pad_function_set(SAI_SDO(SAIn),PI_PAD_FUNC0);
        pi_pad_function_set(SAI_WS(SAIn),PI_PAD_FUNC0);

        return 0;
    }


    static inline uint32_t increment_buffer_idx(uint32_t idx)
    {
        if (idx + 1 < CHUNK_NUM)
            idx += 1;
        else
            idx = 0;
        return idx;
    }

    static void handle_mem_out_end(void * arg)
    {
        // /* Enqueue next buffer to receive from MEM_OUT */
        pi_sfu_enqueue(sfu_graph, sfu_memout_port, &sfu_out_buffers[sfu_out_buff_idx]);
        sfu_out_buff_idx = increment_buffer_idx(sfu_out_buff_idx);
        
        pi_evt_push(&proc_task);
    }

    static void handle_mem_in_end(void *arg)
    {
        // Enqueue next buffer
        pi_sfu_enqueue(sfu_graph, sfu_memin_port, &sfu_in_buffers[sfu_in_buff_idx]);
        sfu_in_buff_idx = increment_buffer_idx(sfu_in_buff_idx);
    }


    int setup_SFU_mem_in_mem_out_chain(int32_t mem_out_node_id,int32_t mem_in_node_id, uint32_t nb_buffers, uint32_t frame_step, size_t size)
    {
        /***********************************************************************************************
         * Configure Input / Output (MEM_OUT,MEM_IN uDMA/1D)
         **********************************************************************************************/

        sfu_out_port_ids = mem_out_node_id;
        sfu_in_port_ids = mem_in_node_id;

        // Allocate I/O buffers
            
            //MEM_OUT
            sfu_memout_port = pi_sfu_mem_port_get(sfu_graph, sfu_out_port_ids);
            if (sfu_memout_port == NULL)
            {
                printf("Failed to get MEM_OUT port references\n");
            }
            pi_evt_callback_irq_init(&sfu_mem_out_task, handle_mem_out_end, &sfu_out_port_ids);

            for (uint32_t i = 0; i < nb_buffers; i++)
            {
                void *data = pi_l2_malloc(frame_step*size);
                if (data == NULL) return -1;
                pi_sfu_buffer_init(&sfu_out_buffers[i], data, frame_step, size);

                sfu_out_buffers[i].task = &sfu_mem_out_task;
            pi_sfu_enqueue(sfu_graph, sfu_memout_port, &sfu_out_buffers[i]);
            }
            // pi_sfu_continuous_enqueue(sfu_memout_port, &sfu_out_buffers[0], &sfu_out_buffers[1]);



            //MEM_IN
            sfu_memin_port = pi_sfu_mem_port_get(sfu_graph, sfu_in_port_ids);
            if (sfu_memin_port == NULL)
            {
                printf("Failed to get MEM_IN port references\n");
            }
            pi_evt_callback_irq_init(&sfu_mem_in_task, handle_mem_in_end, &sfu_in_port_ids);

            for (uint32_t i = 0; i < nb_buffers  ; i++)
            {
                void *data = pi_l2_malloc(frame_step*size);
                if (data == NULL) return -1;
                pi_sfu_buffer_init(&sfu_in_buffers[i], data, frame_step, size);

                sfu_in_buffers[i].task = &sfu_mem_in_task;
            pi_sfu_enqueue(sfu_graph, sfu_memin_port, &sfu_in_buffers[i]);
            }

            // pi_sfu_continuous_enqueue(sfu_memin_port, &sfu_in_buffers[0], &sfu_in_buffers[1]);

        return 0;
    }


    int setup_SFU_pdm_in_pdm_out_chain(int32_t pdm_in_node_descriptor, uint32_t pdm_in_itf, uint32_t pdm_in_channel,
                                        int32_t pdm_out1_node_descriptor, uint32_t pdm_out1_itf, uint32_t pdm_out1_channel,
                                        int32_t pdm_out2_node_descriptor, uint32_t pdm_out2_itf, uint32_t pdm_out2_channel)
    {
        pi_sfu_pdm_itf_id_t in_pdm_itf_id = {pdm_in_itf, pdm_in_channel, 0};
        pi_sfu_graph_pdm_bind(sfu_graph, pdm_in_node_descriptor,  &in_pdm_itf_id);

        
        pi_sfu_pdm_itf_id_t out1_pdm_itf_id = {pdm_out1_itf, pdm_out1_channel, 1};
        pi_sfu_graph_pdm_bind(sfu_graph, pdm_out1_node_descriptor,  &out1_pdm_itf_id);
        
        pi_sfu_pdm_itf_id_t out2_pdm_itf_id = {pdm_out2_itf, pdm_out2_channel, 1}; 
        pi_sfu_graph_pdm_bind(sfu_graph, pdm_out2_node_descriptor,  &out2_pdm_itf_id);

        return 0;
    }

#endif // IS_SFU == 1 


int denoiser(void)
{
    PRINTF("Entering main controller\n");

        /****
        Change Frequency if needed
    ****/
 
    // Voltage-Frequency settings
    uint32_t voltage=VOLTAGE;
    pi_freq_set(PI_FREQ_DOMAIN_FC,      FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH,  FREQ_FC*1000*1000);

#if AUDIO_EVK
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
#endif 

    //PMU_set_voltage(voltage, 0);
    PRINTF("Set VDD voltage as %.2f, FC Frequency as %d MHz, CL Frequency = %d MHz\n", 
        (float)voltage/1000, FREQ_FC, FREQ_CL);

#if AUDIO_EVK
    /****
        Configure GPIO Output.
    ****/
    gpio_pin_o = PI_GPIO_A89; /* PI_GPIO_A02-PI_GPIO_A05 */

    pi_pad_function_set(PI_PAD_089, PI_PAD_FUNC1);

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
        return -3;
    }
    PRINTF("RAM Opened\n");

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
        return -4;
    }
    PRINTF("Cluster Opened\n");
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


    // Configure PDM in
    if (open_i2s_PDM(&i2s_in, SAI_ITF_IN,   3072000, 2, 0)) return -1;
    // Configure PDM out
    if (open_i2s_PDM(&i2s_out, SAI_ITF_OUT, 3072000, 0, 0)) return -1;
 
     int32_t err = open_sfu_graph(FREQ_SFU*1000*1000, &SFU_RTD(GraphINOUT));
    if (err)
    {
        printf("Failed to open sfu graph \n");
        return err;
    }

    /*******
     * Setup SFU chain
     ******/

    setup_SFU_mem_in_mem_out_chain(SFU_Name(GraphINOUT, Out_1),
                                   SFU_Name(GraphINOUT, In1),
                                   CHUNK_NUM, FRAME_STEP, sizeof(int32_t));
    #if AK4332 == 1
    setup_SFU_pdm_in_pdm_out_chain(SFU_Name(GraphINOUT, In_1), SAI_ITF_IN, 2,
                                   SFU_Name(GraphINOUT, Out1), SAI_ITF_OUT, 1,
                                   SFU_Name(GraphINOUT, Out2), SAI_ITF_IN, 1);
    #else
    setup_SFU_pdm_in_pdm_out_chain(SFU_Name(GraphINOUT, In_1), SAI_ITF_IN, 2,
                                   SFU_Name(GraphINOUT, Out1), SAI_ITF_OUT, 0,
                                   SFU_Name(GraphINOUT, Out2), SAI_ITF_IN, 0);
                                   
    #endif



    // Let the microphone start (20ms in datasheet) before capturing
    pi_i2s_ioctl(&i2s_in, PI_I2S_IOCTL_START, NULL);
    pi_i2s_ioctl(&i2s_out, PI_I2S_IOCTL_START, NULL);


    
#if GVSOC != 1
    #if AK4332 == 1
    fxl6408_setup();
    // Setup 2 DAC
    if(setup_dac(0) || setup_dac(1))
    {
        printf("Failed to setup DAC\n");
        return -1;
    }
    #else
    pi_device_t ssm_i2c_0;
    pi_device_t ssm_i2c_1;
    // Setup 2 DAC
    if(initialize_ssm6515(&ssm_i2c_0,(0x34 << 1)) || initialize_ssm6515(&ssm_i2c_1,(0x36 << 1)))
    {
        printf("Failed to setup DAC\n");
        return -1;
    }
    #endif

    pi_time_wait_us(100000);
    //printf("Setup DAC OK\n"); 

    //Enable slicer
    i2c_slider = pi_l2_malloc(sizeof(pi_device_t));
    init_ads1014(i2c_slider);
    #endif
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
        return 18;        
    }
    
    // Allocate L3 buffers for audio IN/OUT
    if (pi_ram_alloc(&DefaultRam, &inSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("inSig Ram malloc failed !\n");
        return -4;
    }
    if (pi_ram_alloc(&DefaultRam, &outSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("outSig Ram malloc failed !\n");
        return -5;
    }

    // Read audio from file
    printf("Reading wav from: %s \n", WavName);
    printf("buffer size: %d\n",AUDIO_BUFFER_SIZE);
    header_struct header_info;
      if (ReadWavFromFile(WavName,
            __PREFIX(_L2_Memory), AUDIO_BUFFER_SIZE*sizeof(short), &header_info)){
        printf("\nError reading wav file\n");
        return 1;
    }
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    printf("Num Samples: %d with BitsPerSample: %d\n", num_samples, header_info.BitsPerSample);
    printf("Finished Read wav.\n");


    if(num_samples*sizeof(short) > denoiser_L2_SIZE){
        printf("The size of the audio exceeds the available L2 memory space!\n");
        return 1;
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
    PRINTF("Setup STFT task!\n");
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
    PRINTF("Setup Cluster Task for inference!\n");
    struct pi_cluster_task* task_net;
    task_net = pi_l2_malloc(sizeof(struct pi_cluster_task));
    pi_cluster_task(task_net,&RunDenoiser,NULL);
    if(task_net==NULL) {
      PRINTF("pi_cluster_task alloc Error!\n");
      return -1;
    }
    pi_cluster_task_stacks(task_net, NULL, SLAVE_STACK_SIZE);
    PRINTF("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
    
    // Reset LSTM
    ResetLSTM = 1;
    for(int i=0; i<RNN_STATE_DIM_0; i++){
        RNN_STATE_0_I[i] = (DATATYPE_SIGNAL_INF) ZERO;
    }
    for(int i=0; i<RNN_STATE_DIM_1; i++){
        RNN_STATE_1_I[i] = (DATATYPE_SIGNAL_INF) ZERO;
    }

    /******
        Denoiser NN constructor
    ******/
    PRINTF("\n\nDenoiser Constructor\n");
    int err_construct = __PREFIX(CNN_Construct)();
    if (err_construct)
    {
        PRINTF("Graph constructor exited with error: %d\n", err_construct);
        return -5;
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
    pi_sfu_graph_load(sfu_graph);


    while(1){
        #if GVSOC != 1
        slider_value = ads1014_read(i2c_slider, 0);
        #endif
        pi_evt_wait(&proc_task);

#if AUDIO_EVK
        pi_gpio_pin_write(gpio_pin_o, 1);
#endif

        int round = (chunk_in_cnt%CHUNK_NUM);
        int round_out = (chunk_in_cnt>(STRUCT_DELAY-1))? ((chunk_in_cnt-(STRUCT_DELAY-1))%CHUNK_NUM):0;

        //First Copy previous loop processed frame to output
        for(int i=0;i<BUFF_SIZE/4;i++) {
            ((int32_t*)sfu_in_buffers[round_out].data)[i]= (int32_t)((float)(Audio_Frame_temp[i])*((int)(1<<Q_BIT_OUT)));
        }


        for(int i=0;i<FRAME_SIZE-FRAME_STEP;i++){
            Audio_Frame[i] = Audio_Frame[i+FRAME_STEP];
            Audio_Frame_temp[i] = Audio_Frame_temp[i+FRAME_STEP];
        }

        for(int i=0;i<FRAME_STEP;i++){
            Audio_Frame[i+FRAME_SIZE-FRAME_STEP] = (DATATYPE_SIGNAL)(((float)((int32_t *)sfu_out_buffers[round].data)[i]) /((int)(1<<Q_BIT_IN)));
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
            return -1;
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
        sprintf(WavName, "%s/samples/mags_%.4d.bin", BUILD_DIR, frame_id);
        printf("File being read is : %s\n", WavName);

        file[0] = pi_fs_open(&fs, WavName, 0);

        if (file[0] == 0) {
            printf("Failed to open file, %s\n", WavName); 
            return 7;
        }
        printf("File %x of size %d\n", file[0], sizeof(pi_fs_file_t));

        int TotBytes = sizeof(float)*AT_INPUT_WIDTH*AT_INPUT_HEIGHT;
        int len = pi_fs_read(file[0], STFT_Spectrogram, TotBytes);
        printf("Bytes read %d - of %d bytes expected\n", len,TotBytes );
        if (len != TotBytes){
            printf("Too few bytes in %s\n", WavName); 
            return 8;
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
                return -1;
            }            
        }

#endif //CHECKSUM

#if AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 0);
#endif

#endif  // DISABLE_NN_INFERENCE



#if IS_INPUT_STFT == 0 // if not loading the STFT

#if AUDIO_EVK
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
            return -1;
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
        PRINTF("\nAudio Out 2: ");

        for (int i= 0 ; i<FRAME_SIZE; i++){
            PRINTF("%f,", Audio_Frame[i] );
        }

        PRINTF("\n");



#if IS_SFU == 1

        // block until next input audio frame is ready
#if AUDIO_EVK
        pi_gpio_pin_write( gpio_pin_o, 0);
#endif
        chunk_in_cnt++;
        pi_evt_sig_init(&proc_task);


#else // audio from file 

        // if denoising auio files, outputs are loaded to the L3 output buffer outSig
        PRINTF("Writing Frame %d/%d to the output buffer\n\n", frame_id+1, tot_frames);

        // Cast if needed
        pi_ram_read(&DefaultRam,  (uint32_t)((short *) outSig + (frame_id*FRAME_STEP)), 
            Audio_Frame_temp, FRAME_SIZE * sizeof(short));

        // from DATA_S
        for (int i= 0 ; i<FRAME_SIZE; i++){
            Audio_Frame_temp[i] += (short int)(Audio_Frame[i] * (1<<15));
        }
        pi_ram_write(&DefaultRam,  (uint32_t) ((short *) outSig + (frame_id*FRAME_STEP)),   
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
        return 18;        
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
        return -1;
    }


#else //CHECKSUM
    // final sample
    PRINTF("\nAudio Out: ");
    for (int i= 0 ; i<num_samples; i++){
        PRINTF("%f, ", ((float) out_temp_buffer[i] )/(1<<15)  );
    }
    PRINTF("\n");


    char output_wav[64];
    #ifdef USE_MAKEFILE
    sprintf(output_wav, "../../../test_gap.wav");
    #else
    sprintf(output_wav, "../test_gap.wav");
    #endif
    WriteWavToFile(output_wav, 16, 16000, 1, 
        (uint32_t *) __PREFIX(_L2_Memory), num_samples* sizeof(short));
    printf("Writing wav file to test_gap.wav completed successfully\n");
#endif //CHECKSUM

    pi_l2_free(__PREFIX(_L2_Memory),denoiser_L2_SIZE);
#endif //IS_INPUT_STFT == 0 && IS_SFU == 0


    // Close the cluster
    pi_cluster_close(&cluster_dev);
    PRINTF("Ended\n");
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

    return denoiser();
}
