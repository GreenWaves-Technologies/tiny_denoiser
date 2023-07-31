#ifndef _SSM6515_REGISTER_MAP_H_
#define _SSM6515_REGISTER_MAP_H_

#define SSM6515_VENDOR_ID		0x00
#define SSM6515_DEVICE_ID1		0x01
#define SSM6515_DEVICE_ID2		0x02
#define SSM6515_REVISION		0x03
#define SSM6515_PWR_CTRL		0x04
#define SSM6515_CLK_CTRL		0x05
#define SSM6515_PDM_CTRL		0x06
#define SSM6515_DAC_CTRL1		0x07
#define SSM6515_DAC_CTRL2		0x08
#define SSM6515_DAC_CTRL3		0x09
#define SSM6515_DAC_VOL			0x0A
#define SSM6515_DAC_HF_CLIP		0x0B
#define SSM6515_SPT_CTRL1		0x0C
#define SSM6515_SPT_CTRL2		0x0D
#define SSM6515_AMP_CTRL		0x0E
#define SSM6515_LIM_CTRL		0x0F
#define SSM6515_LIM_CTRL2		0x10
#define SSM6515_FAULT_CTRL		0x11
#define SSM6515_STATUS_CLR		0x12
#define SSM6515_STATUS			0x13
#define SSM6515_RESET			0x14

typedef struct ssm6515_vendor_id
    {
    uint8_t  vendor;
    } ssm6515_vendor_id;
typedef struct ssm6515_device_id1
    {
    uint8_t device1;
    } ssm6515_device_id1;
typedef struct ssm6515_device_id2
    {
    uint8_t device2;
    } ssm6515_device_id2;
typedef struct ssm6515_revision
    {
    uint8_t revision;
    } ssm6515_revision;
typedef struct ssm6515_pwr_ctrl
    {
    uint8_t system_power_down:1;
	uint8_t automatic_power_down_enable:1;
	uint8_t :3;
	uint8_t limiter_enable:1;
	uint8_t :3;
    } ssm6515_pwr_ctrl;
typedef struct ssm6515_clk_ctrl
    {
    uint8_t bit_clock_rate:5;
	uint8_t :3;
    } ssm6515_clk_ctrl;
typedef struct ssm6515_pdm_ctrl
    {
    uint8_t pdm_mode:1;
	uint8_t pdm_sampling_frequency:2;
	uint8_t pdm_channel_selection:1;
	uint8_t pdm_filter:2;
	uint8_t :1;
	uint8_t pdm_phase_selection:1;
    } ssm6515_pdm_ctrl;
typedef struct ssm6515_dac_ctrl1
    {
    uint8_t dac_sampling_frequency:4;
	uint8_t dac_power_mode:2;
	uint8_t dac_i_bias:2;
    } ssm6515_dac_ctrl1;
typedef struct ssm6515_dac_ctrl2
    {
    uint8_t dac_mute:1;
	uint8_t dac_volume_bypass_fixed_gain:2;
	uint8_t dac_additional_filtering:1;
	uint8_t dac_volume_zero_crossing_control:1;
	uint8_t dac_hard_volume:1;
	uint8_t dac_high_performance_mode_enable:1;
	uint8_t dac_signal_phase_inversion_enable:1;
    } ssm6515_dac_ctrl2;
typedef struct ssm6515_dac_ctrl3
    {
    uint8_t dac_channel0_high_pass_filter_enable:1;
	uint8_t :3;
	uint8_t dac_high_pass_filter_cutoff_frequency:4;
    } ssm6515_dac_ctrl3;
typedef struct ssm6515_dac_vol
    {
    uint8_t dac_volume_control;
    } ssm6515_dac_vol;
typedef struct ssm6515_dac_hf_clip
    {
    uint8_t dac_high_frequency_clip_value;
    } ssm6515_dac_hf_clip;
typedef struct ssm6515_spt_ctrl1
    {
    uint8_t serial_port_SAI_mode:1;
	uint8_t serial_port_data_format:3;
	uint8_t serial_port_TDM_slot_width:2;
	uint8_t serial_port_BCLK_polarity:1;
	uint8_t serial_port_LRCLK_polarity:1;
    } ssm6515_spt_ctrl1;
typedef struct ssm6515_spt_ctrl2
    {
    uint8_t serial_port_slot_selection:5;
	uint8_t :3;
    } ssm6515_spt_ctrl2;
typedef struct ssm6515_amp_ctrl
    {
    uint8_t amplifier_low_power_mode_enable:1;
	uint8_t EMI_mode:1;
	uint8_t amplifier_resistive_load_selection:2;
	uint8_t amplifier_overcurrent_protection_enable:1;
	uint8_t :3;
    } ssm6515_amp_ctrl;
typedef struct ssm6515_lim_ctrl
    {
    uint8_t limiter_attack_rate:2;
	uint8_t :2;
	uint8_t limiter_release_rate:2;
	uint8_t :2;
    } ssm6515_lim_ctrl;
typedef struct ssm6515_lim_ctrl2
    {
    uint8_t limiter_threshold:5;
	uint8_t :3;
    } ssm6515_lim_ctrl2;
typedef struct ssm6515_fault_ctrl
    {
    uint8_t overcurrent_automatic_fault_recovery:1;
	uint8_t overtemperature_automatic_fault_recovery:1;
	uint8_t undervoltage_automatic_fault_recovery:1;
	uint8_t :1;
	uint8_t manual_fault_recovery:1;
	uint8_t :3;
    } ssm6515_fault_ctrl;
typedef struct ssm6515_status_clr
    {
    uint8_t clear_status:1;
	uint8_t :7;
    } ssm6515_status_clr;
typedef struct ssm6515_status
    {
    uint8_t amplifier_overcurrent_fault_status:1;
	uint8_t overtemperature_fault_status:1;
	uint8_t overtemperature_warning_status:1;
	uint8_t undervoltage_fault_condition:1;
	uint8_t clock_ratio_error_status:1;
	uint8_t serial_port_error_status:1;
	uint8_t dac_output_clipping_status:1;
	uint8_t limiter_gain_reduction_active:1;
    } ssm6515_status;
typedef struct ssm6515_reset
    {
    uint8_t soft_reset:1;
	uint8_t :3;
	uint8_t soft_full_reset:1;
	uint8_t :3;
    } ssm6515_reset	;

#define VOLUME_DB(X)  (64-(X<<3)/3)
#define VOLUME_MUTE  0xFF

#define HIGH_FREQUENCY_CLIP_ON_256(X)  (X-1)
#define HIGH_FREQUENCY_CLIP_DISABLED  0xFF

#define SSM6515_PDM_MODE_NORMAL_SAI 0
#define SSM6515_PDM_MODE_AS_INPUT   1

#define SSM6515_PDM_CHANNEL_SELECTION_RISING_EDGE    0
#define SSM6515_PDM_CHANNEL_SELECTION_FALLING_EDGE   1

#define SSM6515_PDM_SAMPLE_RATE_5_6448_TO_6_144MHZ   0b00
#define SSM6515_PDM_SAMPLE_RATE_2_8224_TO_3_072MHZ   0b01

#define SSM6515_PDM_INPUT_FILTERING_HIGHEST 0b00
#define SSM6515_PDM_INPUT_FILTERING_MEDIUM  0b01
#define SSM6515_PDM_INPUT_FILTERING_LOWEST  0b10

#define SSM6515_PDM_PHASE_SELECTION_FALL_RISE    0
#define SSM6515_PDM_PHASE_SELECTION_RISE_FALL    1

#define SSM6515_DAC_BIAS_CONTROL_NORMAL 				0b00
#define SSM6515_DAC_BIAS_CONTROL_POWER_SAVINGS 			0b01
#define SSM6515_DAC_BIAS_CONTROL_IMPROVED_PERFORMANCE 	0b10

#define SSM6515_DAC_POWER_MODE_NO_SAVINGS         0b00
#define SSM6515_DAC_POWER_MODE_POWER_SAVING       0b01
#define SSM6515_DAC_POWER_MODE_MOST_POWER_SAVING  0b10

#define SSM6515_DAC_SAMPLE_RATE_8kHZ			0b0000
#define SSM6515_DAC_SAMPLE_RATE_12kHZ   		0b0001
#define SSM6515_DAC_SAMPLE_RATE_16kHZ   		0b0010
#define SSM6515_DAC_SAMPLE_RATE_24kHZ   		0b0011
#define SSM6515_DAC_SAMPLE_RATE_32kHZ   		0b0100
#define SSM6515_DAC_SAMPLE_RATE_44_1_TO_48kHZ   0b0101
#define SSM6515_DAC_SAMPLE_RATE_88_2_TO_96kHZ   0b0110
#define SSM6515_DAC_SAMPLE_RATE_176_4_TO_192kHZ 0b0111
#define SSM6515_DAC_SAMPLE_RATE_384kHZ			0b1000
#define SSM6515_DAC_SAMPLE_RATE_768kHZ   		0b1001

#define SSM6515_DAC_VOLUME_MODE_VOLUME_ENABLED          0b00
#define SSM6515_DAC_VOLUME_MODE_VOLUME_BYPASSED_ODB     0b01
#define SSM6515_DAC_VOLUME_MODE_VOLUME_BYPASSED_6DB     0b10

#define SSM6515_DAC_MORE_FILTERING_NORMAL                       0b00
#define SSM6515_DAC_MORE_FILTERING_ADDITIONAL_INTERPOLATION     0b01

#define SSM6515_SP_TDM_SLOT_WIDTH_32_BCLK 0b00
#define SSM6515_SP_TDM_SLOT_WIDTH_16_BCLK 0b01
#define SSM6515_SP_TDM_SLOT_WIDTH_24_BCLK 0b10

#define SSM6515_SP_DATA_FORMAT_I2S_MODE_DELAY_BY_1			0b000
#define SSM6515_SP_DATA_FORMAT_LEFT_JUSTIFIED_DELAY_BY_0	0b001
#define SSM6515_SP_DATA_FORMAT_DELAY_BY_8					0b010
#define SSM6515_SP_DATA_FORMAT_DELAY_BY_12					0b011
#define SSM6515_SP_DATA_FORMAT_DELAY_BY_16					0b100

#define SM6515_I2S_LEFT_CHANNEL  	0
#define SM6515_I2S_RIGHT_CHANNEL 	1
#define SM6515_TDM_SLOT_1			0b00000
#define SM6515_TDM_SLOT_2			0b00001
#define SM6515_TDM_SLOT_3			0b00010
#define SM6515_TDM_SLOT_4			0b00011
#define SM6515_TDM_SLOT_5			0b00100
#define SM6515_TDM_SLOT_6			0b00101
#define SM6515_TDM_SLOT_7			0b00110
#define SM6515_TDM_SLOT_8			0b00111
#define SM6515_TDM_SLOT_9			0b01000
#define SM6515_TDM_SLOT_10			0b01001
#define SM6515_TDM_SLOT_11			0b01010
#define SM6515_TDM_SLOT_12			0b01011
#define SM6515_TDM_SLOT_13			0b01100
#define SM6515_TDM_SLOT_14			0b01101
#define SM6515_TDM_SLOT_15			0b01110
#define SM6515_TDM_SLOT_16			0b01111
#define SM6515_TDM_SLOT_17			0b10000
#define SM6515_TDM_SLOT_18			0b10001
#define SM6515_TDM_SLOT_19			0b10010
#define SM6515_TDM_SLOT_20			0b10011
#define SM6515_TDM_SLOT_21			0b10100
#define SM6515_TDM_SLOT_22			0b10101
#define SM6515_TDM_SLOT_23			0b10110
#define SM6515_TDM_SLOT_24			0b10111
#define SM6515_TDM_SLOT_25			0b11000
#define SM6515_TDM_SLOT_26			0b11001
#define SM6515_TDM_SLOT_27			0b11010
#define SM6515_TDM_SLOT_28			0b11011
#define SM6515_TDM_SLOT_29			0b11100
#define SM6515_TDM_SLOT_30			0b11101
#define SM6515_TDM_SLOT_31			0b11110
#define SM6515_TDM_SLOT_32			0b11111

#define SSM6515_AMPLIFIER_RESISTIVE_LOAD_8_OHMS  0b00
#define SSM6515_AMPLIFIER_RESISTIVE_LOAD_16_OHMS 0b01
#define SSM6515_AMPLIFIER_RESISTIVE_LOAD_24_OHMS 0b10
#define SSM6515_AMPLIFIER_RESISTIVE_LOAD_32_OHMS 0b11

#define SSM6515_AUDIO_LIMITER_RELEASE_RATE_3200_MS 0b00
#define SSM6515_AUDIO_LIMITER_RELEASE_RATE_1600_MS 0b01
#define SSM6515_AUDIO_LIMITER_RELEASE_RATE_1200_MS 0b10
#define SSM6515_AUDIO_LIMITER_RELEASE_RATE_800_MS  0b11

#define SSM6515_AUDIO_LIMITER_ATTACK_RATE_120_US 0b00
#define SSM6515_AUDIO_LIMITER_ATTACK_RATE_160_US 0b01
#define SSM6515_AUDIO_LIMITER_ATTACK_RATE_30_US  0b10
#define SSM6515_AUDIO_LIMITER_ATTACK_RATE_20_US  0b11

#define SSM6515_LIMITER_THRESHOLD_3dB			0b00000
#define SSM6515_LIMITER_THRESHOLD_2_5dB			0b00001
#define SSM6515_LIMITER_THRESHOLD_2dB			0b00010
#define SSM6515_LIMITER_THRESHOLD_1_5dB			0b00011
#define SSM6515_LIMITER_THRESHOLD_1dB			0b00100
#define SSM6515_LIMITER_THRESHOLD_0_5dB			0b00101
#define SSM6515_LIMITER_THRESHOLD_0dB			0b00110
#define SSM6515_LIMITER_THRESHOLD_MINUS_0_5dB	0b00111
#define SSM6515_LIMITER_THRESHOLD_MINUS_1dB		0b01000
#define SSM6515_LIMITER_THRESHOLD_MINUS_1_5dB	0b01001
#define SSM6515_LIMITER_THRESHOLD_MINUS_2dB		0b01010
#define SSM6515_LIMITER_THRESHOLD_MINUS_2_5dB	0b01011
#define SSM6515_LIMITER_THRESHOLD_MINUS_3dB		0b01100
#define SSM6515_LIMITER_THRESHOLD_MINUS_3_5dB	0b01101
#define SSM6515_LIMITER_THRESHOLD_MINUS_4dB		0b01110
#define SSM6515_LIMITER_THRESHOLD_MINUS_4_5dB	0b01111
#define SSM6515_LIMITER_THRESHOLD_MINUS_5dB		0b10000
#define SSM6515_LIMITER_THRESHOLD_MINUS_5_5dB	0b10001
#define SSM6515_LIMITER_THRESHOLD_MINUS_6dB		0b10010
#define SSM6515_LIMITER_THRESHOLD_MINUS_6_5dB	0b10011
#define SSM6515_LIMITER_THRESHOLD_MINUS_7dB		0b10100
#define SSM6515_LIMITER_THRESHOLD_MINUS_7_5dB	0b10101
#define SSM6515_LIMITER_THRESHOLD_MINUS_8dB		0b10110
#define SSM6515_LIMITER_THRESHOLD_MINUS_8_5dB	0b10111
#define SSM6515_LIMITER_THRESHOLD_MINUS_9dB		0b11000
#define SSM6515_LIMITER_THRESHOLD_MINUS_9_5dB	0b11001
#define SSM6515_LIMITER_THRESHOLD_MINUS_10dB	0b11010
#define SSM6515_LIMITER_THRESHOLD_MINUS_10_5dB	0b11011
#define SSM6515_LIMITER_THRESHOLD_MINUS_11dB	0b11100
#define SSM6515_LIMITER_THRESHOLD_MINUS_11_5dB	0b11101
#define SSM6515_LIMITER_THRESHOLD_MINUS_12dB	0b11110
#define SSM6515_LIMITER_THRESHOLD_MINUS_12_5dB	0b11111

#define TRUE  1
#define FALSE 0
#define ENABLE 1
#define DISABLE 0
#define ON 1
#define OFF 0

#endif