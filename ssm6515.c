#include "ssm6515.h"
#include "ssm6515_register_map.h"

/*
 * Helpers to write I2C registers.
 */
int write_reg8(pi_device_t *dev, uint8_t addr, uint8_t value)
    {
    uint8_t buffer[2] = { addr, value };
    if(PI_OK != pi_i2c_write(dev, buffer, 2, PI_I2C_XFER_START | PI_I2C_XFER_STOP))
        {
        printf("Writing failed\n");
        return -1;
        }
    return 0;
    }

uint8_t read_reg8(pi_device_t *dev, uint8_t addr)
    {
    uint8_t result;
    pi_i2c_write_read(dev, &addr, &result, 1, 1);
    return result;
    }

int initialize_ssm6515(pi_device_t* ssm_i2c, uint8_t addr)
    {
    pi_pad_function_set(PI_PAD_042,  PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_043,  PI_PAD_FUNC0);

    struct pi_i2c_conf conf;

    ssm6515_pwr_ctrl    my_ssm_power_control;
    ssm6515_amp_ctrl    my_ssm_amp_ctrl;
    ssm6515_pdm_ctrl    my_ssm_pdm_ctrl;
    ssm6515_dac_ctrl1   my_ssm_dac_ctrl1;
    ssm6515_dac_ctrl2   my_ssm_dac_ctrl2;
    ssm6515_dac_ctrl3   my_ssm_dac_ctrl3;
    ssm6515_dac_hf_clip my_ssm_dac_hf_clip;
    ssm6515_fault_ctrl  my_ssm_fault_ctrl;

    uint8_t my_register_content;


    pi_i2c_conf_init(&conf);
    conf.itf = 1;
    conf.max_baudrate = 100000;
    pi_i2c_conf_set_slave_addr(&conf, addr, 0);

    pi_open_from_conf(ssm_i2c, &conf);
    if (pi_i2c_open(ssm_i2c))
        {
        return -1;
        }

    //RESET (0x14)
    //SOFT_FULL_RESET (bit4) => 1 (full reset)
    write_reg8(ssm_i2c, 0x14, (0b1 << 4));

    pi_time_wait_us(5000); /* wait some time for full reset to be done */

    /* STATUS_CLR (0x12) */
    /* STAT_CLR (bit0) => 1 (clear status) */
    write_reg8(ssm_i2c, 0x12, (0b1 << 0));

    /*************************/
    /* read static registers */
    /*************************/

    uint8_t vendor_id = read_reg8(ssm_i2c, 0x00);
    const uint8_t expected_vendor_id = 0x41;
    if (vendor_id != expected_vendor_id)
        {
        printf("Got an invalid vendor_id: 0x%x, expected 0x%x\n",
               vendor_id,
               expected_vendor_id);
        return -1;
        }

    uint8_t device_id1 = read_reg8(ssm_i2c, 0x01);
    const uint8_t expected_device_id1 = 0x65;
    if (device_id1 != expected_device_id1)
        {
        printf("Got an invalid device_id1: 0x%x, expected 0x%x\n",
               device_id1,
               expected_device_id1);
        return -1;
        }

    uint8_t device_id2 = read_reg8(ssm_i2c, 0x02);
    const uint8_t expected_device_id2 = 0x15;
    if (device_id2 != expected_device_id2)
        {
        printf("Got an invalid device_id2: 0x%x, expected 0x%x\n",
               device_id2,
               expected_device_id2);
        return -1;
        }

    uint8_t revision = read_reg8(ssm_i2c, 0x03);
    const uint8_t expected_revision = 0x02;
    if (revision != expected_revision)
        {
        printf("Got an invalid revision: 0x%x, expected 0x%x\n",
               revision,
               expected_revision);
        return -1;
        }

    /*********************************/
    /* write configuration registers */
    /*********************************/

    //PWR_CTRL
    my_ssm_power_control.system_power_down           = TRUE;
    my_ssm_power_control.automatic_power_down_enable = DISABLE;
    my_ssm_power_control.limiter_enable              = DISABLE;

    my_register_content = *(uint8_t *)&my_ssm_power_control;
    write_reg8(ssm_i2c, SSM6515_PWR_CTRL, my_register_content);

    //AMP_CTRL
    my_ssm_amp_ctrl.amplifier_low_power_mode_enable         = DISABLE;
    my_ssm_amp_ctrl.EMI_mode                                = OFF;
    my_ssm_amp_ctrl.amplifier_resistive_load_selection      = SSM6515_AMPLIFIER_RESISTIVE_LOAD_32_OHMS;
    my_ssm_amp_ctrl.amplifier_overcurrent_protection_enable = ENABLE;

    my_register_content = *(uint8_t *)&my_ssm_amp_ctrl;
    write_reg8(ssm_i2c, SSM6515_AMP_CTRL, my_register_content);

    //PDM CTRL
    my_ssm_pdm_ctrl.pdm_mode               = SSM6515_PDM_MODE_AS_INPUT;
    my_ssm_pdm_ctrl.pdm_sampling_frequency = SSM6515_PDM_SAMPLE_RATE_2_8224_TO_3_072MHZ;
    my_ssm_pdm_ctrl.pdm_channel_selection  = SSM6515_PDM_CHANNEL_SELECTION_RISING_EDGE;
    my_ssm_pdm_ctrl.pdm_filter             = SSM6515_PDM_INPUT_FILTERING_LOWEST;
    my_ssm_pdm_ctrl.pdm_phase_selection    = SSM6515_PDM_PHASE_SELECTION_FALL_RISE;

    my_register_content = *(uint8_t *)&my_ssm_pdm_ctrl;
    write_reg8(ssm_i2c, SSM6515_PDM_CTRL, my_register_content);

    //DAC_CTRL (0x07)
    my_ssm_dac_ctrl1.dac_sampling_frequency = SSM6515_DAC_SAMPLE_RATE_44_1_TO_48kHZ;
    my_ssm_dac_ctrl1.dac_power_mode         = SSM6515_DAC_POWER_MODE_NO_SAVINGS;
    my_ssm_dac_ctrl1.dac_i_bias             = SSM6515_DAC_BIAS_CONTROL_NORMAL;

    my_register_content = *(uint8_t *)&my_ssm_dac_ctrl1;
    write_reg8(ssm_i2c, SSM6515_DAC_CTRL1, my_register_content);

    //DAC_CTRL2 (0x08)
    my_ssm_dac_ctrl2.dac_mute                           = DISABLE;
    my_ssm_dac_ctrl2.dac_volume_bypass_fixed_gain       = SSM6515_DAC_VOLUME_MODE_VOLUME_BYPASSED_ODB;
    my_ssm_dac_ctrl2.dac_additional_filtering           = SSM6515_DAC_MORE_FILTERING_NORMAL;
    my_ssm_dac_ctrl2.dac_volume_zero_crossing_control   = ENABLE;
    my_ssm_dac_ctrl2.dac_hard_volume                    = DISABLE;
    my_ssm_dac_ctrl2.dac_high_performance_mode_enable   = DISABLE;
    my_ssm_dac_ctrl2.dac_signal_phase_inversion_enable  = DISABLE;

    my_register_content = *(uint8_t *)&my_ssm_dac_ctrl2;
    write_reg8(ssm_i2c, SSM6515_DAC_CTRL2, my_register_content);

    //DAC_CTRL3 (0x09)
    //high pass filter (disabled by default)

    //DAC_VOL (0x0A)
    // Volume control (not needed, volume control is disabled)

    //DAC_CLIP (0x0B)
    my_ssm_dac_hf_clip.dac_high_frequency_clip_value = HIGH_FREQUENCY_CLIP_ON_256(255);

    my_register_content = *(uint8_t *)&my_ssm_dac_hf_clip;
    write_reg8(ssm_i2c, SSM6515_DAC_HF_CLIP, my_register_content);

    //FAULT_CTRL (0x11)
    my_ssm_fault_ctrl.overcurrent_automatic_fault_recovery        = OFF;
    my_ssm_fault_ctrl.overtemperature_automatic_fault_recovery    = OFF;
    my_ssm_fault_ctrl.undervoltage_automatic_fault_recovery       = OFF;
    my_ssm_fault_ctrl.manual_fault_recovery                       = OFF;

    my_register_content = *(uint8_t *)&my_ssm_fault_ctrl;
    write_reg8(ssm_i2c, SSM6515_FAULT_CTRL, my_register_content);

    //PWR_CTRL
    //SWPDN   (bit 0) => 0 (normal operation, 1 is powerdown)
    //APWD_EN (bit 1) => 0 (automatic power down disabled)
    //LIM_EN  (bit 4) => 0 (limiter disabled)
    my_ssm_power_control.system_power_down           = FALSE;
    my_ssm_power_control.automatic_power_down_enable = DISABLE;
    my_ssm_power_control.limiter_enable              = DISABLE;

    my_register_content = *(uint8_t *)&my_ssm_power_control;
    write_reg8(ssm_i2c, SSM6515_PWR_CTRL, my_register_content);

    return 0;
    }