#include "dac.h"

#include "pmsis.h"
#include "bsp/bsp.h" 

int setup_dac(uint8_t addr);


static int write_reg8(pi_device_t *dev, uint8_t addr, uint8_t value)
{

    uint8_t buffer[2] = { addr, value };
    if (pi_i2c_write(dev, buffer, 2, PI_I2C_XFER_START | PI_I2C_XFER_STOP))
    {
        printf("Error when writing \n");
        return -1;
    }
    // printf("Wrote %x at %x, read back %x\n", value, addr, read_reg8(dev, addr));
    return 0;
}

static uint8_t read_reg8(pi_device_t *dev, uint8_t addr)
{
    uint8_t result;
    pi_i2c_write_read(dev, &addr, &result, 1, 1);
    return result;
}


int fxl6408_setup()
{
    // Setting pads 42 & 43 to Alternate 0 function to enable I2C1 peripheral
    pi_pad_function_set(PI_PAD_042,  PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_043,  PI_PAD_FUNC0);

    //printf("Testing fxl6408\n");
    int errors = 0;
    pi_device_t i2c;
    struct pi_i2c_conf conf;
    pi_i2c_conf_init(&conf);
    conf.itf = 1;
    pi_i2c_conf_set_slave_addr(&conf, 0x86, 0);

    pi_open_from_conf(&i2c, &conf);
    if (pi_i2c_open(&i2c))
    {
        return -1;
    }

    // wait Xms before ak4332 pwd, to stabilize supply voltage
    pi_time_wait_us(100000);

    // Turn-on ak4332
    write_reg8(&i2c, 0x01, 0x1); // reset the GPIO expander

    write_reg8(&i2c, 0x03, 0x2);
    write_reg8(&i2c, 0x05, 0x2);
    write_reg8(&i2c, 0x07, 0x0);

    pi_i2c_close(&i2c);
    // Wait at least 1ms after ak4332 power-up
    pi_time_wait_us(2000);

    return 0;
}

int setup_dac(uint8_t addr)
{

    pi_pad_function_set(PI_PAD_040,  PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_041,  PI_PAD_FUNC0);

    pi_device_t ssm_i2c;
    struct pi_i2c_conf conf;
    pi_i2c_conf_init(&conf);
    conf.itf = 0;
    conf.max_baudrate = 100000;
    pi_i2c_conf_set_slave_addr(&conf, addr, 0);

    pi_open_from_conf(&ssm_i2c, &conf);
    if (pi_i2c_open(&ssm_i2c))
    {
        return -1;
    }


    //RESET (0x14)
    //SOFT_FULL_RESET (bit4) => 1 (full reset)
    write_reg8(&ssm_i2c, 0x14, (0b1 << 4));

    pi_time_wait_us(5000); /* wait some time for full reset to be done */

    /* STATUS_CLR (0x12) */
    /* STAT_CLR (bit0) => 1 (clear status) */
    write_reg8(&ssm_i2c, 0x12, (0b1 << 0));

    /*************************/
    /* read static registers */
    /*************************/

    uint8_t vendor_id = read_reg8(&ssm_i2c, 0x00);
    //const uint8_t expected_vendor_id = 0x41;
    const uint8_t expected_vendor_id = 0x41;

    if (vendor_id != expected_vendor_id)
    {
        printf("Got an invalid vendor_id: 0x%x, expected 0x%x\n",
                vendor_id,
                expected_vendor_id);
        return -1;
    }

    uint8_t device_id1 = read_reg8(&ssm_i2c, 0x01);
    const uint8_t expected_device_id1 = 0x65;
    if (device_id1 != expected_device_id1)
    {
        printf("Got an invalid device_id1: 0x%x, expected 0x%x\n",
                device_id1,
                expected_device_id1);
        return -1;
    }

    uint8_t device_id2 = read_reg8(&ssm_i2c, 0x02);
    const uint8_t expected_device_id2 = 0x15;
    if (device_id2 != expected_device_id2)
    {
        printf("Got an invalid device_id2: 0x%x, expected 0x%x\n",
                device_id2,
                expected_device_id2);
        return -1;
    }

    uint8_t revision = read_reg8(&ssm_i2c, 0x03);
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
    //SWPDN   (bit 0) => 1 (1 is powerdown)
    //APWD_EN (bit 1) => 0 (automatic power down disabled)
    //LIM_EN  (bit 4) => 0 (limiter disabled)
    write_reg8(&ssm_i2c, 0x04, 0x1);

    //AMP_CTRL
    //OCP_EN (bit4)      => 1 (overcurrent protection enabled)
    //AMP_RLOAD (bit3:2) => 0b11 (32ohms)
    //EMI_MODE (bit1)    => 0b0 (normal operation)
    //AMP_LPM (bit0)     => 0b0 (heaphones low power mode off)
    write_reg8(&ssm_i2c, 0x0E, (0b0 << 4) | (0b11 << 2));

    //PDM CTRL
    //PDM_MODE (bit0)     => 0 (SAI/TDM mode)
    //PDM_FS (bit2-1)     => 01 (3.072MHz)
    //PDM_CHAN_SEL (bit3) => 0 (rising)
    //PDM_FILT (bit 5:4)  => 10 (lowest filtering, lowest latency)
    write_reg8(&ssm_i2c, 0x06, (0b0 << 0) | (0b01 << 1) | (0b0 << 3) | (0b10 << 4));

    //DAC_CTRL (0x07)
    //DAC_IBIAS (bit 7:6) => 0 (normal operation)
    //DAC_PWR_MODE (bit5:4) => 0 (no power savings)
    //DAC_PWR_MODE (bit5:4) => 1 (power savings)
    //DAC_FS (bit 3:0) => 0x5 (48Khz) : 0x2 => 16kHz
    write_reg8(&ssm_i2c, 0x07, (0b00 << 6) | (0b00 << 4) | (0x5 << 0));

    //DAC_CTRL2 (0x08)
    //DAC_MUTE      (bit0)    => 0 (unmute)
    //DAC_VOL_MODE  (bit 2:1) => 01 (volume control bypassed, 0dB gain)
    //DAC_MORE_FILT (bit3)    => 0 (normal operation)
    //DAC_VOL_ZC    (bit4)    => 1 (volume change occurs at zero crossing)
    //DAC_HARD_VOL  (bit5)    => 0 (soft volume ramping)
    //DAC_PERF_MODE (bit6)    => 0 (normal operation)
    //DAC_INVERT    (bit7)    => 0 (no phase inversion)
    write_reg8(&ssm_i2c, 0x08, (0b01 << 1) | (0b1 << 4));

    //DAC_CTRL3 (0x09)
    //high pass filter (disabled by default)

    //DAC_VOL (0x0A)
    // Volume control (not needed, volume control is disabled)

    //DAC_CLIP (0x0B)
    //No clipping (0xFF)
    write_reg8(&ssm_i2c, 0x0B, 0xFE);

    //SPT_CTRL1 (0x0C)
    //SPT_LRCLK_POL   (bit 7)   => 0 (normal)
    //SPT_BCLK_POL    (bit 6)   => 0 (normal)
    //SPT_SLOT_WIDTH  (bit 5:4) => 00 (32 bits per slot)
    //SPT_DATA_FORMAT (bit 3:1) => 000 (typical I2S, delay by 1)
    //SPT_SAI_MODE    (bit 0)   => 0
    write_reg8(&ssm_i2c, 0x0C, 0x0);

    //SPT_CTRL2
    //SPT_SLOT_SEL (bit 4:0) => 00000 (slot 1)
    write_reg8(&ssm_i2c, 0x0D, 0x0);

    //FAULT_CTRL (0x11)
    //MRCV (bit 4)      => 0 (0 normal operation, 1 recovery attempt)
    //ARCV_ULVO (bit 2) => 0 (0 automic recovery, 1 manual recovery)
    //ARCV_OTF  (bit 1) => 0 (0 automic recovery, 1 manual recovery)
    //ARCV_OCP  (bit 0) => 0 (0 automic recovery, 1 manual recovery)
    write_reg8(&ssm_i2c, 0x11, (0b1 << 0) | (0b1 << 1) | (0b1 << 2));

    //PWR_CTRL
    //SWPDN   (bit 0) => 0 (normal operation, 1 is powerdown)
    //APWD_EN (bit 1) => 0 (automatic power down disabled)
    //LIM_EN  (bit 4) => 0 (limiter disabled)
    write_reg8(&ssm_i2c, 0x04, 0x0);

    return 0;
}

