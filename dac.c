#include "dac.h"

#include "pmsis.h"
#include "bsp/bsp.h" 

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
    pi_pad_set_function(PI_PAD_042,  PI_PAD_FUNC0);
    pi_pad_set_function(PI_PAD_043,  PI_PAD_FUNC0);

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

int setup_dac(uint8_t id)
{

    /* Choose the right ak4332 with I2C Mux */
    /* The I2C Mux is controlled by the GPIO A68 */
    struct pi_device gpio_ic_mux;
    struct pi_gpio_conf gpio_conf = {0};
    pi_gpio_e gpio_pin_o = PI_GPIO_A68; 

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio_ic_mux, &gpio_conf);
    gpio_conf.port = (gpio_pin_o & PI_GPIO_NUM_MASK) >> 5;
    if ( pi_gpio_open(&gpio_ic_mux) )
    {
        printf("Error opening GPIO \n");
        return -2;
    }

    pi_gpio_flags_e cfg_flags_out = PI_GPIO_OUTPUT|PI_GPIO_PULL_DISABLE|PI_GPIO_DRIVE_STRENGTH_LOW;

    /* Configure gpio output. */
    pi_gpio_pin_configure(&gpio_ic_mux, gpio_pin_o, cfg_flags_out);

    if (id) // Right channel
        pi_gpio_pin_write(&gpio_ic_mux, gpio_pin_o, 1);
    else    // Left channel
        pi_gpio_pin_write(&gpio_ic_mux, gpio_pin_o, 0);

    int errors = 0;
    pi_device_t i2c;
    struct pi_i2c_conf conf;
    pi_i2c_conf_init(&conf);
    conf.itf = 1;
    conf.max_baudrate = 100000;
    pi_i2c_conf_set_slave_addr(&conf, 0x20, 0);

    pi_open_from_conf(&i2c, &conf);
    if (pi_i2c_open(&i2c))
    {
        return -1;
    }

    // DAC initial settings
    write_reg8(&i2c, 0x26, 0x02);
    write_reg8(&i2c, 0x27, 0xC0);

    // program CM and FS
    write_reg8(&i2c, 0x5, 0b0001010); //FS-48kHz
    // write_reg8(&i2c, 0x5, 0b0001001); //FS-44.1kHz

    // PDM bit and PDMMODE (DSD)
    write_reg8(&i2c, 0x8, 0b101);

    //Set HP Gain to 0
    write_reg8(&i2c, 0x0d, 0b101);
    //Set DAC volume to max
    write_reg8(&i2c, 0x0b, 0x1F);

    // Configure PLL to take BLCK/DSDCLK as input clock
    write_reg8(&i2c, 0x0E, 0x1);

    // Configure DAC to take PLL as input clock
    write_reg8(&i2c, 0x13, 0x1);

    // Configure DAC clock divider
    write_reg8(&i2c, 0x14, 0x1);

    int pld = 3;
    write_reg8(&i2c, 0x0F, pld >> 8);
    write_reg8(&i2c, 0x10, pld & 0xff);
    int plm = 31;
    write_reg8(&i2c, 0x11, plm >> 8);
    write_reg8(&i2c, 0x12, plm & 0xff);

    // set volume to max
    write_reg8(&i2c, 0x0b, 0x1f);
    write_reg8(&i2c, 0x0d, 0x7);

    // Power-up PLL
    write_reg8(&i2c, 0x00, 0x1);

    pi_time_wait_us(20000);

    // Power-up PMTIM
    write_reg8(&i2c, 0x00, 0x3);

    // Power-up charge pump for both channels
    write_reg8(&i2c, 0x01, 0x1);

    pi_time_wait_us(65000);

    // Power-up LDO1
    write_reg8(&i2c, 0x01, 0x31);

    pi_time_wait_us(5000);

    // Power up charge pump 2
    write_reg8(&i2c, 0x01, 0x33);

    // Power-up DAC
    write_reg8(&i2c, 0x02, 0x1);

    // Power-up Amplifier
    write_reg8(&i2c, 0x03, 0x1);

    pi_i2c_close(&i2c);

    return 0;
}
