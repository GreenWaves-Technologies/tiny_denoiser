#pragma once
#include "pmsis.h"
#include "bsp/bsp.h"

int write_reg8(pi_device_t *dev, uint8_t addr, uint8_t value);
uint8_t read_reg8(pi_device_t *dev, uint8_t addr);
int initialize_ssm6515(pi_device_t* ssm_i2c, uint8_t addr);