#pragma once
#include <stdint.h>



/*
 * \brief set up the AK4332 (PDM DAC)
 *
 * \return 0 if successful, an error code otherwise
 */
int setup_dac(uint8_t id);
int fxl6408_setup(void);