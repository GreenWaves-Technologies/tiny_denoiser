
#ifndef __KWS_H__
#define __KWS_H__

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

#define __PREFIX(x) denoiser ## x
#include "denoiserKernels.h"
#include "denoiserInfo.h"
#define denoiser_L1_SIZE _denoiser_L1_Memory_SIZE
#define denoiser_L2_SIZE _denoiser_L2_Memory_SIZE

#define SCALE_IN denoiser_Input_1_OUT_SCALE
#define SCALE_OUT denoiser_Output_1_OUT_SCALE

extern AT_DEFAULTFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash);

#ifdef SILENT
# define PRINTF(...) ((void) 0)
#else
# define PRINTF printf
#endif  /* DEBUG */

#endif
