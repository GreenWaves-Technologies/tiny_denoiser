
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

#define __PREFIX(x) denoiser_GRU ## x
#include "denoiser_GRUKernels.h"

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash);

#ifdef SILENT
# define PRINTF(...) ((void) 0)
#else
# define PRINTF printf
#endif  /* DEBUG */

#endif