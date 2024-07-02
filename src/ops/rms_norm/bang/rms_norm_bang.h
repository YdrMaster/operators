#ifndef __BANG_RMS_NORM_H__
#define __BANG_RMS_NORM_H__

#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void rms_norm_bang_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif// __BANG_RMS_NORM_H__
