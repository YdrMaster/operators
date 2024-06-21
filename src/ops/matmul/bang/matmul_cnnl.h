#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__

#include "../../../operators.h"

void matmul_cnnl_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream);

#endif// __CNNL_MATMUL_H__
