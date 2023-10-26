#ifndef ROCC_TEST_FIONA_PARAMS
#define ROCC_TEST_FIONA_PARAMS

#include <stdint.h>
#include <limits.h>

#define PA_MZI_PORT 4
#define PA_MRR_DGRP 8
#define EU_VEC_ELEM 32

typedef int16_t elem_t;
typedef int32_t full_t;
constexpr elem_t elem_t_min = -32768;
constexpr elem_t elem_t_max = 32767;

#endif /* ROCC_TEST_FIONA_PARAMS */