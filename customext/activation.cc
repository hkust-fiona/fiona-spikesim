#include "decode.h"
#include "processor.h"
#include "rocc.h"
#include "fiona_opcodes.h"
/*
 * SPDX-FileCopyrightText: Copyright 2010-2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_activations_q15.c
 * Description:  Q15 neural network activation function using direct table look-up
 *
 * $Date:        8 September 2022
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

// Table of sigmoid(i/24) at 0.16 format - 256 elements.

// We use combined sigmoid and tanh look-up table, since
// tanh(x) = 2*sigmoid(2*x) -1.
// Both functions are symmetric, so the LUT table is only needed
// for the absolute value of the input.
static const uint16_t sigmoid_table_uint16[256] = {
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498,
    40149, 40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255,
    46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865,
    52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174,
    56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544, 58800, 59048, 59288,
    59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441,
    61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886,
    62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
    63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450,
    64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845,
    64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097,
    65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
    65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425,
    65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465,
    65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
    65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518,
    65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
    65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529,
    65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531,
    65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533,
    65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
    65534, 65534, 65535};

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

/*
 * @brief Neural network activation function using direct table look-up
 *
 * @note  Refer header file for details.
 *
 */

int16_t fiona_nn_activation_s16( int16_t type, int16_t input, uint16_t left_shift) {
    uint32_t abs_input_shift, max_saturation;
    switch (type)
    {
        case ACT_BITS_24_20_SIGM: 
            abs_input_shift = 9;
            max_saturation = 0x7FFF << 10;
            break;
        case ACT_BITS_24_20_TANH:
        default:
            abs_input_shift = 8;
            max_saturation = 0xFFFF << 8;
            break;
    }

    // Use the LUT for sigmoid and take into account, that
    // tanh(x) = 2*sigmoid(2*x) - 1
    int32_t input_multiplier = ((int32_t)3) << left_shift;

    int32_t input_data = (input * input_multiplier);

    uint32_t abs_input_data = input_data > 0 ? input_data : -input_data;

    uint32_t uh = abs_input_data >> abs_input_shift;

    uint32_t result;

    if (uh >= 255)
    {
        result = max_saturation;
    }
    else
    {
        uint32_t ua = sigmoid_table_uint16[uh];
        uint32_t ub = sigmoid_table_uint16[uh + 1];
        uint32_t ut;
        if (type == ACT_BITS_24_20_SIGM)
        {
            ut = abs_input_data & 0x1ff;
        }
        else
        {
            ut = abs_input_data & 0x0ff;
        }
        result = (ua << abs_input_shift) + ut * (ub - ua);
    }
    if (type == ACT_BITS_24_20_SIGM)
    {
        result = (input_data >= 0) ? (result + (1 << 9)) : ((1 << 25) - result + (1 << 9) - 1);
        result >>= 10;
    }
    else
    {
        result = (input_data >= 0) ? (result - (1 << 23)) + (1 << 7) : ((-result + (1 << 23)) + (1 << 7) - 1);
        result >>= 8;
    }
    return (int16_t)result;
}

/**
 * @} end of Acti group
 */
