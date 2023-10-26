#ifndef ROCC_TEST_FIONA_MATH
#define ROCC_TEST_FIONA_MATH

#include "fiona_utils.h"

/************************************************ Basic Math Operations ***********************************************/

//! >>> pALU: dotprod
/***********************************************************************/
static void fit_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen=PA_MRR_DGRP) {
    assert(vlen <= PA_MRR_DGRP);
    SET_VLEN(vlen);
    VLD(x1, vec1);  // @x1: [v]
    VLD(x2, vec2);  // @x2: [v]
    DOTP(*retval, x1, x2);   // @retval: [s]
}

static void tiled_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    if(vlen <= PA_MRR_DGRP) {
        fit_dotprod(retval, vec1, vec2, vlen);
    } else {
        size_t remainder = vlen % PA_MRR_DGRP;
        size_t loop_num = vlen / PA_MRR_DGRP;
        elem_t sum = 0, val = 0;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * PA_MRR_DGRP;
            fit_dotprod(&val, &vec1[index], &vec2[index], PA_MRR_DGRP);
            sum += val;
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_dotprod(&val, &vec1[offset], &vec2[offset], remainder);
            sum += val;
        }
        *retval = sum;
    }
}

static void tiled_mvm(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_mvm().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_dotprod(&retval[i], &mat[i * cols], &vec[0], cols);
    }
}

static void tiled_matmul_transpose(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T, size_t I, size_t J, size_t K) {
    // @mat1: size I*J
    // @mat2: size J*K --> mat2_T: size K*J
    // @retval: size K*I (column-major)
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matmul_transpose().\n");
        printf("[HINT] elem_t retval[K=%d][I=%d];\n", K, I);
        exit(-1);
    }
    for(size_t k = 0; k < K; ++k) {
        tiled_mvm(&retval[k * I], &mat1[0], &mat2_T[k * J], I, J);
    }
}

//! >>> eALU: add
/***********************************************************************/
static void fit_vector_add(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec1);
    VLD(x2, vec2);
    ADD_V(x3, x1, x2);
    VST(x3, retval);
}

static void tiled_vector_add(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_add().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_add(retval, vec1, vec2, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_add(&retval[index], &vec1[index], &vec2[index], EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_add(&retval[offset], &vec1[offset], &vec2[offset], remainder);
        }
    }
}

static void tiled_matrix_vector_add(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t vlen) {
    // >>> [add vec to each row of mat] <<<
    // @mat: rows * vlen
    // @vec: vlen
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_vector_add().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_add(&retval[i * vlen], &mat[i * vlen], &vec[0], vlen);
    }
}

//! >>> eALU: sub
/***********************************************************************/
static void fit_vector_sub(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec1);
    VLD(x2, vec2);
    SUB_V(x3, x1, x2);
    VST(x3, retval);
}

static void tiled_vector_sub(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_sub().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_sub(retval, vec1, vec2, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_sub(&retval[index], &vec1[index], &vec2[index], EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_sub(&retval[offset], &vec1[offset], &vec2[offset], remainder);
        }
    }
}

static void tiled_matrix_vector_sub(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t vlen) {
    // >>> [add vec to each row of mat] <<<
    // @mat: rows * vlen
    // @vec: vlen
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_vector_sub().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_sub(&retval[i * vlen], &mat[i * vlen], &vec[0], vlen);
    }
}

//! >>> eALU: add scalar
/***********************************************************************/
static void fit_vector_add_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    ADD_VS(x2, x1, scalar);
    VST(x3, retval);
}

static void tiled_vector_add_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_add_scalar().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_add_scalar(retval, vec, scalar, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_add_scalar(&retval[index], &vec[index], scalar, EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_add_scalar(&retval[offset], &vec[offset], scalar, remainder);
        }
    }
}

static void tiled_matrix_add_scalar(elem_t *retval, const elem_t *mat, const elem_t scalar, size_t rows, size_t vlen) {
    // >>> [add scalar to each element of mat] <<<
    // @mat: rows * vlen
    // @scalar: 1
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_add_scalar().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_add_scalar(&retval[i * vlen], &mat[i * vlen], scalar, vlen);
    }
}

//! >>> eALU: sub scalar
/***********************************************************************/
static void fit_vector_sub_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    SUB_VS(x2, x1, scalar);
    VST(x3, retval);
}

static void tiled_vector_sub_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_sub_scalar().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_sub_scalar(retval, vec, scalar, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_sub_scalar(&retval[index], &vec[index], scalar, EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_sub_scalar(&retval[offset], &vec[offset], scalar, remainder);
        }
    }
}

static void tiled_matrix_sub_scalar(elem_t *retval, const elem_t *mat, const elem_t scalar, size_t rows, size_t vlen) {
    // >>> [sub scalar to each element of mat] <<<
    // @mat: rows * vlen
    // @scalar: 1
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_sub_scalar().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_sub_scalar(&retval[i * vlen], &mat[i * vlen], scalar, vlen);
    }
}

//! >>> eALU: multiply scalar
/***********************************************************************/
static void fit_vector_mul_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    MUL_VS(x2, x1, scalar);
    VST(x3, retval);
}

static void tiled_vector_mul_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_mul_scalar().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_mul_scalar(retval, vec, scalar, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_mul_scalar(&retval[index], &vec[index], scalar, EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_mul_scalar(&retval[offset], &vec[offset], scalar, remainder);
        }
    }
}

static void tiled_matrix_mul_scalar(elem_t *retval, const elem_t *mat, const elem_t scalar, size_t rows, size_t vlen) {
    // >>> [multiply scalar to each element of mat] <<<
    // @mat: rows * vlen
    // @scalar: 1
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_mul_scalar().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_mul_scalar(&retval[i * vlen], &mat[i * vlen], scalar, vlen);
    }
}

//! >>> eALU: divide scalar
/***********************************************************************/
static void fit_vector_div_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    DIV_VS(x2, x1, scalar);
    VST(x3, retval);
}

static void tiled_vector_div_scalar(elem_t *retval, const elem_t *vec, const elem_t scalar, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_div_scalar().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_div_scalar(retval, vec, scalar, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_div_scalar(&retval[index], &vec[index], scalar, EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_div_scalar(&retval[offset], &vec[offset], scalar, remainder);
        }
    }
}

static void tiled_matrix_div_scalar(elem_t *retval, const elem_t *mat, const elem_t scalar, size_t rows, size_t vlen) {
    // >>> [divide scalar to each element of mat] <<<
    // @mat: rows * vlen
    // @scalar: 1
    // @retval: rows * vlen
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_div_scalar().\n");
        printf("[HINT] elem_t retval[rows=%d][vlen=%d];\n", rows, vlen);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_div_scalar(&retval[i * vlen], &mat[i * vlen], scalar, vlen);
    }
}


//! >>> eNLU: relu
/***********************************************************************/
static void fit_vector_relu(elem_t *retval, const elem_t *vec, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    VRELU(x2, x1);
    VST(x2, retval);
}

static void tiled_vector_relu(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_relu().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_relu(retval, vec, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_relu(&retval[index], &vec[index], EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_relu(&retval[offset], &vec[offset], remainder);
        }
    }
}

static void tiled_matrix_relu(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_relu().\n");
        printf("[HINT] elem_t retval[rows=%d][cols=%d];\n", rows, cols);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_relu(&retval[i * cols], &mat[i * cols], cols);
    }
}

//! >>> eNLU: tanh
/***********************************************************************/
static void fit_vector_tanh(elem_t *retval, const elem_t *vec, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    VTANH(x2, x1);
    VST(x2, retval);
}

static void tiled_vector_tanh(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_tanh().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_tanh(retval, vec, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_tanh(&retval[index], &vec[index], EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_tanh(&retval[offset], &vec[offset], remainder);
        }
    }
}

static void tiled_matrix_tanh(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_tanh().\n");
        printf("[HINT] elem_t retval[rows=%d][cols=%d];\n", rows, cols);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_tanh(&retval[i * cols], &mat[i * cols], cols);
    }
}

//! >>> eNLU: sigmoid
/***********************************************************************/
static void fit_vector_sigmoid(elem_t *retval, const elem_t *vec, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    VSIGM(x2, x1);
    VST(x2, retval);
}

static void tiled_vector_sigmoid(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_sigmoid().\n");
        printf("[HINT] elem_t retval[vlen=%d];\n", vlen);
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_sigmoid(retval, vec, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_sigmoid(&retval[index], &vec[index], EU_VEC_ELEM);
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_sigmoid(&retval[offset], &vec[offset], remainder);
        }
    }
}

static void tiled_matrix_sigmoid(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_sigmoid().\n");
        printf("[HINT] elem_t retval[rows=%d][cols=%d];\n", rows, cols);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_sigmoid(&retval[i * cols], &mat[i * cols], cols);
    }
}

//! >>> eMISC: max
/***********************************************************************/
static void fit_vector_max(elem_t *retval, const elem_t *vec, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    VMAX(*retval, x1);
}

static void tiled_vector_max(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_max().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_max(retval, vec, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        elem_t val, maxval = elem_t_min;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_max(&val, &vec[index], EU_VEC_ELEM);
            if(val > maxval) maxval = val;
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_max(&val, &vec[offset], remainder);
            if(val > maxval) maxval = val;
        }
        *retval = maxval;
    }
}

static void tiled_matrix_vector_max(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_vector_max().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_max(&retval[i], &mat[i * cols], cols);
    }
}

//! >>> eMISC: min
/***********************************************************************/
static void fit_vector_min(elem_t *retval, const elem_t *vec, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    SET_VLEN(vlen);
    VLD(x1, vec);
    VMIN(*retval, x1);
}

static void tiled_vector_min(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_vector_min().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    if(vlen <= EU_VEC_ELEM) {
        fit_vector_min(retval, vec, vlen);
    } else {
        size_t remainder = vlen % EU_VEC_ELEM;
        size_t loop_num = vlen / EU_VEC_ELEM;
        elem_t val, minval = elem_t_max;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * EU_VEC_ELEM;
            fit_vector_min(&val, &vec[index], EU_VEC_ELEM);
            if(val < minval) minval = val;
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_vector_min(&val, &vec[offset], remainder);
            if(val < minval) minval = val;
        }
        *retval = minval;
    }
}

static void tiled_matrix_vector_min(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matrix_vector_min().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        tiled_vector_min(&retval[i], &mat[i * cols], cols);
    }
}

//! >>> helper: dropout
/***********************************************************************/
static void fit_vector_dropout(elem_t *retval, const elem_t *vec, const elem_t mask, size_t vlen=EU_VEC_ELEM) {
    assert(vlen <= EU_VEC_ELEM);
    VLD(x11, vec);
    SET_VLEN(vlen);
    SET_VMASK(11, mask);
    ADD_V(x11, x11, x0);
    SET_VMASK(11, -1);
}

//! >>> helper: argmax
/***********************************************************************/
static void vector_argmax(elem_t *retval, const elem_t *vec, size_t vlen) {
    elem_t maxval;
    tiled_vector_max(&maxval, vec, vlen);
    size_t i = 0;
    for(i = 0; i < vlen; ++i) {
        if(maxval == vec[i]) break;
    }
    *retval = i;
}

static void matrix_vector_argmax(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: matrix_vector_argmax().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        vector_argmax(&retval[i], &mat[i * cols], cols);
    }
}

//! >>> helper: reshape
/***********************************************************************/
static void vector_shift(elem_t *vec, size_t vlen, size_t bit_shift, bool direction=1) {
    // @direction: 0 - left shift, 1 - right shift
    if(bit_shift == 0) return;
    if(vec == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vec_shift().\n");
        printf("[HINT] elem_t vec[rows=%d];\n", vlen);
        exit(-1);
    }
    if(direction) { for(size_t i = 0; i < vlen; ++i) { vec[i] >>= bit_shift; } } 
    else { for(size_t i = 0; i < vlen; ++i) { vec[i] <<= bit_shift; } }
}

static void mat_transpose(elem_t *mat_T, const elem_t *mat, size_t rows, size_t cols) {
    // row-major storage
    if(mat_T == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: mat_transpose().\n");
        printf("[HINT] elem_t mat_T[rows=%d][cols=%d];\n", cols, rows);
        exit(-1);
    }
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            auto index = i * cols + j;
            auto index_T = j * rows + i;
            mat_T[index_T] = mat[index];
        }
    }
}


/************************************************ Wrapped Operations ***********************************************/

/************************ VECTOR ***********************/
static void vector_equal(elem_t *ret_bool, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    if(ret_bool == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vector_equal().\n");
        printf("[HINT] elem_t ret_bool[vlen=%d];\n", vlen);
        exit(-1);
    }
    for(size_t i = 0; i < vlen; ++i) {
        if(vec1[i] == vec2[i]) {
            ret_bool[i] = 1;
        } else {
            ret_bool[i] = 0;
        }
    }
}

static void vector_sum(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vector_sum().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0;
    for(size_t i = 0; i < vlen; ++i) {
        sum += vec[i];
    }
    *retval = sum;
}

static void vector_mean(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vector_mean().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0;
    vector_sum(&sum, vec, vlen);
    *retval = sum / vlen;
}

static void vector_var(elem_t *retval, const elem_t *vec, size_t vlen) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vector_var().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t mean = 0;
    elem_t *tmp = new elem_t[vlen];
    vector_mean(&mean, vec, vlen);
    tiled_vector_sub_scalar(tmp, vec, mean, vlen);
    tiled_dotprod(retval, tmp, tmp, vlen);
}

static void vector_softmax(elem_t *retval, const elem_t *vec, size_t vlen, elem_t factor=100) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: vector_softmax().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t maxval;
    tiled_vector_max(&maxval, vec, vlen);
    tiled_vector_sub_scalar(retval, vec, maxval, vlen);
    double exp_sum = 0.0;
    for(size_t i = 0; i < vlen; ++i) {
        exp_sum += exp(retval[i]);
    }
    for(size_t i = 0; i < vlen; ++i) {
        retval[i] = retval[i] * factor / exp_sum;
    }
}

/************************ MATRIX ***********************/
static void matrix_sum(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: matrix_sum().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0, tmp = 0;
    for(size_t i = 0; i < rows; ++i) {
        vector_sum(&tmp, &mat[rows * i], cols);
        sum += tmp;
    }
    *retval = sum;
}

static void matrix_sum(elem_t *retval, const elem_t *mat, size_t rows, size_t cols, size_t channel_size) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: matrix_sum().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0, tmp = 0;
    for(size_t i = 0; i < channel_size; ++i) {
        matrix_sum(&tmp, &mat[channel_size * i], rows, cols);
        sum += tmp;
    }
    *retval = sum;
}

static void matrix_mean(elem_t *retval, const elem_t *mat, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: matrix_mean().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0;
    matrix_sum(&sum, mat, rows, cols);
    *retval = (elem_t)(sum / (rows * cols));
}

static void matrix_mean(elem_t *retval, const elem_t *mat, size_t rows, size_t cols, size_t channel_size) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: matrix_mean().\n");
        printf("[HINT] elem_t retval;\n");
        exit(-1);
    }
    elem_t sum = 0;
    matrix_sum(&sum, mat, rows, cols, channel_size);
    *retval = (elem_t)(sum / (rows * cols * channel_size));
}

#endif /* ROCC_TEST_FIONA_MATH */