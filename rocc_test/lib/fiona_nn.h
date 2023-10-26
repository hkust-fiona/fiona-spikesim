#ifndef ROCC_TEST_FIONA_NN_MODULES
#define ROCC_TEST_FIONA_NN_MODULES

#include "fiona_math.h"

/******************** Linear ********************/
static void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, size_t feature_in, size_t feature_out, size_t batch_size) {
    // @w: feature_out * feature_in
    // @x: batch_size * feature_in
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear().\n");
        printf("[HINT] elem_t y[batch_size=%d][feature_out=%d];\n", batch_size, feature_out);
        exit(-1);
    }
    tiled_matmul_transpose(y, w, x, feature_out, feature_in, batch_size);
}

static void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, const elem_t *b, size_t feature_in, size_t feature_out, size_t batch_size) {
    // @w: feature_out * feature_in
    // @x: batch_size * feature_in
    // @b: feature_out
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear().\n");
        printf("[HINT] elem_t y[batch_size=%d][feature_out=%d];\n", batch_size, feature_out);
        exit(-1);
    }
    tiled_matmul_transpose(y, w, x, feature_out, feature_in, batch_size);
    tiled_matrix_vector_add(y, y, b, batch_size, feature_out);
}

/******************** Conv2d ********************/
static void nn_conv2d(elem_t *y, const elem_t *k, const elem_t *_x, size_t batch_size, size_t x_rows, size_t x_cols,
    size_t channel_in, size_t channel_out, size_t kernel_size, size_t stride=1, size_t padding=0) {
    // @k: (kernel_size * kernel_size) * (channel_in * channel_out)
    // @x: (x_cols * x_rows) * channel_in * batch_size
    // @y: (y_cols * y_rows) * channel_out * batch_size
    assert(stride > 0);
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_conv2d().\n");
        printf("[HINT] elem_t y[batch_size][out_channels][y_rows][y_cols];\n");
        printf("[HINT] --> generate using conv2d_shape() in <fiona_utils.h>\n");
        exit(-1);
    }
    size_t padded_size, padded_rows, padded_cols;
    padding2d_shape(&padded_size, &padded_rows, &padded_cols, padding, x_rows, x_cols, channel_in, batch_size);
    elem_t x[padded_size];
    padding2d(x, _x, padding, 0, x_rows, x_cols, channel_in, batch_size);

    size_t y_rows = size_t((x_rows + 2 * padding - (kernel_size - 1) - 1) / stride) + 1;
    size_t y_cols = size_t((x_cols + 2 * padding - (kernel_size - 1) - 1) / stride) + 1;
    
    const size_t kelem_per_chin = kernel_size * kernel_size;
    const size_t kelem_per_chout = kelem_per_chin * channel_in;
    const size_t xelem_per_channel = x_cols * x_rows;
    const size_t xelem_per_batch = xelem_per_channel * channel_in;
    const size_t yelem_per_channel = y_cols * y_rows;
    const size_t yelem_per_batch = yelem_per_channel * channel_out;

    for(size_t l = 0; l < batch_size; ++l) {
        size_t xoffset_batch = l * xelem_per_batch;
        size_t yoffset_batch = l * yelem_per_batch;

        for(size_t m = 0; m < channel_out; ++m) {
            size_t koffset_chout = m * kelem_per_chout;
            size_t yoffset_chout = m * yelem_per_channel;
            for(size_t n = 0; n < channel_in; ++n) {
                size_t koffset_chin = n * kelem_per_chin;
                size_t xoffset_chin = n * xelem_per_channel;

                for(size_t r = 0; r < y_rows; ++r){
                    for(size_t c = 0; c < y_cols; ++c) {
                        size_t xoffset_pixel = (r * x_cols + c) * stride;
                        size_t yoffset_pixel = r * y_cols + c;

                        size_t y_index = yoffset_batch + yoffset_chout + yoffset_pixel;
                        
                        // DEBUG: printf("----------| in pixel: y_rows = %d, y_cols = %d |-------------\n", r, c);
                        for(size_t s = 0; s < kernel_size; ++s) {
                            size_t k_index = koffset_chout + koffset_chin + kernel_size * s;
                            size_t x_index = xoffset_batch + xoffset_chin + xoffset_pixel + x_cols * s;

                            elem_t val;
                            tiled_dotprod(&val, &k[k_index], &x[x_index], kernel_size);
                            y[y_index] += val;
                            // DEBUG: printf(">>> [conv] \n\tk: "); print_vec(&k[k_index], kernel_size);
                            // DEBUG: printf("\tx: "); print_vec(&x[x_index], kernel_size);
                        }
                    }
                }

            }
        }

    }
}

/******************** BatchNorm2d ********************/
struct BatchNormParam {
    elem_t *weight;
    elem_t *bias;
    elem_t *running_mean;
    elem_t *running_var;
    float eps;
    float momentum;

    BatchNormParam(size_t channel_size) {
        this->eps = 1e-5;
        this->momentum = 0.1;

        this->weight = new elem_t[channel_size];
        this->bias = new elem_t[channel_size];
        this->running_mean = new elem_t[channel_size];
        this->running_var = new elem_t[channel_size];
    }
};

static void nn_batchnorm2d(elem_t *y, const elem_t *x, size_t batch_size, size_t channel_size, size_t rows, size_t cols,
    BatchNormParam& bn_param, bool stage=0) {
    // @x: batch_size * channel_size * rows * cols
    // @bn_param: channel_size
    // @stage: 0 - inference, 1 - training
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_batchnorm2d().\n");
        printf("[HINT] elem_t y[batch_size=%d][channel_size=%d][rows=%d][cols=%d];\n", batch_size, channel_size, rows, cols);
        exit(-1);
    }
    const size_t pixels_per_channel = rows * cols;
    const size_t pixels_per_batch = channel_size * pixels_per_channel;
    if(stage == 0) {
        for(size_t b = 0; b < batch_size; ++b) {
            size_t batch_index = b * pixels_per_batch;
            for(size_t c = 0; c < channel_size; ++c) {
                size_t offset = batch_index + c * pixels_per_channel; 
                elem_t running_std = sqrt(bn_param.running_var[c] + bn_param.eps); 
                tiled_matrix_sub_scalar(&y[offset], &x[offset], bn_param.running_mean[c], rows, cols);
                tiled_matrix_div_scalar(&y[offset], &y[offset], running_std, rows, cols);
                tiled_matrix_mul_scalar(&y[offset], &y[offset], bn_param.weight[c], rows, cols);
                tiled_matrix_add_scalar(&y[offset], &y[offset], bn_param.bias[c], rows, cols);
            }
        }
    } else if(stage == 1) {
        printf("[ERROR] nn_batchnorm2d yet supports training.");
        exit(-1);
    }
}

/******************** LayerNorm2d ********************/
struct LayerNormParam {
    elem_t *gamma;
    elem_t *beta;
    float eps;

    LayerNormParam(size_t batch_size) {
        this->eps = 1e-10;

        this->gamma = new elem_t[batch_size];
        this->beta = new elem_t[batch_size];
    }
};

static void nn_layernorm2d(elem_t *y, const elem_t *x, size_t batch_size, size_t channel_size, size_t rows, size_t cols,
    LayerNormParam& ln_param, bool stage=0) {
    // @x: batch_size * channel_size * rows * cols
    // @ln_param: batch_size
    // @stage: 0 - inference, 1 - training
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_layernorm2d().\n");
        printf("[HINT] elem_t y[batch_size=%d][channel_size=%d][rows=%d][cols=%d];\n", batch_size, channel_size, rows, cols);
        exit(-1);
    }
    const size_t pixels_per_channel = rows * cols;
    const size_t pixels_per_batch = channel_size * pixels_per_channel;
    if(stage == 0) {
        for(size_t b = 0; b < batch_size; ++b) {
            size_t batch_index = b * pixels_per_batch;
            elem_t layer_mean, layer_var;
            matrix_mean(&layer_mean, &x[batch_index], rows, cols, channel_size);
            vector_var(&layer_var, &x[batch_index], pixels_per_batch);
            elem_t layer_std = sqrt(layer_var + ln_param.eps);
            for(size_t c = 0; c < channel_size; ++c) {
                size_t offset = batch_index + c * pixels_per_channel; 
                tiled_matrix_sub_scalar(&y[offset], &x[offset], layer_mean, rows, cols);
                tiled_matrix_div_scalar(&y[offset], &y[offset], layer_std, rows, cols);
                tiled_matrix_mul_scalar(&y[offset], &y[offset], ln_param.gamma[b], rows, cols);
                tiled_matrix_add_scalar(&y[offset], &y[offset], ln_param.beta[b], rows, cols);
            }
        }
    } else if(stage == 1) {
        printf("[ERROR] nn_layernorm2d yet supports training.");
        exit(-1);
    }
}

/******************** MaxPooling ********************/
static void nn_maxpooling2d(elem_t *y, const elem_t *_x, size_t rows, size_t cols, size_t channel_size, size_t batch_size,
    size_t kernel_size, size_t padding=0, size_t stride=0) {
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_maxpooling2d().\n");
        printf("[HINT] elem_t y[batch_size][out_channels][y_rows][y_cols];\n");
        printf("[HINT] --> generate using pooling2d_shape() in <fiona_utils.h>\n");
        exit(-1);
    }
    // stride
    assert(kernel_size > 0);
    if(stride == 0) stride = kernel_size;

    // padding
    size_t padded_size, padded_rows, padded_cols;
    padding2d_shape(&padded_size, &padded_rows, &padded_cols, padding, rows, cols, channel_size, batch_size);
    elem_t *x = new elem_t[padded_size];
    padding2d(x, _x, padding, elem_t_min, rows, cols, channel_size, batch_size);

    // pooling
    size_t pooled_sized, pooled_rows, pooled_cols;
    pooling2d_shape(&pooled_sized, &pooled_rows, &pooled_cols, rows, cols, channel_size, batch_size, kernel_size, padding, stride);

    const size_t yelem_per_channel = pooled_rows * pooled_cols;
    const size_t yelem_per_batch = channel_size * yelem_per_channel;
    const size_t xelem_per_channel = padded_rows * padded_cols;
    const size_t xelem_per_batch = channel_size * xelem_per_channel;
    for(size_t b = 0; b < batch_size; ++b) {
        size_t yoffset_batch = b * yelem_per_batch;
        size_t xoffset_batch = b * xelem_per_batch;
        for(size_t c = 0; c < channel_size; ++c) {
            size_t yoffset_channel = c * yelem_per_channel;
            size_t xoffset_channel = c * xelem_per_channel;

            for(size_t i = 0; i < pooled_rows; ++i) {
                for(size_t j = 0; j < pooled_cols; ++j) {
                    elem_t maxval = elem_t_min;
                    size_t yoffset_pixel = yoffset_batch + yoffset_channel + i * pooled_cols + j;
                    size_t xoffset_pixel = xoffset_batch + xoffset_channel + (i * padded_cols + j) * stride;

                    for(size_t p = 0; p < kernel_size; ++p) {
                        for(size_t q = 0; q < kernel_size; ++q) {
                            size_t offset_x = xoffset_pixel + p * padded_cols + q;
                            if(maxval < x[offset_x]) {
                                maxval = x[offset_x];
                            }
                        }
                    }
                    y[yoffset_pixel] = maxval;
                }
            }

        }
    }
}

/******************** Transformer ********************/
static void nn_scaled_dotprod_attention(elem_t *out, elem_t *attn, const elem_t *q, const elem_t *k, const elem_t *v) {
    //TODO:
    /*
    tiled_matmul_transpose(attn, q / temperature, k_T, I, J, K);
    vector_softmax(attn, dim=-1);
    tiled_matmul_transpose(out, attn, v);
    */
}



#endif /* ROCC_TEST_FIONA_NN_MODULES */