#include <assert.h>
#include <cstdint>
#include <stdio.h>
#include <stdint.h>

#include "fiona_utils.h"
#include "fiona_nn.h"
#include "fiona_instr.h"

// use: test/Testcase-Codgen.ipynb to generate

int main() {
    // display for testbench info
    printf("------------- FIONA DNN -------------\n");
    printf("  *type: AlexNet --> in[227x227], out[10]\n");
    printf("  *dataset: cifar-10\n");

    {
        size_t batch_size = 1, in_channels = 3, out_channels = 64;
        size_t kernel_size = 11, stride = 4, padding = 2;
        size_t x_rows = 227, x_cols = 227;
        size_t y_size, y_rows, y_cols;
        conv2d_shape(&y_size, &y_rows, &y_cols, batch_size, x_rows, x_cols, in_channels, out_channels, kernel_size, stride, padding);
        elem_t y[batch_size][out_channels][y_rows][y_cols];
        elem_t k[out_channels][in_channels][kernel_size][kernel_size];
        elem_t x[batch_size][in_channels][x_rows][x_cols];

        nn_conv2d(&y[0][0][0][0], &k[0][0][0][0], &x[0][0][0][0], batch_size, x_rows, x_cols,
        in_channels, out_channels, kernel_size, stride, padding);
    }
    

    DUMP_STAT;

    return 0;
}
