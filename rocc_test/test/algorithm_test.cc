#include "fiona_utils.h"
#include "fiona_math.h"
#include <iostream>


/************************ TEST [dotprod] ***********************/
void test_mat_transpose() {
    print_sep();
    std::cout << __func__ << std::endl;

    int r = 4, c = 6;
    elem_t mat[r][c];
    for(auto i = 0; i < r; ++i) {
        for(auto j = 0; j < c; ++j) {
            mat[i][j] = i * c + j;
        }
    }

    elem_t mat_T[c][r];
    mat_transpose(&mat_T[0][0], &mat[0][0], r, c);
    printf("mat = "); print_mat(&mat[0][0], r, c);
    printf("mat.T = "); print_mat(&mat_T[0][0], c, r);
}

void test_tiled_dotprod() {
    print_sep();
    std::cout << __func__ << std::endl;

    int vlen = 19;
    elem_t vec1[vlen], vec2[vlen];
    for(auto i = 0; i < vlen; ++i) {
        vec1[i] = i;
        vec2[i] = i + 1; 
    }

    elem_t val;
    tiled_dotprod(&val, vec1, vec2, vlen);
    printf("vec1 = "); print_vec(vec1, vlen);
    printf("vec2 = "); print_vec(vec2, vlen);
    printf("result = %d\n", val);
}

void test_tiled_mvm() {
    print_sep();
    std::cout << __func__ << std::endl;

    int r = 8, vlen = 16;
    elem_t mat[r][vlen], vec[vlen];
    for(auto i = 0; i < r; ++i) {
        for(auto j = 0; j < vlen; ++j) {
            mat[i][j] = i - j;
        }
    }
    for(auto i = 0; i < vlen; ++i) {
        vec[i] = i * 2;
    }

    elem_t val[r];
    tiled_mvm(val, &mat[0][0], &vec[0], r, vlen);
    printf("mat = "); print_mat(&mat[0][0], r, vlen);
    printf("vec = "); print_vec(vec, vlen);
    printf("result = "); print_vec(val, r);
}

void test_tiled_matmul_transpose() {
    print_sep();
    std::cout << __func__ << std::endl;

    int I = 4, J = 6, K = 8;
    elem_t mat1[I][J], mat2_T[K][J];
    for(auto i = 0; i < I; ++i) {
        for(auto j = 0; j < J; ++j) {
            mat1[i][j] = i * J - j;
        }
    }
    for(auto i = 0; i < K; ++i) {
        for(auto j = 0; j < J; ++j) {
            mat2_T[i][j] = i * J + j;
        }
    }


    elem_t val[K][I];
    tiled_matmul_transpose(&val[0][0], &mat1[0][0], &mat2_T[0][0], I, J, K);
    printf("mat1 = "); print_mat(&mat1[0][0], I, J);
    printf("mat2.T = "); print_mat(&mat2_T[0][0], K, J);
    printf("result = "); print_mat(&val[0][0], K, I);
}

/************************ TEST [vector_add] ***********************/
void test_tiled_matrix_vector_add() {
    print_sep();
    std::cout << __func__ << std::endl;

    int R = 7, C = 61;
    elem_t mat[R][C], vec[C];
    for(auto i = 0; i < R; ++i) {
        for(auto j = 0; j < C; ++j) {
            mat[i][j] = i * C + j;
        }
    }
    for(auto i = 0; i < C; ++i) {
        vec[i] = i + 1;
    }


    elem_t val[R][C];
    tiled_matrix_vector_add(&val[0][0], &mat[0][0], &vec[0], R, C);
    printf("mat = "); print_mat(&mat[0][0], R, C);
    printf("vec = "); print_vec(&vec[0], C);
    printf("result = "); print_mat(&val[0][0], R, C);
}

/************************ TEST [vector_relu] ***********************/
void test_tiled_matrix_relu() {
    print_sep();
    std::cout << __func__ << std::endl;

    int R = 7, C = 61;
    elem_t mat[R][C];
    for(auto i = 0; i < R; ++i) {
        for(auto j = 0; j < C; ++j) {
            mat[i][j] = i - j;
        }
    }

    elem_t val[R][C];
    tiled_matrix_relu(&val[0][0], &mat[0][0], R, C);
    printf("mat = "); print_mat(&mat[0][0], R, C);
    printf("result = "); print_mat(&val[0][0], R, C);
}

/************************ TEST [vector_max] ***********************/
void test_tiled_matrix_vector_max() {
    print_sep();
    std::cout << __func__ << std::endl;

    int R = 7, C = 61;
    elem_t mat[R][C];
    for(auto i = 0; i < R; ++i) {
        for(auto j = 0; j < C; ++j) {
            mat[i][j] = - i + j;
        }
    }

    elem_t val[R];
    tiled_matrix_vector_max(&val[0], &mat[0][0], R, C);
    printf("mat = "); print_mat(&mat[0][0], R, C);
    printf("result = "); print_vec(&val[0], R);
}

/************************ TEST [vector_argmax] ***********************/
void test_matrix_vector_argmax() {
    print_sep();
    std::cout << __func__ << std::endl;

    int R = 7, C = 61;
    elem_t mat[R][C];
    for(auto i = 0; i < R; ++i) {
        for(auto j = 0; j < C; ++j) {
            mat[i][j] = j % i;
        }
    }

    elem_t val[R];
    matrix_vector_argmax(&val[0], &mat[0][0], R, C);
    printf("mat = "); print_mat(&mat[0][0], R, C);
    printf("result = "); print_vec(&val[0], R);
}

/************************ MAIN ***********************/
int main() {
    test_mat_transpose();
    test_tiled_dotprod();
    test_tiled_mvm();
    test_tiled_matmul_transpose();
    test_tiled_matrix_vector_add();
    test_tiled_matrix_relu();
    test_tiled_matrix_vector_max();
    test_matrix_vector_argmax();
    return 0;
}