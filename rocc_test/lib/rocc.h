// Based on code by Schuyler Eldridge. Copyright (c) Boston University
// https://github.com/seldridge/rocket-rocc-examples/blob/master/src/main/c/rocc.h

#ifndef SRC_MAIN_C_ROCC_H
#define SRC_MAIN_C_ROCC_H

#include <stdint.h>

#ifndef STR1
#define STR1(x) #x
#define STR(x) STR1(x)
#endif
#define EXTRACT(a, size, offset) (((~(~0 << size) << offset) & a) >> offset)

#define CUSTOMX_OPCODE(x) CUSTOM_ ## x
#define CUSTOM_0 0b0001011
#define CUSTOM_1 0b0101011
#define CUSTOM_2 0b1011011
#define CUSTOM_3 0b1111011

#define CUSTOMX(X, xd, xs1, xs2, rd, rs1, rs2, funct) \
  CUSTOMX_OPCODE(X)                     |             \
  (rd                 << (7))           |             \
  (xs2                << (7+5))         |             \
  (xs1                << (7+5+1))       |             \
  (xd                 << (7+5+2))       |             \
  (rs1                << (7+5+3))       |             \
  (rs2                << (7+5+3+5))     |             \
  (EXTRACT(funct, 7, 0) << (7+5+3+5+5))

// Standard macro that passes rd, rs1, and rs2 via registers
#define FIONA_ARITH_V(funct, vd, vs1, vs2) \
    ROCC_INSTRUCTION_V_V_V(0, vd, vs1, vs2, funct)

#define FIONA_ARITH_VS(funct, vd, vs1, rs2) \
    ROCC_INSTRUCTION_V_V_S(0, vd, vs1, rs2, funct, 12)

#define FIONA_ACTIVATION(funct, vd, vs1) \
    ROCC_INSTRUCTION_V_V_V(0, vd, vs1, 0, funct)

// Scalar <- Vector
#define FIONA_REDUCE_V(funct, rd, vs1) \
    ROCC_INSTRUCTION_S_V_V(0, rd, vs1, 0, funct, 10)

#define FIONA_VLD(funct, vd, rs1) \
    ROCC_INSTRUCTION_V_S(0, vd, rs1, funct, 11)

#define FIONA_VST(funct, rd, vs1, rs2) \
    ROCC_INSTRUCTION_D_V_S(0, 0, vs1, rs2, funct, 12)

// rd, rs1, and rs2 are data
// rd_n, rs_1, and rs2_n are the register numbers to use
#define ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct, rd_n, rs1_n, rs2_n) { \
    register uint64_t rd_  asm ("x" # rd_n);                                 \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;               \
    register uint64_t rs2_ asm ("x" # rs2_n) = (uint64_t) rs2;               \
    asm volatile (                                                           \
        ".word " STR(CUSTOMX(X, 1, 1, 1, rd_n, rs1_n, rs2_n, funct)) "\n\t"  \
        : "=r" (rd_)                                                         \
        : [_rs1] "r" (rs1_), [_rs2] "r" (rs2_));                             \
    rd = rd_;                                                                \
  }


// Vector <- Vector op Vector | unary Op Vector
// Example: ADD.V (Two vectors), Relu.V (Unary)
#define ROCC_INSTRUCTION_V_V_V(X, vd, vs1, vs2, funct) {                 \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 0, 0, 0, vd, vs1, vs2, funct)) "\n\t" ); \
  }

#define ROCC_INSTRUCTION_V_V_S(X, vd, vs1, rs2, funct, rs2_n) {         \
    register uint64_t rs2_ asm ("x" # rs2_n) = (uint64_t) rs2;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 0, 1, vd, vs1, rs2_n, funct)) "\n\t" \
        :: [_rs2] "r" (rs2_));                                          \
  }

// Scalar <- Reduce(Vector)
// Example: VMax, VMin
#define ROCC_INSTRUCTION_S_V(X, rd, vs1, funct, rd_n) {                  \
    register uint64_t rd_  asm ("x" # rd_n);                             \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 1, 0, 0, rd_n, vs1, 0, funct)) "\n\t"  \
        : "=r" (rd_));                                                   \
    rd = rd_;                                                            \
  }

// Scalar <- Vector op Vector
// example: dot product
#define ROCC_INSTRUCTION_S_V_V(X, rd, vs1, vs2, funct, rd_n) {           \
    register uint64_t rd_  asm ("x" # rd_n);                             \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 1, 0, 0, rd_n, vs1, vs2, funct)) "\n\t"  \
        : "=r" (rd_));                                                   \
    rd = rd_;                                                            \
  }

// Load
// vd: Dest v register
// rs1: Base address
#define ROCC_INSTRUCTION_V_S(X, vd, rs1, funct, rs1_n) {         \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 1, 0, vd, rs1_n, 0, funct)) "\n\t" \
        :: [_rs1] "r" (rs1_));                                          \
  }

// Store
// D: Dummy
// vs1: Store value
// rs2: base addr
#define ROCC_INSTRUCTION_D_V_S(X, vs1, rs2, funct, rs2_n) {         \
    register uint64_t rs2_ asm ("x" # rs2_n) = (uint64_t) rs2;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 0, 1, 0, vs1, rs2_n, funct)) "\n\t" \
        :: [_rs2] "r" (rs2_));                                          \
  }


// Scalar <- Vector op Scalar
// Currently no use case
#define ROCC_INSTRUCTION_S_S_V(X, rd, rs1, rs2, funct, rd_n, rs1_n) {     \
    register uint64_t rd_  asm ("x" # rd_n);                              \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;            \
    asm volatile (                                                        \
        ".word " STR(CUSTOMX(X, 1, 1, 0, rd_n, rs1_n, rs2, funct)) "\n\t" \
        : "=r" (rd_) : [_rs1] "r" (rs1_));                                \
    rd = rd_;                                                             \
  }


#endif  // SRC_MAIN_C_ACCUMULATOR_H
