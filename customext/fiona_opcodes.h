#ifndef __FIONA_OPCODE_H__
#define __FIONA_OPCODE_H__

#define FUNCT_ADD_V	1
#define FUNCT_SUB_V	2
#define FUNCT_ADD_VS	3
#define FUNCT_SUB_VS	4
#define FUNCT_MUL_VS	5
#define FUNCT_DIV_VS	6

#define FUNCT_ACTIVATION 7
#define ACT_BITS_24_20_RELU 0
#define ACT_BITS_24_20_TANH 1
#define ACT_BITS_24_20_SIGM 2

#define FUNCT_VLD	8
#define FUNCT_VST	9

#define FUNCT_VSHFL	10
#define FUNCT_MINMAX	11

#define FUNCT_CONFIG	12

#define FUNCT_DOTP	13
#define FUNCT_MVM	14
#define FUNCT_DUMP	15

#endif
