#include "decode.h"
#include "processor.h"
#include "rocc.h"
#include "mmu.h"
#include "debug.h"
#include <cstdlib>
#include <math.h>
#include "fiona_opcodes.h"
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <map>
// Python bridge
#include "engine.h"

#define FIONAVLENMax 32

using std::string;
using std::map;
using std::cout;
using std::endl;

map<unsigned, string> instr_name = {
    {FUNCT_ADD_V, "add"},
    {FUNCT_SUB_V, "sub"},
    {FUNCT_ADD_VS, "add"},
    {FUNCT_SUB_VS, "sub"},
    {FUNCT_MUL_VS, "mul"},
    {FUNCT_DIV_VS, "div"},
    {FUNCT_ACTIVATION, "activation"},
    {FUNCT_VLD, "ld"},
    {FUNCT_VST, "st"},
    {FUNCT_VSHFL, "shfl"},
    {FUNCT_MINMAX, "minmax"},
    {FUNCT_CONFIG, "cfg"},
    {FUNCT_DOTP, "dotp"},
    {FUNCT_MVM, "mvm"},
    {FUNCT_DUMP, "dump"},
};
map<string, int> instr_count;

extern int16_t fiona_nn_activation_s16( int16_t type, int16_t input, uint16_t left_shift);

inline bool bit_set(uint32_t bit, int pos) {
  return (bit & (1 << pos)) != 0;
}

typedef int16_t vreg_t;
#define FOR_EACH_ELEMENT(op) {\
    for(uint32_t i = 0; i < vlen && i < 32; i++) { \
        op; \
    }\
}

#define CLEAR_REMAINING(vd) {\
    for(uint32_t i = vlen; i < 32; i++) { \
        vregs[rd_num][i] = 0; \
    }\
}

class fiona_rocc_t : public rocc_t
{
    public:
        const char* name() { return "fiona"; }

        reg_t custom0(rocc_insn_t insn, reg_t xs1, reg_t xs2)
        {
            // reg_t prev_acc = acc[insn.rs2];
            uint32_t rd_num = insn.rd;
            uint32_t rs1_num = insn.rs1;
            uint32_t rs2_num = insn.rs2;
            vreg_t result = 0;
            // Log("rd = %d, rs1 = %d, rs2 = %d", rd_num, rs1_num, rs2_num);
            // Log("xs1 = %x, xs2 = %x", xs1, xs2);

            instr_count[instr_name[insn.funct]] += 1;
            uint32_t mask_1 = vmask[rs1_num];
            uint32_t mask_2 = vmask[rs2_num];
            uint16_t masked_op1[32];
            uint16_t masked_op2[32];
            for(int i = 0; i < 32; i++) {
              masked_op1[i] = bit_set(mask_1, i) ? vregs[rs1_num][i] : 0;
              masked_op2[i] = bit_set(mask_2, i) ? vregs[rs2_num][i] : 0;
            }
            switch (insn.funct)
            {
                case FUNCT_ADD_V: FOR_EACH_ELEMENT(vregs[rd_num][i] = masked_op1[i] + masked_op2[i]); CLEAR_REMAINING(rd_num); break;
                case FUNCT_SUB_V: FOR_EACH_ELEMENT(vregs[rd_num][i] = masked_op1[i] - masked_op2[i]); CLEAR_REMAINING(rd_num); break;
                case FUNCT_ADD_VS: FOR_EACH_ELEMENT(vregs[rd_num][i] = bit_set(mask_2, i) ? (masked_op2[i] + xs1) : 0); CLEAR_REMAINING(rd_num);  break;
                case FUNCT_SUB_VS: FOR_EACH_ELEMENT(vregs[rd_num][i] = bit_set(mask_2, i) ? (masked_op2[i] - xs1) : 0); CLEAR_REMAINING(rd_num);  break;
                case FUNCT_MUL_VS: FOR_EACH_ELEMENT(vregs[rd_num][i] = bit_set(mask_2, i) ? (masked_op2[i] * xs1) : 0); CLEAR_REMAINING(rd_num);  break;
                case FUNCT_DIV_VS: FOR_EACH_ELEMENT(vregs[rd_num][i] = bit_set(mask_2, i) ? (masked_op2[i] / xs1) : 0); CLEAR_REMAINING(rd_num);  break;
                case FUNCT_VSHFL: FOR_EACH_ELEMENT(vregs[rd_num][i] = vregs[rs1_num][vregs[rs2_num][i]]); CLEAR_REMAINING(rd_num);  break;
                case FUNCT_CONFIG: fiona_config(rd_num, xs1, xs2); break;
                case FUNCT_ACTIVATION: FOR_EACH_ELEMENT(vregs[rd_num][i] = fiona_activation(rs2_num, vregs[rs1_num][i])); break;
                case FUNCT_MINMAX: 
                                       result = vregs[rs1_num][0];
                                       if(rs2_num == 0) { // Max
                                           FOR_EACH_ELEMENT( if(masked_op1[i] > result && bit_set(mask_1, i)) result = masked_op1[i] );
                                       } else if (rs2_num == 1) { // Min
                                           FOR_EACH_ELEMENT( if(masked_op1[i] < result && bit_set(mask_1, i)) result = masked_op1[i] );
                                       } else {
                                           illegal_instruction();
                                       }
                                       break;
                case FUNCT_DOTP:
                                       {
                                           // Convert the data to 32-bits to fit the physical model
                                           vreg_t* vec_0 = (vreg_t*)malloc(FIONAVLENMax * sizeof(vreg_t));
                                           vreg_t* vec_1 = (vreg_t*)malloc(FIONAVLENMax * sizeof(vreg_t));
                                           memset(vec_0, 0, FIONAVLENMax * sizeof(vreg_t));
                                           memset(vec_1, 0, FIONAVLENMax * sizeof(vreg_t));
                                           for(uint32_t i = 0; i < vlen; i++) {
                                               vec_0[i] = masked_op1[i];
                                               vec_1[i] = masked_op2[i];
                                           }
                                           vreg_t* res;
                                           array_handle("ideal_numerical", "dotp", &res, 1, 1, vec_0, FIONAVLENMax, 1, vec_1, FIONAVLENMax, 1);
                                           // Convert the data back to 16-bits
                                           result = (int16_t)*res;
                                           break;
                                       }

                case FUNCT_MVM:
                                       {
                                           // Convert the data to 32-bits to fit the physical model
                                           vreg_t* mat = (vreg_t*)malloc(FIONAVLENMax * FIONAVLENMax * sizeof(vreg_t));
                                           vreg_t* vec = (vreg_t*)malloc(FIONAVLENMax * sizeof(vreg_t));
                                           memset(mat, 0, FIONAVLENMax * FIONAVLENMax * sizeof(vreg_t));
                                           memset(vec, 0, FIONAVLENMax * sizeof(vreg_t));
                                           for(uint32_t i = 0; i < vlen; i++) {
                                               vec[i] = masked_op1[i];
                                               for(uint32_t j = 0; j < vlen; j++) {
                                                   mat[i * FIONAVLENMax + j] = matrix[i][j];
                                               }
                                           }
                                           vreg_t *res;
                                           array_handle("ideal_numerical", "mvm", &res, FIONAVLENMax, 1, vec, FIONAVLENMax, 1, mat, FIONAVLENMax, FIONAVLENMax);
                                           // array_handle("ideal_numerical", "mvm", &res, FIONAVLENMax, 1, mat, FIONAVLENMax, FIONAVLENMax, vec, FIONAVLENMax, 1);
                                           // Convert the data back to 16-bits
                                           for(uint32_t i = 0; i < vlen; i++) {
                                               vregs[rd_num][i] = res[i];
                                           }
                                           break;
                                       }
                case FUNCT_VLD: // Load rs1=base, vd=vec
                                  {
                                      // Log("Load");
                                      reg_t ptr = xs1;
                                      for(uint32_t i = 0; i < vlen && i < 32; i++) {
                                          int16_t load_val = p->get_mmu()->load<int16_t>(ptr);
                                          // printf("Load from %p\n", ptr);
                                          // printf("Value = %x\n", load_val);
                                          vregs[rd_num][i] = load_val;
                                          ptr = ptr + stride * sizeof(vreg_t);
                                      }
                                      break;
                                  }
                case FUNCT_VST: // Store rs1=base, vs2=vec
                                  {
                                      // Log("Store\n");
                                      reg_t ptr = xs1;
                                      for(uint32_t i = 0; i < vlen && i < 32; i++) {
                                          vreg_t store_val = vregs[rs2_num][i];
                                          // Log("Store to %p", ptr);
                                          // Log("Value = %x", store_val);
                                          p->get_mmu()->store<vreg_t>(ptr, store_val);
                                          ptr = ptr + stride * sizeof(vreg_t);
                                      }
                                      break;
                                  }


                case FUNCT_DUMP: printf("DUMP"); dump(); break;
                default:
                                  printf("UnImp Opcode %d\n", insn.funct);
                                  break;
            }

            return result; // in all cases, xd <- previous value of acc[rs2]
        }

        inline vreg_t fiona_activation(uint32_t function_code, vreg_t x)   // TODO: We need quantitize
        {
            vreg_t result = 0;
            int16_t signed_x = x;
            switch (function_code) {
                case ACT_BITS_24_20_RELU: result = signed_x > 0 ? signed_x : 0; break;
                case ACT_BITS_24_20_TANH: illegal_instruction();
                case ACT_BITS_24_20_SIGM: illegal_instruction();
            }
            return (int16_t)result;
        }
        void fiona_config(uint32_t config_reg, reg_t rs1, reg_t rs2) 
        {
            switch (config_reg) {
                case 0:    // VLEN
                    if(rs1 > 32) illegal_instruction();
                    vlen = rs1;
                    break;
                case 1:   // VMASK
                    vmask[rs2] = rs1;
                    break;
                case 2:   // VMatrix
                    {
                        reg_t ptr = rs1;
                        for(uint32_t i = 0; i < vlen && i < 32; i++) {
                            for(uint32_t j = 0; j < vlen && j < 32; j++) {
                                int16_t load_val = p->get_mmu()->load<int16_t>(ptr);
                                matrix[i][j] = load_val;
                                ptr = ptr + stride * sizeof(vreg_t);
                            }
                        }
                    }
                    break;
                default:
                    printf("UnImp config reg!\n");
                    illegal_instruction();
            }
        }
        fiona_rocc_t()
        {
            // memset(acc, 0, sizeof(acc));
            vlen = 32;
            stride = 1;
            init_python_env();
            for(int i = 0; i < 32; i++) {
              vmask[i] = 0xffffffff;
            }
        }
        void show_reg_state() 
        {
            for(int i = 0; i < 32; i++) {
                printf("[%d] ", i);
                for(int j = 0; j < 32; j++) {
                    printf("%d ", vregs[i][j] );
                }
                printf("\n");
            }
        }
        void dump() {
            for(auto x: instr_count)
            {
                cout << x.first << "->" <<
                    x.second << endl;
            }
        }

    private:
        vreg_t vregs[32][32];
        vreg_t matrix[32][32];
        uint32_t vmask[32];
        uint32_t vlen;
        uint32_t stride;
};

REGISTER_EXTENSION(fiona, []() { return new fiona_rocc_t; })
