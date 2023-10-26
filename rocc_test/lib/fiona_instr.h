#ifndef __FIONA_INSTR_H__
#define __FIONA_INSTR_H__
#define STR1(s) #s
#define STR(s) STR1(s)

#define ADD_V(vd, vs, vt)        { asm volatile ("add.v.fiona "  STR(vd) "," STR(vs) "," STR(vt)); }
#define SUB_V(vd, vs, vt)        { asm volatile ("sub.v.fiona "  STR(vd) "," STR(vs) "," STR(vt)); }
#define VSHFL(vd, vs, vt)        { asm volatile ("vshfl.fiona "  STR(vd) "," STR(vs) "," STR(vt)); }

#define ADD_VS(vd, vs, rt)       { asm volatile ("add.vs.fiona "  STR(vd) "," STR(vs) ", %0" : : "r"(rt)); }
#define SUB_VS(vd, vs, rt)       { asm volatile ("sub.vs.fiona "  STR(vd) "," STR(vs) ", %0" : : "r"(rt)); }
#define DIV_VS(vd, vs, rt)       { asm volatile ("div.vs.fiona "  STR(vd) "," STR(vs) ", %0" : : "r"(rt)); }
#define MUL_VS(vd, vs, rt)       { asm volatile ("mul.vs.fiona "  STR(vd) "," STR(vs) ", %0" : : "r"(rt)); }

#define VRELU(vd, vs)            { asm volatile ("vrelu.fiona "  STR(vd) "," STR(vs)); }
#define VTANH(vd, vs)            { asm volatile ("vtanh.fiona "  STR(vd) "," STR(vs)); }
#define VSIGM(vd, vs)            { asm volatile ("vsigm.fiona "  STR(vd) "," STR(vs)); }

#define SET_VLEN(vlen)           { asm volatile ("config.fiona "  "x0,%0,%0"    : : "r"(vlen)); }
#define SET_VMASK(vregnum, mask) { asm volatile ("config.fiona "  "x1,%0,%1" : : "r"(vregnum), "r"(mask)); }
#define SET_MAT(mat_addr)        { asm volatile ("config.fiona "  "x2,%0,%0"    : : "r"(mat_addr)); }
#define SET_STRIDE(val)           { asm volatile ("config.fiona "  "x3,%0,%0"    : : "r"(val)); }

#define VLD(vreg, addr)          { asm volatile ("vld.fiona "  STR(vreg) " ,%0" : : "r"(addr)); }
#define VST(vreg, addr)          { asm volatile ("vst.fiona "  "%0, " STR(vreg) : : "r"(addr)); }

#define VMAX(rd, vs)             { asm volatile ("vmax.fiona "  "%0 ," STR(vs) : "=r"(rd) :); }
#define VMIN(rd, vs)             { asm volatile ("vmin.fiona "  "%0, " STR(vs) : "=r"(rd) :); }

#define DOTP(rd, vs, vt)         { asm volatile ("dotp.fiona "  "%0 ," STR(vs) "," STR(vt) : "=r"(rd) :); }
#define MVM(vd, vs)              { asm volatile ("mvm.fiona "  STR(vd) "," STR(vs)); }

#include "rocc.h"
#define DUMP_STAT                ROCC_INSTRUCTION_V_V_V(0, 0, 0, 0, 15);

#endif
