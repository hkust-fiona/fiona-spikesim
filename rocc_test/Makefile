CFLAG = -I cfg/ -I lib/
ELF_FILES = fiona_nn_mlp.cc fiona_nn_conv.cc fiona_nn_pad.cc

all: elf test disasm

run: elf
	spike --extension=fiona pk bin/nn_mlp_iris

elf: ${ELF_FILES}
	riscv64-unknown-elf-g++ ${CFLAG} fiona_nn_mlp.cc -o bin/nn_mlp_iris --static
	riscv64-unknown-elf-g++ ${CFLAG} fiona_nn_conv.cc -o bin/nn_conv_gen --static
	riscv64-unknown-elf-g++ ${CFLAG} fiona_nn_pad.cc -o bin/nn_pad --static
	riscv64-unknown-elf-g++ ${CFLAG} fiona_nn_pool.cc -o bin/nn_pool --static

test: test/algorithm_test.cc
	riscv64-unknown-elf-g++ ${CFLAG} test/algorithm_test.cc -o bin/algorithm_test --static

disasm: fiona_nn_mlp.cc
	riscv64-unknown-elf-g++ ${CFLAG} fiona_nn_mlp.cc -o mlp_iris_dylib
	riscv64-unknown-elf-objdump -S mlp_iris_dylib -M no-aliases,numeric > mlp_iris_dylib.S
	rm -rf mlp_iris_dylib

clean:
	rm -rf bin/algorithm_test bin/nn_mlp_iris bin/nn_conv_gen mlp_iris_dylib.S
