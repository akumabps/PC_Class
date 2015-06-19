all:
	gcc ./openMP/p2-matrixMult_cpu-omp.c -fopenmp -o ./bin/p2-matrixMult_cpu-omp
	gcc ./openMP/false_sharing-omp.c -fopenmp -o ./bin/false_sharing-omp
	gcc ./POSIX/p2-matrixMult_cpu-threads.c -lpthread -o ./bin/p2-matrixMult_cpu-threads
	nvcc ./CUDA/p2-matrixMult_gpu-threads.cu -o ./bin/p2-matrixMult_cpu-cuda