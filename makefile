all:
	gcc ./openMP/p2-matrixMult_cpu-omp.c -fopenmp -o ./bin/p2-matrixMult_cpu-omp
	gcc ./openMP/false_sharing-omp.c -fopenmp -o ./bin/false_sharing-omp
