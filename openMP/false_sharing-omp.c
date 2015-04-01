/*
 * Example for false sharing practice, Parallel Computing course.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 50000
#define PAD 8
#define NUM_THREADS 32

void pointers()
{
    int array[SIZE];
    int i, j;
    int * pointer_to_result[NUM_THREADS];
    int sum = 0;

    for(i = 0; i < SIZE; i++)
        array[i] = i;

#pragma omp parallel private(i,j) num_threads(NUM_THREADS)
{  
    int r_t = 0;
    int id = omp_get_thread_num();
    pointer_to_result[id] = &r_t;
    
    #pragma omp for
        for(i = 0; i < SIZE; i++)
            for(j = 0; j < SIZE; j++)
                r_t += array[i] + array[j];
 }
 
    for(i = 0; i < NUM_THREADS; i++)
        sum += *(pointer_to_result[i]);

    printf("Pointer Test\n");
    fflush(stdout);
}

void openMPPAD()
{
    int array[SIZE];
    int i,j,k;
    int result[NUM_THREADS][PAD] = {0};
    int sum = 0;

    for(i = 0; i < SIZE; i++)
    {
        array[i] = i;
    }

#pragma omp parallel private(i,j) num_threads(NUM_THREADS)
{
    int id = omp_get_thread_num();
    #pragma omp for
        for(i = 0; i < SIZE; i++)
            for(j = 0; j < SIZE; j++)
                result[id][0] += array[i] + array[j];
}

    for(i = 0; i < NUM_THREADS; i++)
        sum += result[i][0];

    printf("Matrix PAD Test\n");
    fflush(stdout);
}

void sequential()
{
    int array[SIZE];
    int i, j;
    int sum;

    for(i = 0; i < SIZE; i++)
        array[i] = i;

    for(i = 0; i < SIZE; i++)
        for(j = 0; j < SIZE; j++)
            sum += array[i] + array[j];

    printf("Sequential Test\n");
    fflush(stdout);
}

void simpleOpenMP()
{
    int array[SIZE];
    int i, j;
    int result[NUM_THREADS] = {0};
    int sum = 0;

    for(i = 0; i < SIZE; i++)
        array[i] = i;

#pragma omp parallel private(i,j) num_threads(NUM_THREADS)
{
    int id = omp_get_thread_num();
    
    #pragma omp for
        for(i = 0; i < SIZE; i++)
            for(j = 0; j < SIZE; j++)
                result[id] += array[i] + array[j];
}

    for(i = 0; i < NUM_THREADS; i++)
        sum += result[i];

    printf("OpenMP without PAD Test\n");
    fflush(stdout);
}

void printUsage()
{
    printf("Usage: testOpenMP [OPTION]\n");
    printf("1 \t Sequential\n");
    printf("2 \t Simple OpenMP\n");
    printf("3 \t openMP with PAD\n");
    printf("4 \t openMP with pointers\n");
    
    exit(0);
}

void main(int argc, char * argv[])
{
    if(argc != 2) printUsage();
    if(*argv[1] == '1') sequential();             // Sequential
    if(*argv[1] == '2') simpleOpenMP();       // Simple openMP without PAD, 
    if(*argv[1] == '3') openMPPAD();           // openMP with PAD using a Matrix representation
    if(*argv[1] == '4') pointers();                 // openMP without PAD using pointers to a separated location of memory initialized in each thread.
}
