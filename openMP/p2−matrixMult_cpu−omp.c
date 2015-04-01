#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define new_matrix_array(size) (int***)malloc(sizeof(int**)*size);
#define new_matrix(size) (int**)malloc(sizeof(int*)*size);
#define new_array(size) malloc(sizeof(int)*size);

enum {A_,B_,C_};
typedef enum {false, true} bool;
typedef int** matrix_t;

bool DEBUG;
int  XDIM;
int NUM_THREADS;

void calculateIndex(matrix_t A, matrix_t B, matrix_t C, int row, int column)
{
    int i;
    for(i = 0; i < XDIM ;i++)
        C[row][column] += A[row][i]*B[i][column];
}

void calculateRow(matrix_t A, matrix_t B, matrix_t C, int row)
{
    int column;
    for(column = 0; column < XDIM; column++)
        calculateIndex(A, B, C, row, column);
}

void productRow(matrix_t A, matrix_t B, matrix_t C)
{
    int row;
    #pragma omp parallel private(row) num_threads(NUM_THREADS)
    {
        #pragma omp for
        for(row = 0; row < XDIM; row++)
            calculateRow(A, B, C, row);
    }
}

void productIndex(matrix_t A, matrix_t B, matrix_t C)
{
    int row;
    int column;
    #pragma omp parallel private(row, column) num_threads(NUM_THREADS)
    {
        #pragma omp for collapse(2)
        for(row = 0; row < XDIM; row++)
            for(column = 0; column < XDIM; column++)
                calculateIndex(A, B, C, row, column);
    }
}

void fillMatrix(matrix_t * M, char zeros)
{
    *M = new_matrix(XDIM);
    int i,j;
    for(i = 0; i < XDIM; i++)
    {
        (*M)[i] = new_array(XDIM);
        for(j = 0; j < XDIM; j++)
            (*M)[i][j] = zeros == '0' ? rand() & 0xF : 0;
    }
}

void freeMemory(matrix_t M)
{
   int i = 0;
   for(i = 0; i < XDIM; i++)
       free(M[i]);
   free(M);
}

void printMatrix(matrix_t M)
{
    int row, column;
    for(row = 0; row < XDIM; row++)
    {
        for(column = 0; column < XDIM; column++)
            printf("%d\t", M[row][column]);
        printf("\n");
    }
    printf("\n");
}

void printAll(matrix_t* M)
{
    printf("Print result.\n");
    printf("A:\n");
    printMatrix(M[A_]);
    printf("B:\n");
    printMatrix(M[B_]);
    printf("C:\n");
    printMatrix(M[C_]);
}

void usage()
{
    printf("Usage: p2−matrixMult_cpu−omp [Matrix size] [Num threads] [Optional 1] [Optional 2]\n");
    printf("\t Matrix size: \t Integer\n");
    printf("\t Num threads: \t Integer\n");
    printf("\t Optional 1: \t 0 -> Use product parallel by Index\n");
    printf("\t\t\t 1 -> Use product parallel by Rows\n");
    printf("\t Optional 2: \t 0 -> No print result\n");
    printf("\t\t\t 1 -> Print result\n");

    exit(0);
}


void init(int argc, char **argv, void (**product)(matrix_t, matrix_t, matrix_t))
{
    if(argc < 3)
        usage();

    DEBUG = false;
    XDIM = atoi(argv[1]);
    NUM_THREADS = atoi(argv[2]);

    // Set the function to evaluate.
    if(argc > 3)
        *product = *argv[3] == '0' ? &productIndex : &productRow;

    // Set output flag
    if(argc > 4)
        DEBUG = *argv[4] == '0' ? false : true;
}


void main(int argc, char **argv)
{
    // Counter
    int i;
    // Function to use for the matrix product calc
    void (*product)(matrix_t, matrix_t, matrix_t) = &productRow;
    // Set the matrix dimension, the number of threads, and control flags
    init(argc, argv, &product);

    // Create the A, B and C matrix
    matrix_t * Mat = new_matrix_array(3);

    // Filling the A, B and C Matrix
    for(i = 0; i < 3; i++)
        fillMatrix(&Mat[i], i != C_ ? '0' : '1');

    // Calculating the product
    (*product)(Mat[A_], Mat[B_], Mat[C_]);

    if(DEBUG)
        printAll(Mat);

    // TODO: Free memory
}
