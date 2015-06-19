#include <stdio.h>
#include <stdlib.h>

//Store in C the multiplication A*B
int * A;
int * B;
int * C;
int  XDIM;
int  YDIM;
int NUM_PROCESS = 2;

//Processes Matrix Multiplication
void matrixMult(int * C, int init , int end){
    //Execute operations in [init..end] range
    int row = init;
    int column;
    for(row = init; row < end; row++){
        for(column = 0; column < XDIM; column++){
            int i;
            int idx = row*XDIM + column;
            for(i = 0; i < XDIM ;i++){
                //in 2D array notation: C[row][col] = A[row][i]*B[i][col]
                C[idx] += A[row*XDIM + i]*B[i*XDIM + column];
            }
        }
    }
}

//recursive function that forks a process acording to times argument
void createChild(int init, int end, int times, int * C){
    if(times == 0){
        matrixMult(C, init, end);
        return; 
    }
    pid_t pid;
    int pipefd[2];
    int r;
    int * C_child;
    r = pipe(pipefd); if(r < 0) perror("error pipe ");
    //fork
    pid = fork(); if(r < 0) perror("error fork "); 
    //child
    if(pid == 0){
        close(pipefd[0]);
        createChild(init +((end - init)/(times)), end, times-1, C);
        write(pipefd[1], C, sizeof(int*)*XDIM*XDIM);
        close(pipefd[1]);
        exit(0);
    }else{
        //parent
        close(pipefd[1]);
        matrixMult(C, init, init +((end - init)/(times)));        
        C_child = (int*)malloc(sizeof(int*)*XDIM*XDIM);
        read(pipefd[0], C_child, sizeof(int*)*XDIM*XDIM);
        close(pipefd[0]);
        int i = 0;
        for(i = init +((end - init)/(times)); i < end; i++){
            int j = 0;
            for(j = 0; j < XDIM; j++){
                C[i*XDIM + j] = C_child[i*XDIM + j];
            }
        }
    }
}

void print_matrix(int * M){
    int row, column;
    for(row = 0; row < YDIM; row++){
        for(column = 0; column < XDIM; column++){
            printf("%d ", M[row*XDIM + column]);
        }
        printf("\n");
    }
}

void fill_matrix(int ** M, char zeros){
    *M = (int*)malloc(sizeof(int*)*XDIM*XDIM);
    int i,j;
    for(i = 0; i < XDIM*XDIM; i++){
        (*M)[i] = zeros == '0' ? rand()&0xF : 0;
    }
}

void free_memory(int * M){
   free(M);
}

int main(int argc, char **argv){   
    XDIM    = 40;
    YDIM    = 40;
    //read from arguments: size of the matrix, number of processes
    if(argc > 1){
        XDIM = atoi(argv[1]);
        YDIM = XDIM;
    }
    if(argc > 2) NUM_PROCESS = atoi(argv[2]);

    //fill A & B with random numbers, C with 0's
    fill_matrix(&A, '0');
    fill_matrix(&B, '0');
    fill_matrix(&C, '1');

    //print_matrix(A);
    //print_matrix(B);
    //create the desire processes
    createChild(0, XDIM, NUM_PROCESS, C);

    //printf("\n-------\n");
    //print_matrix(C);

    //free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
