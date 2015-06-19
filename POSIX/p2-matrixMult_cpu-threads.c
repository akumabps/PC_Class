#include <stdio.h>
#include <stdlib.h>

int ** A;
int ** B;
int ** C;
int  XDIM;
int  YDIM;
long MATSIZE;

//Threads Matrix Multiplication (Thread per row)
struct args{
    int * val;
    int i;
};

//Calculates a row per thread
void matrixMult(void * arguments){
    struct args * temp_args = arguments;
    int i = temp_args->i;
    int j = 0; int k = 0;
    for(k = 0; k < XDIM; k++){
        for(j = 0; j < XDIM; j++)
            temp_args->val[k] += A[i][j] * B[j][k];
    }
}

void fill_matrix(int *** M, char zeros){
    *M = (int**)malloc(sizeof(int*)*YDIM);
    int i,j;
   for(i = 0; i < XDIM; i++){
        (*M)[i] = malloc(sizeof(int)*XDIM);
        for(j = 0; j < YDIM; j++){
            (*M)[i][j] = zeros == '0' ? rand()&0xF : 0;
        }
    }
}

void free_memory(int ** M){
   int i = 0;
   for(i = 0; i < XDIM; i++){
       free(M[i]);
   }
   free(M);
}

void print_matrix(int ** M){
    int row, column;
    for(row = 0; row < YDIM; row++){
        for(column = 0; column < XDIM; column++){
            printf("%d ", M[row][column]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv){   
    XDIM    = 40;
    YDIM    = 40;
    if(argc > 1){
        XDIM = atoi(argv[1]);
        YDIM = XDIM;
    }
    MATSIZE = XDIM * YDIM;
    fill_matrix(&A, '0');
    fill_matrix(&B, '0');
    fill_matrix(&C, '1');

    //print_matrix(A);
    //print_matrix(B);
    pthread_t * threads;
    threads = malloc(sizeof(pthread_t) * XDIM);

    int rc; 
    long t,k; k = 0;
    struct args * t_args;
    t_args = (struct args *) malloc(sizeof(struct args) * XDIM);

    for(t = 0; t < XDIM; t++){
        t_args[t].val = C[t];
        int w = 0;
        
        t_args[t].i = t;
        rc = pthread_create(&threads[t], NULL, &matrixMult, (void *)&t_args[t]); 
        if(rc){
            //wait for a bunch of thread to end (if exceds the max quantity)
            if(rc == 11){                
                for (; k < t; k++){
                    rc = pthread_join(threads[k], NULL);
                    if(rc){
                     printf("ERROR; return code from pthread_join() is %d\n",rc);
                     exit(-1);
                    }
                }
                k=t;t--;
            }
            else{
                printf("ERROR; return code from pthread_create() is %d\n", rc); 
                exit(-1);
            }
        }
    }    
    for (t = k; t < XDIM; t++){
        rc = pthread_join(threads[t], NULL);
		if(rc){
		 printf("ERROR; return code from pthread_join() is %d\n",rc);
		 exit(-1);
		}
    }

    //printf("\n");
    //print_matrix(C);

    free(A);
    free(B);
    free(C);

    return 0;
}
