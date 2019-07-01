#include <stdio.h>
#include <stdlib.h>

//Defines the number of cities to be used in the problem
#define N_OF_CS 5

//Defines the amount of cities an MPI slave will be required to perform permutations on
#define MPI_GRAIN	4
//Defines the number of cities an OMP thread will be required to perform permutations on
#define OMP_GRAIN   3

typedef struct {
    char *name;
    double x;
    double y;
}City;

/* Used for process comunication.
   Master sends a partially filled city path that will
   be used by a slave as basis for possible permutations
   with the remaining cities. best_length holds the best
   distance received by the master until now.
   A slave will use the struct to send back the best
   path found along with its length.
   */
typedef struct {
    int path[N_OF_CS];
    double best_length;
}Message;


void print_cities(City cities[]);
void print_int_arr(int arr[]);
double distance(City *c1, City *c2);
void copy_path(int path1[], int path2[]);
void fill_distance_m(double distance_m[N_OF_CS][N_OF_CS], City cities[]);
double calc_length(int path[], double distance_m[N_OF_CS][N_OF_CS]);
void print_distance_m(double *distance_m, int n);
void print_distance_m2(double distance_m[N_OF_CS][N_OF_CS]);

void master_routine(Message *message_ptr, int best_path[],
                    int path_size, int *burst);
void tsp(int path[], int path_size, int available[]);
void thread_setup(int path[], int path_size, int available[]);
int slave_routine(Message *msg_ptr);
