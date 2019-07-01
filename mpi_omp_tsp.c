#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include "mpi.h"
#include "tsp_mpi_headers.h"

#define OMP_THREADS 5

//These define tags for mpi communication
#define WORK	1
#define RESULT	2
#define DIE	3


int proc_n;
int my_rank;
MPI_Status status;
int jobs;

City cities[N_OF_CS];
int available[N_OF_CS];
double distance_m[N_OF_CS][N_OF_CS];

//Each omp thread will use one value of best_lengths
//and one row of best_paths.
double best_lengths[OMP_THREADS];
int best_paths[OMP_THREADS][N_OF_CS];

int debug_n = 0;

void print_debug() {
    printf("[%d]print debug %d\n", my_rank, debug_n);
    debug_n++;
}

void master_routine(Message *message_ptr, int best_path[],
                    int path_size, int *burst) {
    int i;
    if(path_size < N_OF_CS - MPI_GRAIN) {
        //Still building the partial path
        for(i=0; i<N_OF_CS; i++) {
            if(available[i]) {
                //City not yet chosen. Add it to the path. Mark it as unavailable.
                message_ptr->path[path_size] = i;
                available[i] = 0;
                //Go on with the recursion
                master_routine(message_ptr, best_path, path_size+1, burst);
                //Back from recursion. City is available again.
                available[i] = 1;
            } //Else we try the next city in the loop.
        }
        
    } else {
        //Time to forward the rest of the job for one of the MPI slaves
        if((*burst) < proc_n) {
            //First burst of jobs is sent with no need for slave request.
            MPI_Send(message_ptr, sizeof(Message), MPI_BYTE, (*burst), WORK, MPI_COMM_WORLD);
            //printf("job %d sent\n", *burst);
            (*burst)++;
            jobs++;
        } else {
            //All slaves received jobs already. Time to colect results
            //before sending anything else.
            Message results;
            MPI_Status status;
            //printf("Waiting results.\n");
            MPI_Recv(&results, sizeof(Message), MPI_BYTE, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            //printf("Results received.\n");
            if(results.best_length < message_ptr->best_length) {
                //A better path has been found.
                //Save it's length.
                message_ptr->best_length = results.best_length;
                //printf("*[M]: %.2f\n", message_ptr->best_length);
                //Copy path.
                for(i=0; i<N_OF_CS; i++) {
                    best_path[i] = results.path[i];
                }
            } //Else ignore results received
            
            //Send new job to this slave
            MPI_Send(message_ptr, sizeof(Message), MPI_BYTE, status.MPI_SOURCE, WORK, MPI_COMM_WORLD);
            jobs++;
        }
    }
    
    
}


//Final stretch of permutations performed by individual threads.
void tsp(int path[], int path_size, int available[]) {
    //If the path contains a suficient number of cities. Let another thread
    //perform the remeining permutations.
    if(path_size == N_OF_CS) {
        //print_int_arr(path);
        int th_id = omp_get_thread_num();
        double path_length = calc_length(path, distance_m);
        if(path_length < best_lengths[th_id]) {
            printf("(%d) found length %.2f\n(%d) ", th_id, path_length, th_id);
            print_int_arr(path);
            //Update best path and length for this thread
            best_lengths[th_id] = path_length;
            copy_path(path, &(best_paths[th_id][0]));
        }
    } else {
        int i;
        printf("(%d) else\n", omp_get_thread_num());
        //Finds the next city from the list of
        //cities which haven't been visited yet
        for(i=0; i<N_OF_CS; i++) {
            if(available[i]) {
                //Mark city as visited
                available[i] = 0;
                //Includes the new city in the current path
                path[path_size] = i;
                //Go on with the recursion. Notice the increment on path_size.
                tsp(path, path_size+1, available);
                //At this point the above recursion is backtracking.
                //Mark this city as available again before continuing on this loop,
                //so that deeper levels of the recursion can utilize it.
                available[i] = 1;
            }
        }
    }
}


//Preprocessing before sending final tasks for each thread
void thread_setup(int path[], int path_size, int available[]) {
    //If the path contains a suficient number of cities. Let another thread
    //perform the remaining permutations.
    if(path_size == N_OF_CS - OMP_GRAIN) {
        //Copy current built path and available cities.
        int copy_avail[N_OF_CS];
        copy_path(available, copy_avail);
        int path_copy[N_OF_CS];
        copy_path(path, path_copy);
        //Let another thread finish this job.
        #pragma omp task firstprivate(copy_avail, path_copy)
        {
            printf("[%d] starting thread (%d)\n", my_rank, omp_get_thread_num());
            tsp(path_copy, path_size, copy_avail);
        }
    } else {
        int i;
        //Finds the next city from the list of
        //cities which haven't been visited yet
        for(i=0; i<N_OF_CS; i++) {
            if(available[i]) {
                //Mark city as visited
                available[i] = 0;
                //Includes the new city in the current path
                path[path_size] = i;
                //Go on with the recursion. Notice the increment on path_size.
                thread_setup(path, path_size+1, available);
                //At this point the above recursion is backtracking.
                //Mark this city as available again before continuing on this loop,
                //so that deeper levels of the recursion can utilize it.
                available[i] = 1;
            }
        }
    }
}


//Works on distributing the work received by the MPI master among the OMP threads.
//Then checks the results. If the results are better, writes then in the message
//buffer and resturns 1. Otherwise returns 0.
int slave_routine(Message *msg_ptr) {
    //Mark all unavailable cities
	printf("[%d] slave routine start\n", my_rank);
	int i;
	for(i=0; i<N_OF_CS-MPI_GRAIN; i++) {
	    available[msg_ptr->path[i]] = 0;
	}
	//Update best found length for everyone according to the message of MPI master.
	for(i=0; i<OMP_THREADS; i++) {
	    best_lengths[i] = msg_ptr->best_length;
	}
	print_debug();
	//int path_copy[N_OF_CS];
	//copy_path(msg_ptr->path, path_copy);
	
	//Work on permutations. OMP time.
	#pragma omp parallel num_threads(OMP_THREADS)
	{
	    #pragma omp single
	    {
	        thread_setup(msg_ptr->path, N_OF_CS - MPI_GRAIN, available);
	    
	        print_debug();
	        //Cities already on the path were not marked as available again
	        //during tsp_aux recursion. This must be corrected here.
	        for(i=0; i<N_OF_CS-MPI_GRAIN; i++) {
	            available[msg_ptr->path[i]] = 1;
	        }
	    }
	}//All threads done.
	print_debug();
	//Check results of each thread.
	int best = -1;
	for(i=0; i<OMP_THREADS; i++) {
        printf("%d: %f\n", i, best_lengths[i]);
        if(best_lengths[i] < msg_ptr->best_length) {
            //Found a better result.
            best = i;
            msg_ptr->best_length = best_lengths[i];
        }
    }
    print_debug();
    if(best!=-1) {
        //Store best path if found.
        copy_path(&(best_paths[i][0]), msg_ptr->path);
        return 1;
    } else {
        return 0;
    }
}







int main(int argc, char **argv) {
    
    Message message;         //Message buffer (see header file)
    
    int i;
    
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
	
    
    //Fill paths with placeholder -1 value.
    //Mark all cities as available.
    for(i=0; i<N_OF_CS; i++) {
        message.path[i] = -1;
        available[i] = 1;
    }
    message.best_length = DBL_MAX;
    
    
    
	cities[0].name = "a"; cities[0].x = 125; cities[0].y = 832;
	cities[1].name = "b"; cities[1].x = 18; cities[1].y = 460;
	cities[2].name = "c"; cities[2].x = 176; cities[2].y = 386;
	cities[3].name = "d"; cities[3].x = 472; cities[3].y = 1000;
	cities[4].name = "e"; cities[4].x = 110; cities[4].y = 57;
	/*
	cities[5].name = "f"; cities[5].x = 790; cities[5].y = 166;
	cities[6].name = "g"; cities[6].x = 600; cities[6].y = 532;
	cities[7].name = "h"; cities[7].x = 398; cities[7].y = 40;
	cities[8].name = "i"; cities[8].x = 83; cities[8].y = 720;
	cities[9].name = "j"; cities[9].x = 829; cities[9].y = 627;
	cities[10].name = "k"; cities[10].x = 155; cities[10].y = 567;
	cities[11].name = "l"; cities[11].x = 930; cities[11].y = 106;
	cities[12].name = "m"; cities[12].x = 710; cities[12].y = 266;
	cities[13].name = "n"; cities[13].x = 33; cities[13].y = 680;
	//cities[14].name = "o"; cities[14].x = 672; cities[14].y = 415;
	*/
	
	//Creates and fills a distance table with distances between all cities
    fill_distance_m(distance_m, cities);
    
    
    if(my_rank == 0) {
    //================master======================
        printf("Computing solution using %d processes\n", proc_n);
        printf("Number of cities: %d\n", N_OF_CS);
        printf("MPI grain: %d\n", MPI_GRAIN);
        printf("OMP grain: %d\n", OMP_GRAIN);
        
        double t1,t2;
	    t1 = MPI_Wtime();  // inicia a contagem do tempo
	
	    int best_path[N_OF_CS];
	    for(i=0; i<N_OF_CS; i++) {
            best_path[i] = -1;
        }
	    //Computation starts defining city 0 as starting city
        message.path[0] = 0;
        //The city is marked as unavailable.
		available[0] = 0;
        int burst = 1;
        master_routine(&message, best_path, 1, &burst);
        
        //All work sent. Slaves are blocked on send with their last results.
        //Receive and send final message.
        printf("All work sent. Waiting final results.\n");
        int done = 0;
        Message results;
        while(done<proc_n-1) {
            
            MPI_Recv(&results, sizeof(Message), MPI_BYTE,
                     MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            if(results.best_length < message.best_length) {
                //A better path has been found.
                //Save it's length.
                message.best_length = results.best_length;
                //printf("final check: %.2f\n", message.best_length);
                //Copy path.
                for(i=0; i<N_OF_CS; i++) {
                    best_path[i] = results.path[i];
                }
            } //Else ignore results received.
            
            //Send final message to this slave.
            MPI_Send(&message, sizeof(Message), MPI_BYTE,
                     status.MPI_SOURCE, DIE, MPI_COMM_WORLD);
            done++;
        }
        
        t2 = MPI_Wtime(); // termina a contagem do tempo
        //Print solution
        printf("Best path: ");
        for(i=0; i<N_OF_CS; i++) {
            printf("%d ", best_path[i]);
        }
        printf("\nLength: %.2f\n", message.best_length);
        printf("Time: %.2f seconds\n", t2-t1);
        printf("Jobs sent: %d\n", jobs);
        
	
	} else {
	//================slave=======================
	    printf("[%d] start\n", my_rank);
	    int j;
	    for(i=0; i<OMP_THREADS; i++) {
	        for(j=0; j<N_OF_CS; j++) {
	            //Start all paths with placeholder -1.
	            best_paths[i][j] = -1;
	        }
	    }
	    while(1) {
		    MPI_Recv(&message, sizeof(Message), MPI_BYTE, 0,
		             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == WORK) { //Received a job
				
				//Perform computations.
				slave_routine(&message);
				
				//Send back results.
				MPI_Send(&message, sizeof(Message), MPI_BYTE,
				         0, RESULT, MPI_COMM_WORLD);
                
			} else { //No more work to do
			    printf("[%d]finished\n", my_rank);
			    break;
			}
		}
    
    }
    //================wrapping up=====================
    MPI_Finalize();
}    
