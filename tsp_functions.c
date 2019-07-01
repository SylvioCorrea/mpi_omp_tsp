#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "tsp_mpi_headers.h"

 //print list of cities
void print_cities(City cities[]) {
    int i, j;
    for(i=0; i<N_OF_CS; i++) {
        printf("%s (%f, %f)\n", cities[i].name, cities[i].x, cities[i].y);
    }
}

void print_int_arr(int arr[]) {
    int i;
    printf("[");
    for(i=0; i<N_OF_CS-1; i++) {
        printf("%d, ", arr[i]);
    }
    printf("%d]\n", arr[N_OF_CS-1]);
    
}

//returns the euclidean distance between 2 cities
double distance(City *c1, City *c2) {
    double a = c2->x - c1->x;
    double b = c2->y - c1->y;
    a *= a;
    b *= b;
    return sqrt(a+b);
}

//Copies contents from path1 to path2
void copy_path(int path1[], int path2[]) {
    //printf("Checkpoint copy in\n");
    int i;
    for(i=0; i<N_OF_CS; i++) {
        path2[i] = path1[i];
    }
    //printf("Checkpoint copy out\n");
}

//fills a table of distances between each city
void fill_distance_m(double distance_m[N_OF_CS][N_OF_CS], City cities[]) {
    int i, j;
    for(i=0; i<N_OF_CS; i++) {
        for (j=0; j<N_OF_CS; j++) {
            distance_m[i][j] = distance(&cities[i], &cities[j]);
        }
    }
}

//Returns the length of a given path containing all cities
double calc_length(int path[], double distance_m[N_OF_CS][N_OF_CS]) {
   //printf("Checkpoint calc in\n");
   int i;
    double length = 0.0;
    int c1, c2;
    for(i=0; i<N_OF_CS-1; i++) {
        c1 = path[i];
        c2 = path[i+1];
        //printf("%d %d\n", c1, c2);
        length += distance_m[c1][c2];
    }
    //printf("Checkpoint calc out\n");
    //Finally adds the distance between the last and first cities
    c1 = path[0];
    c2 = path[N_OF_CS-1];
    length += distance_m[c1][c2];
    return length;
}

//prints the 2d array of distances between each city
void print_distance_m(double *distance_m, int n) {
    int i, j;
    for(i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            printf("%f   ", *(distance_m + (i*n) + j));
        }
        printf("\n");
    }
}

//prints the 2d array of distances between each city
void print_distance_m2(double distance_m[N_OF_CS][N_OF_CS]) {
    int i, j;
    for(i=0; i<N_OF_CS; i++) {
        for (j=0; j<N_OF_CS; j++) {
            printf("%f   ", distance_m[i][j]);
        }
        printf("\n");
    }
}
