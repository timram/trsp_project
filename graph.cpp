#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

#define ROOT 0
#define START_VERTEX 0
#define SUPER_BIG_NUM 99999
long totalTimeInMs;

int procAmount;
int rank;
int graphSize;

void updateTime(timeval start, timeval end) {
    long seconds, useconds, mseconds;
    
    seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mseconds = ((seconds)*4000 + useconds / 4000.0) + 0.5;

    totalTimeInMs += mseconds;
}

void initData(int *distances, int *prevVertex, int size) {
    for (int i = 0; i < size; i++) {
        distances[i] = SUPER_BIG_NUM;
        prevVertex[i] = -1;
    }
}

void printOneDArrayAsMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int id = i * cols + j;
            cout << matrix[id] << "  ";
        }
        cout << endl;
    }
}

void printSpanningTree(int *distances, int *prevVertexes) {
    for (int i = 0; i < graphSize; i++) {
        if (i != START_VERTEX) {
            cout << prevVertexes[i] << " -- ("<< distances[i] << ") -- >" << i << endl;
        }
    }
}

int **readAdjacencyMatrix() {
    ifstream file;
    file.open("graph_4000_50.0.txt");

    file >> graphSize;

    int **matrix = new int*[graphSize];
    for (int i = 0; i < graphSize; i++) {
        matrix[i] = new int[graphSize];
    }

    for (int i = 0; i < graphSize; i++) {
        matrix[i] = new int[graphSize];
        for (int j = 0; j < graphSize; j++) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

int *readAdjacencyMatrixForParallel() {
    ifstream file;
    file.open("graph_4000_50.0.txt");

    file >> graphSize;

    int chunkSize = graphSize / procAmount;
    int procMatrixSize = chunkSize * graphSize;

    int *matrix = new int[graphSize * graphSize];

    for (int i = 0; i < graphSize; i++) {
        int chunkCount = -1;
        for (int j = 0, k = 0; j < graphSize; j++, k++) {
            if (j % chunkSize == 0) {
                chunkCount++;
                k = 0;
            }
            int id = i * chunkSize + k + chunkCount * procMatrixSize;
            file >> matrix[id];
        }
    }

    return matrix;
}

void printMinimalSpanningTree(int **matrix) {
    int lastAddedVertex = 0;
    bool *usedVeretex = new bool[graphSize];
    int *distanceToSpanningTree = new int[graphSize];
    int *prevVertex = new int[graphSize];
    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i = 0; i < graphSize; i++) {
        usedVeretex[i] = false;
        distanceToSpanningTree[i] = SUPER_BIG_NUM;
        prevVertex[i] = -1;
    }

    usedVeretex[lastAddedVertex] = true;
    distanceToSpanningTree[lastAddedVertex] = START_VERTEX;
    prevVertex[lastAddedVertex] = START_VERTEX;

    for (int i = 0; i < graphSize - 1; i++) {
        int minWeight = SUPER_BIG_NUM;
        int closestNeighbour = lastAddedVertex;

        for (int i = 0; i < graphSize; i++) {
            if (!usedVeretex[i]) {
                int weight = matrix[lastAddedVertex][i] != 0 ? matrix[lastAddedVertex][i] : matrix[i][lastAddedVertex];

                if (weight != 0 && weight < distanceToSpanningTree[i]) {
                    distanceToSpanningTree[i] = weight;
                    prevVertex[i] = lastAddedVertex;
                }

                if (distanceToSpanningTree[i] < minWeight) {
                    minWeight = distanceToSpanningTree[i];
                    closestNeighbour = i;
                }
            }
        }

        usedVeretex[closestNeighbour] = true;

        lastAddedVertex = closestNeighbour;
    }

    gettimeofday(&end, NULL);

    updateTime(start, end);

    // printSpanningTree(distanceToSpanningTree, prevVertex);
}

void mainParallelProcess() {
    int *oneDMatrix;
    int *processMatrix;
    int *usedVertex;
    int *processDistances;
    int *processPrevVertex;
    int lastAddedVertex;
    int *allDistances;
    int *allPrevVertex;
    int workArraySize;
    int *workArray;
    struct timeval start, end;

    if (rank == ROOT) {
        oneDMatrix = readAdjacencyMatrixForParallel();
        gettimeofday(&start, NULL);
    }

    MPI_Bcast(&graphSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    usedVertex = new int[graphSize + 1];

    if (rank == ROOT) {
        lastAddedVertex = START_VERTEX;
        allDistances = new int[graphSize];
        allPrevVertex = new int[graphSize];

        for (int i = 0; i < graphSize; i++) {
            usedVertex[i] = 0;
        }

        usedVertex[lastAddedVertex] = 1;
        usedVertex[graphSize] = lastAddedVertex;
    }

    if (graphSize % procAmount != 0) {
        cout << "Can't spread data to the all process" << endl;
        return;
    }

    int chunkSize = graphSize / procAmount;
    int procMatrixSize = chunkSize * graphSize;

    processMatrix = new int[procMatrixSize];
    processDistances = new int[chunkSize];
    processPrevVertex = new int[chunkSize];
    workArraySize = procAmount * 2;
    workArray = new int[workArraySize];

    initData(processDistances, processPrevVertex, chunkSize);

    MPI_Scatter(oneDMatrix, procMatrixSize, MPI_INT, processMatrix, procMatrixSize, MPI_INT, ROOT, MPI_COMM_WORLD);

    for (int i = 0; i < graphSize - 1; i++) {
        MPI_Bcast(usedVertex, graphSize + 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        lastAddedVertex = usedVertex[graphSize];
        int *currClosestVertex = new int[2];
        currClosestVertex[0] = lastAddedVertex;
        currClosestVertex[1] = SUPER_BIG_NUM;

        for (int j = 0; j < chunkSize; j++) {
            int vtID  = rank * chunkSize + j;
            if (!usedVertex[vtID]) {
                int id = lastAddedVertex * chunkSize + j;
                
                int weight = processMatrix[id];

                if (weight != 0 && weight < processDistances[j]) {
                    processDistances[j] = weight;
                    processPrevVertex[j] = lastAddedVertex;
                }

                if (processDistances[j] < currClosestVertex[1]) {
                    currClosestVertex[0] = chunkSize * rank + j;
                    currClosestVertex[1] = processDistances[j];
                }
            }
        }

        MPI_Gather(currClosestVertex, 2, MPI_INT, workArray, 2, MPI_INT, ROOT, MPI_COMM_WORLD);

        if (rank == ROOT) {
            int closest = workArray[0];
            int minWeight = workArray[1];

            for (int i = 2; i < workArraySize; i += 2) {
                int id = workArray[i];
                int weight = workArray[i + 1];
                if (!usedVertex[id] && weight < minWeight) {
                    minWeight = weight;
                    closest = id;
                }
            }

            usedVertex[closest] = 1;
            usedVertex[graphSize] = closest;
        }
    }

    // MPI_Gather(processDistances, chunkSize, MPI_INT, allDistances, chunkSize, MPI_INT, ROOT, MPI_COMM_WORLD);
    // MPI_Gather(processPrevVertex, chunkSize, MPI_INT, allPrevVertex, chunkSize, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        gettimeofday(&end, NULL);
        updateTime(start, end);
        // printSpanningTree(allDistances, allPrevVertex);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procAmount);

    if (procAmount == 1) {
        int **matrix = readAdjacencyMatrix();
        printMinimalSpanningTree(matrix);
    } else {
        mainParallelProcess();
    }

    if (rank == ROOT) {
        cout << "Total vertexes: " << graphSize << endl;
        cout << "Total time in ms is: " << totalTimeInMs << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
