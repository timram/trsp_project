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

int procAmount;
int rank;
int graphSize;

void initData(int *distances, int *prevVertex, int size) {
    for (int i = 0; i < size; i++) {
        distances[i] = SUPER_BIG_NUM;
        prevVertex[i] = -1;
    }
}

void printSpanningTree(int *distances, int *prevVertexes) {
    for (int i = 0; i < graphSize; i++) {
        if (i != START_VERTEX) {
            cout << prevVertexes[i] << " -- ("<< distances[i] << ") -- >" << i << endl;
        }
    }
}


int *readAdjacencyMatrixForParallel() {
    ifstream file;
    file.open("text_graph_1.txt");
    int graphSize;

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
    int graphSize;

    if (rank == ROOT) {
        oneDMatrix = readAdjacencyMatrixForParallel();
    }

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

    MPI_Bcast(&graphSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

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

    MPI_Gather(processDistances, chunkSize, MPI_INT, allDistances, chunkSize, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(processPrevVertex, chunkSize, MPI_INT, allPrevVertex, chunkSize, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printSpanningTree(allDistances, allPrevVertex);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procAmount);

    mainParallelProcess();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
