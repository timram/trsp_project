#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

#define ROOT 0

long originalLen;
int rank;
int processesNum;
int *srcData;
long totalTimeInMs;

int* generateArray() {
    int *arr = new int[originalLen];

    for (int i = 0; i < originalLen; i++) {
        arr[i] = rand() % 10000;
    }

    return arr;
}

int* readArrayFromFile() {    
    ifstream file;
    file.open("numbers.txt");

    file >> originalLen;
    cout << "Length of initial array: " << originalLen << endl;
    int *nums = new int[originalLen];

    for (int i = 0; i < originalLen; i++) {
        file >> nums[i];
    }

    file.close();

    return nums;
}

void printArr(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    
    cout << endl;
}

void bubbleSort(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j]  = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int *compareExchangeMin(int *arr, int *recv, int size) {
    int arrID = 0, recvID = 0, tempID = 0;
    int *temp = new int[size];

    while (tempID < size) {
        if (arr[arrID] <= recv[recvID]) {
            temp[tempID++] = arr[arrID++];
        } else {
            temp[tempID++] = recv[recvID++];
        }
    }

    return temp;
}

int *compareExchangeMax(int *arr, int *recv, int size)
{
    int arrID = size - 1, recvID = size - 1, tempID = size - 1;
    int *temp = new int[size];

    while (tempID >= 0) {
        if (arr[arrID] >= recv[recvID]) {
            temp[tempID--] = arr[arrID--];
        }
        else {
            temp[tempID--] = recv[recvID--];
        }
    }

    return temp;
}

int *parallelBubbelSort(int *procData, int nPerProc) {
    int oddRank;
    int evenRank;

    if (rank % 2 == 0) {
        oddRank = rank - 1;
        evenRank = rank + 1;
    }
    else {
        oddRank = rank + 1;
        evenRank = rank - 1;
    }

    int *received = new int[nPerProc];

    bubbleSort(procData, nPerProc);

    for (int i = 0;i < processesNum; i++) {
        if (i % 2 == 1 && oddRank >= 0 && oddRank < processesNum) {
            MPI_Sendrecv(procData, nPerProc, MPI_INT, oddRank, 0, received, nPerProc, MPI_INT, oddRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (rank % 2 == 1) {
                procData = compareExchangeMin(procData, received, nPerProc);
            } else {
                procData = compareExchangeMax(procData, received, nPerProc);
            }
        } else if (i % 2 == 0 && evenRank >= 0 && evenRank < processesNum) {
            MPI_Sendrecv(procData, nPerProc, MPI_INT, evenRank, 0, received, nPerProc, MPI_INT, evenRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (rank % 2 == 0) {
                procData = compareExchangeMin(procData, received, nPerProc);
            }
            else {
                procData = compareExchangeMax(procData, received, nPerProc);
            }
        }
    }
    
    return procData;
}

void updateTime(timeval start, timeval end) {
    long seconds, useconds, mseconds;
    
    seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mseconds = ((seconds)*1000 + useconds / 1000.0) + 0.5;

    totalTimeInMs += mseconds;
}

void mainProcess() {
    int *sortedData;
    int *procData;
    struct timeval start, end;

    if (rank == ROOT) {
        gettimeofday(&start, NULL);
        if (processesNum == 1) {
            cout << "Going to sort data in single process" << endl;
            bubbleSort(srcData, originalLen);
            cout << "Sorting finished" << endl;
            gettimeofday(&end, NULL);

            updateTime(start, end);
            return;
        }
        cout << "Start sorting" << endl;
    }

    MPI_Bcast(&originalLen, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    int nPerProc = originalLen / processesNum;

    if (originalLen % processesNum != 0) {
        cout << "Can't spread data through all processes" << endl;
        return;
    }

    procData = new int[nPerProc];

    MPI_Scatter(srcData, nPerProc, MPI_INT, procData, nPerProc, MPI_INT, ROOT, MPI_COMM_WORLD);

    procData = parallelBubbelSort(procData, nPerProc);

    if (rank == ROOT) {
        sortedData = new int[originalLen];
    }

    MPI_Gather(procData, nPerProc, MPI_INT, sortedData, nPerProc, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        cout << "Sorting finished" << endl;
        gettimeofday(&end, NULL);

        updateTime(start, end);
    }
}

int main(int argc, char*argv[]) {
    MPI_Init(&argc, &argv);  
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processesNum);

    if (rank == ROOT) {
        srcData = readArrayFromFile();
    }

    mainProcess();

    if (rank == ROOT) {
        cout << "Ellapsed time: " << totalTimeInMs << " ms" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}