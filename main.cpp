#include <iostream>
#include <random>

#include "PatternMatching.h"

struct MemoryTrace {
  int** ip;
  int ips;
};

MemoryTrace generateMockTuples() {
  // Generation parameters
  int matrixRows = 1000;
  int elementsPerRow = 100;
  int tpr = 3;

  // Memory Allocation - Padded to a width of 4
  int nd = tpr * (elementsPerRow) * matrixRows;
  int* d = new int[nd];
  posix_memalign(reinterpret_cast<void **>(d), 32, nd);
  for (int i = 0; i < nd; i++) {
      d[i] = i;
  }
  int** dPointer = new int*[matrixRows+1];

  // Assign pointers to start of rows
  for (int i = 0; i < matrixRows+1; i++) {
    dPointer[i] = d+(i*tpr*elementsPerRow);
  }

  // Return
  MemoryTrace mt{
          dPointer,
          matrixRows+1
  };

  return mt;
}

int main() {
  auto mt = generateMockTuples();

  // Allocate space for differences
  int ds = 0;
  for (int i = 0; i < mt.ips-1; i++) {
    ds += (reinterpret_cast<int*>(mt.ip[i+1]) - reinterpret_cast<int*>(mt.ip[i]));
  }

  auto d = new int[ds];

  // Compute and profile FOD compuation
  computeParallelizedFOD(mt.ip, mt.ips, d);

  return 0;
}
