//
// Created by cetinicz on 2021-07-07.
//
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

typedef std::vector<std::tuple<int,int,double>> RawMatrix;

#ifndef DDT_PARSEMATRIXMARKET_H
#define DDT_PARSEMATRIXMARKET_H

class Matrix {
  public:
    Matrix(int r, int c, int nz, RawMatrix m) : r(r), c(c), nz(nz), m(m) {}

    int r;
    int c;
    int nz;
    RawMatrix m;
};

class CSR : public Matrix {
  public:
    CSR(int r, int c, int nz, RawMatrix m) : Matrix(r,c,nz,m){}
};

class CSC : public Matrix {};

template <class type>
auto readSparseMatrix(std::string path) {
  std::ifstream file;
  file.open(path, std::ios_base::in);

  std::vector<std::tuple<int,int,double>> mat;

  int rows,cols,nnz;
  std::string line;
  bool parsed = false;
  if (file.is_open()) {
    std::stringstream ss;
    while ( std::getline (file,line) ) {
      if (line[0] == '%') {
        continue;
      }
      if (!parsed) {
        ss << line;
        ss >> rows >> cols >> nnz;
        parsed = true;
        mat.reserve(nnz);
        ss.clear();
      }
      std::tuple<int,int,double> t;
      ss << line;
      int row, col;
      double value;
      ss >> row >> col >> value;
      mat.emplace_back(std::make_tuple(row-1,col-1,value));
      ss.clear();
    }
  }
  file.close();

  std::sort(mat.begin(), mat.end(), [](std::tuple<int,int,double> lhs, std::tuple<int,int,double> rhs){
      bool c0 = std::get<0>(lhs) == std::get<0>(rhs);
      bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
      bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
      return c0 ? c2 : c1;
      });

  return CSR{ rows, cols, nnz, mat };
}

#endif//DDT_PARSEMATRIXMARKET_H
