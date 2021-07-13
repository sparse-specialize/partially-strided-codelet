//
// Created by cetinicz on 2021-07-07.
//
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

typedef std::vector<std::tuple<int,int,double>> RawMatrix;

#ifndef DDT_PARSEMATRIXMARKET_H
#define DDT_PARSEMATRIXMARKET_H

class Matrix {
  public:
    Matrix() = default;
    Matrix(int r, int c, int nz, RawMatrix m) : r(r), c(c), nz(nz), m(m), Lp(nullptr), Li(nullptr), Lx(nullptr) {}
    ~Matrix() = default;

    int r;
    int c;
    int nz;

    int* Lp;
    int* Li;
    double* Lx;

    RawMatrix m;
};

class CSR : public Matrix {
  public:
    ~CSR(){
        delete[] this->Lp;
        delete[] this->Li;
        delete[] this->Lx;
    }
    CSR(int r, int c, int nz, RawMatrix m) : Matrix(r,c,nz,m) {
      this->Lp = new int[r+1]();
      this->Li = new int[nz]();
      this->Lx = new double[nz]();

      // Parse CSR Matrix
      for (int i = 0, LpCnt = 0; i < nz; i++) {
        auto& v = m[i];

        int ov = std::get<0>(v);
        double im = std::get<2>(v);
        int iv = std::get<1>(v);

        this->Li[i] = iv;
        this->Lx[i] = im;

        if (i == 0 || std::get<0>(m[i-1]) != ov) {
          this->Lp[LpCnt++] = i;
        }
        if (i == nz - 1) {
          this->Lp[LpCnt] = i + 1;
        }
      }
    }

    // Copy Constructor
    CSR(const CSR &lhs) : CSR(lhs.r,lhs.c,lhs.nz,lhs.m) {
    }

    // Move Constructor
    CSR (CSR&& lhs)  noexcept : Matrix(lhs) {
        this->Lp = lhs.Lp;
        this->Lx = lhs.Lx;
        this->Li = lhs.Li;

        lhs.Lp = nullptr;
        lhs.Li = nullptr;
        lhs.Lx = nullptr;
    }
};

class CSC : public Matrix {};

template <class type>
auto readSparseMatrix(const std::string& path) -> type {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) {
      std::cout << "File could not be found..." << std::endl;
      exit(1);
  }

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


  if (std::is_same<type, CSR>::value) {
    std::sort(mat.begin(), mat.end(), [](std::tuple<int,int,double> lhs, std::tuple<int,int,double> rhs){
        bool c0 = std::get<0>(lhs) == std::get<0>(rhs);
        bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
        bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
        return c0 ? c2 : c1;
        });
    return CSR( rows, cols, nnz, mat);
  }
}

#endif//DDT_PARSEMATRIXMARKET_H
