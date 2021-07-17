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
    Matrix(int r, int c, int nz) : r(r), c(c), nz(nz), Lp(nullptr), Li(nullptr), Lx(nullptr) {}
    ~Matrix() = default;

    int r;
    int c;
    int nz;

    int* Lp;
    int* Li;
    double* Lx;
};

class CSR : public Matrix {
  public:
    ~CSR(){
        delete[] this->Lp;
        delete[] this->Li;
        delete[] this->Lx;
    }
    CSR(int r, int c, int nz, RawMatrix m) : Matrix(r,c,nz) {
      this->Lp = new int[r+1]();
      this->Li = new int[nz]();
      this->Lx = new double[nz]();

        std::sort(m.begin(), m.end(), [](std::tuple<int,int,double> lhs, std::tuple<int,int,double> rhs){
          bool c0 = std::get<0>(lhs) == std::get<0>(rhs);
          bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
          bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
          return c0 ? c2 : c1;
        });

      // Parse CSR Matrix
      for (int i = 0, LpCnt = 0; i < nz; i++) {
        auto& v = m[i];

        int ov = std::get<0>(v);
        double im = std::get<2>(v);
        int iv = std::get<1>(v);

        this->Li[i] = iv;
        this->Lx[i] = im;

          if (i == 0) {
              this->Lp[LpCnt] = i;
          }
          if (i != 0 && std::get<0>(m[i-1]) != ov) {
              while (LpCnt != ov) {
                  this->Lp[++LpCnt] = i;
              }
          }
          if (nz - 1 == i) {
              this->Lp[++LpCnt] = i+1;
          }
      }
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

class CSC : public Matrix {
public:
    ~CSC(){
        delete[] this->Lp;
        delete[] this->Li;
        delete[] this->Lx;
    }
    CSC(int r, int c, int nz, RawMatrix m) : Matrix(r,c,nz) {
        this->Lp = new int[c+1]();
        this->Li = new int[nz]();
        this->Lx = new double[nz]();

        std::sort(m.begin(), m.end(), [](std::tuple<int,int,double> lhs, std::tuple<int,int,double> rhs){
          bool c0 = std::get<1>(lhs) == std::get<1>(rhs);
          bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
          bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
          return c0 ? c1 : c2;
        });

        // Parse CSR Matrix
        for (int i = 0, LpCnt = 0; i < nz; i++) {
            auto& v = m[i];

            int ov = std::get<0>(v);
            double im = std::get<2>(v);
            int iv = std::get<1>(v);

            this->Li[i] = ov;
            this->Lx[i] = im;

            if (i == 0) {
                this->Lp[LpCnt] = i;
            }
            if (i != 0 && std::get<1>(m[i-1]) != iv) {
                while (LpCnt != iv) {
                    this->Lp[++LpCnt] = i;
                }
            }
            if (nz - 1 == i) {
                this->Lp[++LpCnt] = i+1;
            }
        }
    }

    void make_full() {
        auto lpc = new int[this->c+1]();
        auto lxc = new double[this->nz*2-this->c]();
        auto lic = new int[this->nz*2-this->c]();

        auto ind = new int[this->c]();

        for(size_t i = 0; i < this->c; i++) {
            for(size_t p = this->Lp[i]; p < this->Lp[i+1]; p++) {
                int row = this->Li[p];
                ind[i]++;
                if(row != i)
                    ind[row]++;
            }
        }
        lpc[0] = 0;
        for(size_t i = 0; i < this->c; i++)
            lpc[i+1] = lpc[i] + ind[i];

        for(size_t i = 0; i < this->c; i++)
            ind[i] = 0;
        for(size_t i = 0; i < this->c; i++) {
            for(size_t p = this->Lp[i]; p < this->Lp[i+1]; p++) {
                int row = this->Li[p];
                int index = lpc[i] + ind[i];
                lic[index] = row;
                lxc[index] = this->Lx[p];
                ind[i]++;
                if(row != i) {
                    index = lpc[row] + ind[row];
                    lic[index] = i;
                    lxc[index] = this->Lx[p];
                    ind[row]++;
                }
            }
        }
        delete[]ind;
        delete Lp;
        delete Li;
        delete Lx;

        this->nz = this->nz*2-this->c;
        this->Lx = lxc;
        this->Li = lic;
        this->Lp = lpc;
    }


    // Move Constructor
    CSC (CSC&& lhs)  noexcept : Matrix(lhs) {
            this->Lp = lhs.Lp;
            this->Lx = lhs.Lx;
            this->Li = lhs.Li;

            lhs.Lp = nullptr;
            lhs.Li = nullptr;
            lhs.Lx = nullptr;
    }
};

template <class type>
auto readSparseMatrix(const std::string& path) -> type {
    std::ifstream file;
    file.open(path, std::ios_base::in);
    if (!file.is_open()) {
        std::cout << "File could not be found..." << std::endl;
        exit(1);
    }

    RawMatrix mat;


  int rows,cols,nnz;
  std::string line;
  bool parsed = false;
  bool sym = false;
  if (file.is_open()) {
    std::stringstream ss;
    std::getline (file,line);
    ss << line;
    // Junk
    std::getline(ss, line, ' ');
      // Matrix
      std::getline(ss, line, ' ');
      // Type
      std::getline(ss, line, ' ');
      if (line != "coordinate") {
          std::cout << "Can only process real matrices..." << std::endl;
          exit(1);
      }
      std::getline(ss, line, ' ');

      // Symmetric
      std::getline(ss, line, ' ');
      if (line == "symmetric") {
          sym = true;
      }

      ss.clear();

    while (std::getline (file,line)) {
          if (line[0] == '%') { continue; }
          if (!parsed) {
              ss << line;
              ss >> rows >> cols >> nnz;
              parsed = true;
              mat.reserve(sym ? nnz*2-rows : nnz);
              ss.clear();
              break;
          }
      }
      for (int i = 0; i < nnz; i++) {
          std::getline (file,line);
          std::tuple<int,int,double> t;
          ss << line;
          int row, col;
          double value;
          ss >> row >> col >> value;
          mat.emplace_back(std::make_tuple(row-1,col-1,value));
          if (sym && col != row) {
              mat.emplace_back(std::make_tuple(col - 1, row - 1, value));
          }
          ss.clear();
    }
  }
  file.close();

  // Turn into CSC
//  CSC cc(rows, cols, nnz, mat);

  // Make full
//  if (sym) {
//      cc.make_full();
//  }


  // Turn into CSR
//  RawMatrix mat2;
//  mat2.reserve(cc.nz);
//  for (int i = 0; i < cc.c; i++) {
//      for (int j = cc.Lp[i]; j < cc.Lp[i+1]; j++) {
//          mat2.emplace_back(cc.Li[j], i, cc.Lx[j]);
//      }
//  }

  auto ccr = CSR( rows, cols, sym ? nnz*2-rows : nnz, mat);

  if (std::is_same<type, CSR>::value) {
    return ccr;
  }
}

#endif  //DDT_PARSEMATRIXMARKET_H
