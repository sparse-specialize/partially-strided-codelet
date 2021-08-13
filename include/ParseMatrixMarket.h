//
// Created by cetinicz on 2021-07-07.
//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <metis_interface.h>
#include <sparse_utilities.h>

typedef std::vector<std::tuple<int,int,double>> RawMatrix;

#ifndef DDT_PARSEMATRIXMARKET_H
#define DDT_PARSEMATRIXMARKET_H

namespace DDT {
class Matrix {
  public:
    Matrix() = default;
    Matrix(int r, int c, int nz) : r(r), c(c), nz(nz), Lp(nullptr), Li(nullptr), Lx(nullptr) {}
    ~Matrix(){
        delete[] this->Lp;
        delete[] this->Li;
        delete[] this->Lx;
    };

    // Assignment operator
    Matrix& operator=(Matrix&& lhs)  noexcept {
        this->nz = lhs.nz;
        this->r = lhs.r;
        this->c = lhs.c;

        this->Lp = lhs.Lp;
        this->Li = lhs.Li;
        this->Lx = lhs.Lx;

        lhs.Lp = nullptr;
        lhs.Li = nullptr;
        lhs.Lx = nullptr;

        return *this;
    }

    // Assignment operator
    Matrix& operator=(const Matrix& lhs) {
        if (this == &lhs) {
            return *this;
        }
        this->nz = lhs.nz;
        this->r = lhs.r;
        this->c = lhs.c;

        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r+1);
        std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
        std::memcpy(this->Lx, lhs.Lx, sizeof(double) * lhs.nz);

        return *this;
    }

    // Copy Constructor
    Matrix(const Matrix& lhs) {
        this->r = lhs.r;
        this->c = lhs.c;
        this->nz = lhs.nz;

        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r+1);
        std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
        std::memcpy(this->Lx, lhs.Lx, sizeof(double) * lhs.nz);
    }

    // Move Constructor
    Matrix(Matrix&& lhs)  noexcept {
        this->r = lhs.r;
        this->c = lhs.c;
        this->nz = lhs.nz;

        this->Lp = lhs.Lp;
        this->Lx = lhs.Lx;
        this->Li = lhs.Li;

        lhs.Lp = nullptr;
        lhs.Li = nullptr;
        lhs.Lx = nullptr;
    }

    int r;
    int c;
    int nz;

    int* Lp;
    int* Li;
    double* Lx;

    void print() {
        for (int i = 0; i < this->r; ++i) {
            for (int j = this->Lp[i]; j < this->Lp[i+1]; ++j) {
                std::cout << i << "," << this->Li[j] << std::endl;
            }
        }
    }
};

class CSR : public Matrix {
  public:
    ~CSR(){
        delete[] this->Lp;
        delete[] this->Li;
        delete[] this->Lx;
    }
    CSR(int r, int c, int nz) : Matrix(r,c,nz) {
        this->Lp = new int[r+1]();
        this->Li = new int[nz]();
        this->Lx = new double[nz]();
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

    // Copy constructor
    CSR (const CSR& lhs) : Matrix(lhs.r, lhs.c, lhs.nz) {
        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r+1);
        std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
        std::memcpy(this->Lx, lhs.Lx, sizeof(double) * lhs.nz);
    }

    // Assignment operator
    CSR& operator=(const CSR& lhs) {
        this->nz = lhs.nz;
        this->r = lhs.r;
        this->c = lhs.c;

        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r+1);
        std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
        std::memcpy(this->Lx, lhs.Lx, sizeof(double) * lhs.nz);

        return *this;
    }

    // Move Constructor
    CSR (CSR&& lhs)  noexcept : Matrix(lhs.r, lhs.c, lhs.nz) {
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
    CSC(int r, int c, int nz) : Matrix(r,c,nz) {
        this->Lp = new int[c + 1]();
        this->Li = new int[nz]();
        this->Lx = new double[nz]();
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

    // Assignment copy operator
    CSC& operator=(CSC&& lhs)  noexcept {
        this->Lp = lhs.Lp;
        this->Lx = lhs.Lx;
        this->Li = lhs.Li;

        lhs.Lp = nullptr;
        lhs.Li = nullptr;
        lhs.Lx = nullptr;

        return *this;
    }

    // Assignment operator
    CSC& operator=(const CSC& lhs) {
        if (&lhs == this) {
            return *this;
        }
        this->r  = lhs.r;
        this->c  = lhs.c;
        this->nz = lhs.nz;

        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::copy(lhs.Lp, lhs.Lp+lhs.r+1, this->Lp);
        std::copy(lhs.Lx, lhs.Lx+lhs.nz, this->Lx);
        std::copy(lhs.Li, lhs.Li+lhs.nz, this->Li);

        return *this;
    }

    // Copy Constructor
    CSC (CSC& lhs) : Matrix(lhs.r, lhs.c, lhs.nz) {
        this->Lp = new int[lhs.r+1]();
        this->Li = new int[lhs.nz]();
        this->Lx = new double[lhs.nz]();

        std::copy(lhs.Lp, lhs.Lp+lhs.r+1, this->Lp);
        std::copy(lhs.Lx, lhs.Lx+lhs.nz, this->Lx);
        std::copy(lhs.Li, lhs.Li+lhs.nz, this->Li);
    }

    // Move Constructor
    CSC (CSC&& lhs)  noexcept : Matrix(lhs.r, lhs.c, lhs.nz) {
            this->Lp = lhs.Lp;
            this->Lx = lhs.Lx;
            this->Li = lhs.Li;

            lhs.Lp = nullptr;
            lhs.Li = nullptr;
            lhs.Lx = nullptr;
    }
};

template <typename T>
void copySymLibMatrix(Matrix& m, T symLibMat) {
    // Convert matrix back into regular format
    m.nz = symLibMat->nnz;
    m.r  = symLibMat->m;
    m.c  = symLibMat->n;

    delete m.Lp;
    delete m.Lx;
    delete m.Li;

    m.Lp = new int[m.r+1]();
    m.Lx = new double[m.nz]();
    m.Li = new int[m.nz]();

    std::copy(symLibMat->p, symLibMat->p+symLibMat->m+1, m.Lp);
    std::copy(symLibMat->i, symLibMat->i+symLibMat->nnz, m.Li);
    std::copy(symLibMat->x, symLibMat->x+symLibMat->nnz, m.Lx);
}

template <typename type>
auto reorderSparseMatrix(const CSC& m) {
    // Organize data into sympiler based format
    auto symMat = new sym_lib::CSC(m.r, m.c, m.nz, m.Lp, m.Li, m.Lx);
    symMat->stype = 1;
    int  *perm;

    // Permute matrix into new configuration
    sym_lib::CSC* A_full = sym_lib::make_full(symMat);
    sym_lib::metis_perm_general(A_full, perm);
    sym_lib::CSC *Lt = transpose_symmetric(A_full, perm);
    sym_lib::CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);

    Matrix nm;
    if (std::is_same_v<CSR,type>) {
        auto csr = sym_lib::csc_to_csr(L1_ord);
        nm = CSR(csr->m, csr->n, csr->nnz);
        copySymLibMatrix(nm, csr);
    } else if (std::is_same_v<CSC,type>) {
        nm = CSR(L1_ord->m, L1_ord->n, L1_ord->nnz);
        copySymLibMatrix(nm, L1_ord);
    } else {
        throw std::runtime_error("Error: Unsupported matrix type in template instruction");
    }

    // Clean up memory
    delete Lt;
    delete[]perm;
    delete symMat;
    delete L1_ord;

    return nm;
}

template <class type>
Matrix readSparseMatrix(const std::string& path) {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) {
    std::cout << "File could not be found..." << std::endl;
    exit(1);
  }
  RawMatrix mat;

  int rows, cols, nnz;
  std::string line;
  bool parsed = false;
  bool sym = false;
  if (file.is_open()) {
    std::stringstream ss;
    std::getline(file, line);
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

#ifdef METISA
    auto ccc = CSC( rows, cols, sym ? nnz*2-rows : nnz, mat);
    if (std::is_same_v<type, CSR>) {
        return reorderSparseMatrix<CSR>(ccc);
    } else if (std::is_same_v<type, CSC>) {
        return reorderSparseMatrix<CSC>(ccc);
    } else {
        throw std::runtime_error("Error: Matrix storage format not supported");
    }
#endif

  if (std::is_same_v<type, CSR>) {
    return CSR( rows, cols, sym ? nnz*2-rows : nnz, mat);
  } else if (std::is_same_v<type, CSC>) {
    return CSC( rows, cols, sym ? nnz*2-rows : nnz, mat);
  } else {
    throw std::runtime_error("Error: Matrix storage format not supported");
  }
}
}

#endif  //DDT_PARSEMATRIXMARKET_H
