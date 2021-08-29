//
// Created by cetinicz on 2021-08-07.
//

#include "Analyzer.h"
#include "DDT.h"
#include "DDTDef.h"
#include "DDTUtils.h"
#include "ParseMatrixMarket.h"
#include "GenericCodelets.h"

#include <numeric>
#include <valarray>

namespace DDT {
    const std::string ANALYSIS_HEADER =
            "MATRIX_NAME,"
            "THREADS,"
            "codelet_min_width,"
            "codelet_max_distance,"
            "only_fsc_codelets,"
            "rows,"
            "cols,"
            "nnz,"
            "average_row_length,"
            "average_row_length_std_deviation,"
            "average_row_sequential_component,"
            "average_row_sequential_component_std_deviation,"
            "average_length_row_sequential_component,"
            "average_length_row_sequential_component_std_deviation,"
            "average_row_overlap,"
            "average_row_overlap_std_deviation,"
            "average_row_skew,"
            "average_row_skew_std_deviation,"
            "average_col_distance,"
            "average_col_distance_std_deviation,"
            "unique_row_patterns," // FOD hash
            "average_num_unique_row_patterns,"
            "average_num_unique_row_patterns_std_deviation,"
            "percent_vectorizable,"
            "num_fsc,"
            "average_fsc_width,"
            "average_fsc_width_std_deviation,"
            "average_fsc_height,"
            "average_fsc_height_std_deviation,"
            "average_fsc_points,"
            "average_fsc_points_std_deviation,"
            "total_loads_fsc,"
            "num_psc1,"
            "average_psc1_width,"
            "average_psc1_width_std_deviation,"
            "average_psc1_height,"
            "average_psc1_height_std_deviation,"
            "average_psc1_points,"
            "average_psc1_points_std_deviation,"
            "total_loads_psc1,"
            "num_psc2,"
            "average_psc2_width,"
            "average_psc2_width_std_deviation,"
            "average_psc2_height,"
            "average_psc2_height_std_deviation,"
            "average_psc2_points,"
            "average_psc2_points_std_deviation,"
            "total_loads_psc2,"
            "num_psc3,"
            "average_psc3_width,"
            "average_psc3_width_std_deviation"
            "num_psc3_v1,"
            "average_psc3_v1_width,"
            "average_psc3_v1_width_std_deviation"
            "num_psc3_v2,"
            "average_psc3_v2_width,"
            "average_psc3_v2_width_std_deviation,"
            "total_loads_psc3,"
            "total_loads";
    void analyzeData(const DDT::GlobalObject& d, std::vector<DDT::Codelet*>** cll, const DDT::Config& config) {
        auto cl = new std::vector<DDT::Codelet*>[config.nThread]();

        for (int i = 0; i < d.sm->_final_level_no; i++) {
            for (int j = 0; j < d.sm->_wp_bounds[i]; ++j) {
                cl[0].insert(cl[0].end(), cll[i][j].begin(), cll[i][j].end());
            }
        }
        analyzeData(d, cl, config);
    }
    void analyzeData(const DDT::GlobalObject& d, const std::vector<DDT::Codelet*>* cll, const DDT::Config& config) {
        // Get matrix statistics
        auto m = readSparseMatrix<CSR>(config.matrixPath);

        std::vector<DDT::Codelet*> cl;
        for (int i = 0; i < config.nThread; ++i) {
            cl.insert(cl.end(), cll[i].begin(), cll[i].end());
        }

        // Get data from matrix
        int rows = m.r,
            cols = m.c,
            nnz = m.nz;
        double average_row_length = 0,
              average_row_length_std_deviation = 0,
              average_row_overlap = 0,
              average_row_overlap_std_deviation = 0,
              average_col_distance = 0,
              average_col_distance_std_deviation = 0,
              average_row_sequential_component = 0,
              average_length_row_sequential_component = 0,
              average_row_skew = 0,
              average_row_skew_std_deviation = 0.,
              unique_row_patterns = 0,
              average_num_unique_row_patterns = 0,
              average_num_unique_row_patterns_std_deviation = 0.,
              average_row_sequential_component_std_deviation = 0.,
              average_length_row_sequential_component_std_deviation = 0.,
              numSeqPoints = 0.;

        std::unordered_map<std::string, int> unique_row_patterns_hash;

        for (int ii = 0; ii < 2; ++ii) {
            for (int i = 0; i < m.r; ++i) {
                if (ii == 0) {
                    average_row_length += m.Lp[i + 1] - m.Lp[i];
                } else {
                    average_row_length_std_deviation += std::pow((m.Lp[i + 1] - m.Lp[i]) - average_row_length, 2);
                }

                std::stringstream ss;
                int length_row_sequential_component_cnt = 0,
                    row_sequential_component_cnt = 0;
                int currentSeqSize = 1;
                const int VW_SQS  = 4;
                for (int j = m.Lp[i]; j < m.Lp[i + 1]; ++j) {
                    if (j + 1 < m.Lp[i + 1]) {
                        if (ii == 0) {
                            average_col_distance += m.Li[j + 1] - m.Li[j];
                        } else {
                            average_col_distance_std_deviation += std::pow((m.Li[j + 1] - m.Li[j]) - average_col_distance, 2);
                        }
                        if (m.Li[j + 1] - m.Li[j] != 1) {
                            if (ii == 0) {
                                if (currentSeqSize >= VW_SQS) {
                                    numSeqPoints += currentSeqSize;
                                }
                                currentSeqSize = 1;
                                average_row_sequential_component++;
                            } else {
                                row_sequential_component_cnt++;
                            }
                        } else {
                            if (ii == 0) {
                                currentSeqSize++;
                                average_length_row_sequential_component++;
                            } else {
                                length_row_sequential_component_cnt++;
                            }
                        }
                        ss << (m.Li[j + 1] - m.Li[j]);
                    }
                }
                if (ii == 1) {
                    average_row_sequential_component_std_deviation += std::pow(row_sequential_component_cnt - average_row_sequential_component, 2);
                    average_length_row_sequential_component_std_deviation += std::pow(length_row_sequential_component_cnt - average_length_row_sequential_component, 2);
                }
                if (ii == 0) {
                    unique_row_patterns_hash[ss.str()] += 1;
                    if (currentSeqSize >= VW_SQS) {
                        numSeqPoints += currentSeqSize;
                    }
                }
                currentSeqSize = 1;
                if (i + 1 < m.r) {
                    if (ii == 0) {
                        average_row_skew += m.Li[m.Lp[i + 1]] - m.Li[m.Lp[i]];
                        average_row_overlap +=
                                std::min(m.Li[m.Lp[i + 1] - 1],
                                         m.Li[m.Lp[i + 2] - 1]) -
                                         std::max(m.Li[m.Lp[i]], m.Li[m.Lp[i + 1]]);
                    } else {
                        average_row_skew_std_deviation += std::pow(m.Li[m.Lp[i + 1]] - m.Li[m.Lp[i]] - average_row_skew,2);
                        average_row_overlap_std_deviation += std::pow((std::min(m.Li[m.Lp[i + 1] - 1],
                                                                               m.Li[m.Lp[i + 2] - 1]) -
                                                                                       std::max(m.Li[m.Lp[i]], m.Li[m.Lp[i + 1]])) - average_row_skew,2);
                    }
                }
            }
            if (ii == 0) {
                average_row_length /= m.r;
                average_row_overlap /= m.r;
                average_col_distance /= m.nz-m.r;
                average_row_skew /= m.r;
                average_row_sequential_component /= m.r;
                average_length_row_sequential_component /= m.r;
            } else {
                average_row_length_std_deviation = std::sqrt(average_row_length_std_deviation/m.r);
                average_row_overlap_std_deviation = std::sqrt(average_row_overlap_std_deviation/m.r);
                average_col_distance_std_deviation = std::sqrt(average_col_distance_std_deviation/(m.nz-m.r));
                average_row_skew_std_deviation = std::sqrt(average_row_skew_std_deviation/m.r);
                average_row_sequential_component_std_deviation = std::sqrt(average_row_sequential_component_std_deviation/m.r);
                average_length_row_sequential_component_std_deviation = std::sqrt(average_length_row_sequential_component_std_deviation/m.r);
            }
        }

        unique_row_patterns = unique_row_patterns_hash.size();
        average_num_unique_row_patterns =
                std::accumulate(
                      unique_row_patterns_hash.begin(),
                      unique_row_patterns_hash.end(),
                        0.,
                      [](float acc, const std::pair<std::string, int>& entry) {
                            return acc + entry.second;
                      }) / unique_row_patterns_hash.size();
        average_num_unique_row_patterns_std_deviation =
                std::sqrt(std::accumulate(
                        unique_row_patterns_hash.begin(),
                        unique_row_patterns_hash.end(),
                        0.,
                        [&](float acc, const std::pair<std::string, int>& entry) {
                            return acc + std::pow(entry.second-average_num_unique_row_patterns,2);
                        }) / unique_row_patterns_hash.size());

        int num_fsc  = 0,
            num_psc1 = 0,
            num_psc2 = 0,
            num_psc3 = 0,
            num_psc3_v1 = 0,
            num_psc3_v2 = 0;
        float average_fsc_width = 0,
              average_fsc_width_std_deviation = 0.,
              average_fsc_height = 0.,
              average_fsc_height_std_deviation = 0.,
              average_fsc_points = 0.,
              average_fsc_points_std_deviation = 0.,
              total_loads_fsc = 0.,
              average_psc1_width = 0.,
              average_psc1_width_std_deviation = 0.,
              average_psc1_height = 0.,
              average_psc1_height_std_deviation = 0.,
              average_psc1_points = 0.,
              average_psc1_points_std_deviation = 0.,
              total_loads_psc1 = 0.,
              average_psc2_width = 0.,
              average_psc2_width_std_deviation = 0.,
              average_psc2_height = 0.,
              average_psc2_height_std_deviation = 0.,
              average_psc2_points = 0.,
              average_psc2_points_std_deviation = 0.,
              total_loads_psc2 = 0.,
              average_psc3_width = 0.,
              average_psc3_width_std_deviation = 0.,
              average_psc3_v1_width = 0.,
              average_psc3_v1_width_std_deviation = 0.,
              average_psc3_v2_width = 0.,
              average_psc3_v2_width_std_deviation = 0.,
              total_loads_psc3 = 0.;

        // Calculate Codelet Means/Stdev
        for (int i = 0; i < 2; i++) {
            for (auto const &c : cl) {
                switch (c->get_type()) {
                    case DDT::TYPE_FSC:
                        if (i == 0) {
                            num_fsc++;
                            average_fsc_width += c->col_width;
                            average_fsc_height += c->row_width;
                            average_fsc_points += c->row_width*c->col_width;
                            total_loads_fsc += 6;
                        } else {
                            average_fsc_width_std_deviation += std::pow(c->col_width - average_fsc_width,2);
                            average_fsc_height_std_deviation += std::pow(c->row_width - average_fsc_height,2);
                            average_fsc_points_std_deviation += std::pow(c->row_width*c->col_width - average_fsc_points, 2);
                        }
                        break;
                    case DDT::TYPE_PSC1:
                        if (i == 0) {
                            num_psc1++;
                            average_psc1_width += c->col_width;
                            average_psc1_height += c->row_width;
                            average_psc1_points += c->col_width * c->row_width;
                            total_loads_psc1 += 4 + c->row_width;
                        } else {
                            average_psc1_width_std_deviation += std::pow(c->col_width - average_psc1_width,2);
                            average_psc1_height_std_deviation += std::pow(c->row_width - average_psc1_height,2);
                            average_psc1_points_std_deviation += std::pow(c->col_width * c->row_width - average_psc1_points, 2);
                        }
                        break;
                    case DDT::TYPE_PSC2:
                        if (i == 0) {
                            num_psc2++;
                            average_psc2_width  += c->col_width;
                            average_psc2_height += c->row_width;
                            average_psc2_points += c->col_width*c->row_width;
                            total_loads_psc2    += 5 + c->col_width;
                        } else {
                            average_psc2_width_std_deviation += std::pow(c->col_width - average_psc2_width,2);
                            average_psc2_height_std_deviation += std::pow(c->row_width - average_psc2_height,2);
                            average_psc2_points_std_deviation += std::pow(c->col_width*c->row_width - average_psc2_points, 2);
                        }
                        break;
                    case DDT::TYPE_PSC3:
                        if (i == 0) {
                            num_psc3++;
                            average_psc3_width += c->col_width;
                            total_loads_psc3   += 5 + c->col_width;
                        } else {
                            average_psc3_width_std_deviation += std::pow(c->col_width - average_psc3_width,2);
                        }
                        break;
                    case DDT::TYPE_PSC3_V1:
                        if (i == 0) {
                            num_psc3_v1++;
                            average_psc3_v1_width += c->col_width;
                        } else {
                            average_psc3_v1_width_std_deviation += std::pow(c->col_width - average_psc3_v1_width,2);
                        }
                        break;
                    case DDT::TYPE_PSC3_V2:
                        if (i == 0) {
                            num_psc3_v2++;
                            average_psc3_v2_width += c->col_width;
                        } else {
                            average_psc3_v2_width_std_deviation += std::pow(c->col_width - average_psc3_v2_width,2);
                        }
                        break;
                    default:
                        throw std::runtime_error(
                                "Error: codelet type not recognized by "
                                "analyzer... Exiting... ");
                }
            }
            if (i == 0) {
                average_fsc_width /= std::max(num_fsc, 1);
                average_psc1_width /= std::max(num_psc1, 1);
                average_psc2_width /= std::max(num_psc2, 1);
                average_psc3_width /= std::max(num_psc3, 1);
                average_psc3_v1_width /= std::max(num_psc3_v1, 1);
                average_psc3_v2_width /= std::max(num_psc3_v2, 1);
                average_fsc_height /= std::max(num_fsc, 1);
                average_psc1_height /= std::max(num_psc1, 1);
                average_psc2_height /= std::max(num_psc2, 1);
                average_fsc_points /= std::max(num_fsc, 1);
                average_psc1_points /= std::max(num_psc1, 1);
                average_psc2_points /= std::max(num_psc2, 1);
            } else {
                average_fsc_width_std_deviation = std::sqrt(average_fsc_width_std_deviation/std::max(num_fsc,1));
                average_psc1_width_std_deviation = std::sqrt(average_psc1_width_std_deviation/std::max(num_psc1,1));
                average_psc2_width_std_deviation = std::sqrt(average_psc2_width_std_deviation/std::max(num_psc2,1));
                average_psc3_width_std_deviation = std::sqrt(average_psc3_width_std_deviation/std::max(num_psc3,1));
                average_psc3_v1_width_std_deviation = std::sqrt(average_psc3_v1_width_std_deviation/std::max(num_psc3_v1,1));
                average_psc3_v2_width_std_deviation = std::sqrt(average_psc3_v2_width_std_deviation/std::max(num_psc3_v2,1));
                average_fsc_height_std_deviation = std::sqrt(average_fsc_height_std_deviation/std::max(num_fsc,1));
                average_psc1_height_std_deviation = std::sqrt(average_psc1_height_std_deviation/std::max(num_psc1,1));
                average_psc2_height_std_deviation = std::sqrt(average_psc2_height_std_deviation/std::max(num_psc2,1));
                average_fsc_points_std_deviation = std::sqrt(average_fsc_points_std_deviation/std::max(num_fsc,1));
                average_psc1_points_std_deviation = std::sqrt(average_psc1_points_std_deviation/std::max(num_psc1,1));
                average_psc2_points_std_deviation = std::sqrt(average_psc2_points_std_deviation/std::max(num_psc2,1));
            }
        }

        if (config.header) {
            std::cout << ANALYSIS_HEADER << std::endl;
        }

        std::cout <<
                config.matrixPath << "," <<
                config.nThread << "," <<
                DDT::clt_width << "," <<
                DDT::col_th << "," <<
                DDT::prefer_fsc << "," <<
                rows << "," <<
                cols << "," <<
                nnz  << "," <<
                average_row_length << "," <<
                average_row_length_std_deviation << "," <<
                average_row_sequential_component << "," <<
                average_row_sequential_component_std_deviation << "," <<
                average_length_row_sequential_component << "," <<
                average_length_row_sequential_component_std_deviation << "," <<
                average_row_overlap << "," <<
                average_row_overlap_std_deviation << "," <<
                average_row_skew << "," <<
                average_row_skew_std_deviation << "," <<
                average_col_distance << "," <<
                average_col_distance_std_deviation << "," <<
                unique_row_patterns << "," <<
                average_num_unique_row_patterns << "," <<
                average_num_unique_row_patterns_std_deviation << "," <<
                (numSeqPoints / nnz) << ",";
        std::cout <<
            num_fsc << "," <<
            average_fsc_width << "," <<
            average_fsc_width_std_deviation << "," <<
            average_fsc_height << "," <<
            average_fsc_height_std_deviation << "," <<
            average_fsc_points << "," <<
            average_fsc_points_std_deviation << "," <<
            total_loads_fsc << "," <<
            num_psc1 << "," <<
            average_psc1_width << "," <<
            average_psc1_width_std_deviation << "," <<
            average_psc1_height << "," <<
            average_psc1_height_std_deviation << "," <<
            average_psc1_points << "," <<
            average_psc1_points_std_deviation << "," <<
            total_loads_psc1 << "," <<
            num_psc2 << "," <<
            average_psc2_width << "," <<
            average_psc2_width_std_deviation << "," <<
            average_psc2_height << "," <<
            average_psc2_height_std_deviation << "," <<
            average_psc2_points << "," <<
            average_psc2_points_std_deviation << "," <<
            total_loads_psc2 << "," <<
            num_psc3 << "," <<
            average_psc3_width << "," <<
            average_psc3_width_std_deviation << "," <<
            num_psc3_v1 << "," <<
            average_psc3_v1_width << "," <<
            average_psc3_v1_width_std_deviation << "," <<
            num_psc3_v2 << "," <<
            average_psc3_v2_width << "," <<
            average_psc3_v2_width_std_deviation << "," <<
            total_loads_psc3 << "," <<
            (total_loads_fsc + total_loads_psc1 + total_loads_psc2 + total_loads_psc3) << "," <<
                std::endl;

        exit(0);
    }

    /**
     * @brief Calculates average run-time of given codelet
     *
     * @description Given a codelet c, this function runs the codelet
     * 100000 times to find the average run-time of the codelet given the
     * input vector x, matrix Ax and output vector y.
     *
     * @param y  Output vector for SpMV
     * @param x  Input vector for SpMV
     * @param m  Input sparse matrix for SpMV
     * @param c  Codelet to time on numerical method
     *
     * @return   Average run-time of codelet
     */
    double timeSingleCodelet(double*y, double* x, Matrix& m, Codelet *c) {
        const int NUM_RUNS = 1000000;

        const auto Ax = m.Lx;
        const auto Ai = m.Li;

        // Reset memory
        for (int i = 0; i < m.r; ++i) {
            y[i] = 0.;
        }
        for (int i = 0; i < m.c; ++i) {
            x[i] = 1.;
        }

        std::chrono::time_point<std::chrono::steady_clock> t1, t2;
        switch (c->get_type()) {
            case TYPE_FSC:
                t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < NUM_RUNS; ++i) {
                    fsc_t2_2DC(y, Ax, x, c->row_offset, c->first_nnz_loc,
                               c->lbr, c->lbr + c->row_width, c->lbc,
                               c->col_width + c->lbc, c->col_offset);
                }
                t2 = std::chrono::steady_clock::now();
                break;
            case TYPE_PSC1:
                t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < NUM_RUNS; ++i) {
                    psc_t1_2D4R(y, Ax, x, c->offsets, c->lbr,
                                c->lbr + c->row_width, c->lbc,
                                c->lbc + c->col_width);
                }
                t2 = std::chrono::steady_clock::now();
                break;
            case TYPE_PSC2:
                t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < NUM_RUNS; ++i) {
                    psc_t2_2DC(y, Ax, x, c->offsets, c->row_offset,
                               c->first_nnz_loc, c->lbr,
                               c->lbr + c->row_width, c->col_width,
                               c->col_offset);
                }
                t2 = std::chrono::steady_clock::now();
                break;
            case TYPE_PSC3:
                t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < NUM_RUNS; ++i) {
                    psc_t3_1D1R(y, Ax, Ai, x, c->offsets, c->lbr,
                                c->first_nnz_loc, c->col_width);
                }
                t2 = std::chrono::steady_clock::now();
                break;
            default:
                throw std::runtime_error("Error: codelet type not recognized...");
        }

        return getTimeDifference(t1,t2) / NUM_RUNS;
    }

    void printCodeletStatsHeader() {
        std::cout << "CODELET_TYPE,EXECUTION_TIME,WIDTH,HEIGHT,SEQUENTIAL_COMPONENTS\n";
    }

    /**
     * @brief Returns string associated with codelet type enum
     * @param cl Codelet object
     *
     * @return String associated with DDT::CodeletTypes enum
     */
    std::string getCodeletTypeString(DDT::Codelet* cl) {
        switch (cl->get_type()) {
            case DDT::TYPE_FSC:
                return "FSC";
            case DDT::TYPE_PSC1:
                return "PSC1";
            case DDT::TYPE_PSC2:
                return "PSC2";
            case DDT::TYPE_PSC3:
                return "PSC3";
            default:
                throw std::runtime_error("Error: Codelet Type not supported...");
        }
    }

    /**
     * @brief Returns number of regions in codelet with unit strides
     *
     * @description A strided region is one where the offset between two memory
     * tuples from each dimension is 0 or 1. This code finds all regions where
     * adjacent elements have 0/1 strides.
     *
     * @param cl Codelet to check
     * @return Number of strided regions in codelet with 0 or 1 as stride
     */
    int getSequentialComponents(DDT::Codelet* cl) {
        int sc = 1;
        if (cl->get_type() == DDT::TYPE_FSC || cl->get_type() == DDT::TYPE_PSC1) {
            return sc;
        }
        for (int i = 0; i < cl->col_width-1; ++i) {
            if (cl->offsets[i+1]-cl->offsets[i] != 1) {
                sc++;
            }
        }
        return sc;
    }

    void printCodeletStats(double* y, double* x, Matrix& m, Codelet *cl) {
        auto codeletRuntime = timeSingleCodelet(y,x,m,cl);
        auto sc = getSequentialComponents(cl);
        std::cout
                << getCodeletTypeString(cl) << ","
                << codeletRuntime << ","
                << cl->row_width  << ","
                << cl->col_width  << ","
                << sc << ",\n";
    }

    /**
     * @brief Prints out information about codelets found in a sparsity pattern
     *
     * @description Goes over every codelet inside a matrix sparsity pattern
     * and numerical method and prints out detailed statistics per codelet
     * on efficiencies, size and run-times.
     *
     * @param d The global object containing runtime information
     * @param cll The list of codelets associated with each thread
     * @param config The global configuration object
     *
     * @note This function with exit the program
     */
    void analyzeCodeletExection(const DDT::GlobalObject& d, const std::vector<DDT::Codelet*>* cll, const DDT::Config& config) {
        // Allocate memory and read matrix for timings
        auto m = readSparseMatrix<CSR>(config.matrixPath);
        auto y = new double[m.r]();
        auto x = new double[m.c]();

        printCodeletStatsHeader();
        for (int i = 0; i < config.nThread; ++i) {
            for (auto const&  cl : cll[i]) {
                printCodeletStats(y,x,m,cl);
            }
        }

        delete[] y;
        delete[] x;

        exit(0);
    }
}