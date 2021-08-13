//
// Created by cetinicz on 2021-08-07.
//

#include "Analyzer.h"
#include "DDT.h"
#include "DDTDef.h"
#include "ParseMatrixMarket.h"

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
            "num_fsc,"
            "average_fsc_width,"
            "average_fsc_width_std_deviation,"
            "average_fsc_height,"
            "average_fsc_height_std_deviation,"
            "average_fsc_points,"
            "average_fsc_points_std_deviation,"
            "num_psc1,"
            "average_psc1_width,"
            "average_psc1_width_std_deviation,"
            "average_psc1_height,"
            "average_psc1_height_std_deviation,"
            "average_psc1_points,"
            "average_psc1_points_std_deviation,"
            "num_psc2,"
            "average_psc2_width,"
            "average_psc2_width_std_deviation,"
            "average_psc2_height,"
            "average_psc2_height_std_deviation,"
            "average_psc2_points,"
            "average_psc2_points_std_deviation,"
            "num_psc3,"
            "average_psc3_width,"
            "average_psc3_width_std_deviation"
            "num_psc3_v1,"
            "average_psc3_v1_width,"
            "average_psc3_v1_width_std_deviation"
            "num_psc3_v2,"
            "average_psc3_v2_width,"
            "average_psc3_v2_width_std_deviation";
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
              average_length_row_sequential_component_std_deviation = 0.;

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
                for (int j = m.Lp[i]; j < m.Lp[i + 1]; ++j) {
                    if (j + 1 < m.Lp[i + 1]) {
                        if (ii == 0) {
                            average_col_distance += m.Li[j + 1] - m.Li[j];
                        } else {
                            average_col_distance_std_deviation += std::pow((m.Li[j + 1] - m.Li[j]) - average_col_distance, 2);
                        }
                        if (m.Li[j + 1] - m.Li[j] != 1) {
                            if (ii == 0) {
                                average_row_sequential_component++;
                            } else {
                                row_sequential_component_cnt++;
                            }
                        } else {
                            if (ii == 0) {
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
                }
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
              average_psc1_width = 0.,
              average_psc1_width_std_deviation = 0.,
              average_psc1_height = 0.,
              average_psc1_height_std_deviation = 0.,
              average_psc1_points = 0.,
              average_psc1_points_std_deviation = 0.,
              average_psc2_width = 0.,
              average_psc2_width_std_deviation = 0.,
              average_psc2_height = 0.,
              average_psc2_height_std_deviation = 0.,
              average_psc2_points = 0.,
              average_psc2_points_std_deviation = 0.,
              average_psc3_width = 0.,
              average_psc3_width_std_deviation = 0.,
              average_psc3_v1_width = 0.,
              average_psc3_v1_width_std_deviation = 0.,
              average_psc3_v2_width = 0.,
              average_psc3_v2_width_std_deviation = 0.;

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
                        } else {
                            average_psc1_width_std_deviation += std::pow(c->col_width - average_psc1_width,2);
                            average_psc1_height_std_deviation += std::pow(c->row_width - average_psc1_height,2);
                            average_psc1_points_std_deviation += std::pow(c->col_width * c->row_width - average_psc1_points, 2);
                        }
                        break;
                    case DDT::TYPE_PSC2:
                        if (i == 0) {
                            num_psc2++;
                            average_psc2_width += c->col_width;
                            average_psc2_height += c->row_width;
                            average_psc2_points += c->col_width*c->row_width;
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
                average_num_unique_row_patterns_std_deviation << ",";
        std::cout <<
            num_fsc << "," <<
            average_fsc_width << "," <<
            average_fsc_width_std_deviation << "," <<
            average_fsc_height << "," <<
            average_fsc_height_std_deviation << "," <<
            average_fsc_points << "," <<
            average_fsc_points_std_deviation << "," <<
            num_psc1 << "," <<
            average_psc1_width << "," <<
            average_psc1_width_std_deviation << "," <<
            average_psc1_height << "," <<
            average_psc1_height_std_deviation << "," <<
            average_psc1_points << "," <<
            average_psc1_points_std_deviation << "," <<
            num_psc2 << "," <<
            average_psc2_width << "," <<
            average_psc2_width_std_deviation << "," <<
            average_psc2_height << "," <<
            average_psc2_height_std_deviation << "," <<
            average_psc2_points << "," <<
            average_psc2_points_std_deviation << "," <<
            num_psc3 << "," <<
            average_psc3_width << "," <<
            average_psc3_width_std_deviation << "," <<
            num_psc3_v1 << "," <<
            average_psc3_v1_width << "," <<
            average_psc3_v1_width_std_deviation << "," <<
            num_psc3_v2 << "," <<
            average_psc3_v2_width << "," <<
            average_psc3_v2_width_std_deviation << "," <<
                std::endl;

        exit(0);
    }
}