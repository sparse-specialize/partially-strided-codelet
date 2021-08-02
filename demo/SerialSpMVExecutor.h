//
// Created by cetinicz on 2021-08-02.
//

#ifndef DDT_SERIALSPMVEXECUTOR_H
#define DDT_SERIALSPMVEXECUTOR_H

#include <c++/10/iostream>
typedef void (*FunctionPtr)();

    inline double *yp;
    inline double *xp;
    inline double *axp;
    inline int    *app;
    inline int *axi;
    inline int m;
    inline FunctionPtr* fp;
    inline FunctionPtr* mapped_fp;

    void v1();

    inline void v1_3() {
        auto p = axi[1];
        *yp += *axp * xp[*axi];
        *yp += axp[1] * xp[p+0];
        *yp += axp[2] * xp[p+1];
        *yp += axp[3] * xp[p+2];
        axi+=4;
        axp+=4;
    }

    inline void v1_2_1() {
        auto p1 = axi[1];
        *yp += *axp * xp[*axi];
        *yp += axp[1] * xp[p1];
        *yp += axp[2] * xp[p1+1];
        *yp += axp[3] * xp[axi[3]];
        axi+=4;
        axp+=4;
    }

    inline void v1_3_1() {
        auto p1 = axi[1];
        *yp += *axp * xp[*axi];
        *yp += axp[1] * xp[p1];
        *yp += axp[2] * xp[p1+1];
        *yp += axp[3] * xp[p1+2];
        *yp += axp[4] * xp[axi[4]];
        axi+=5;
        axp+=5;
    }

    inline void v1_1_2_1() {
        *yp += *axp++ * xp[*axi++];
        *yp += *axp++ * xp[*axi++];
        *yp += *axp++ * xp[*axi++];
        *yp += *axp++ * xp[*axi++];
        *yp += *axp++ * xp[*axi++];
    }

    inline void v1_1_3_1() {
        auto p = axi[2];
        *yp += axp[0] * xp[axi[0]];
        *yp += axp[1] * xp[axi[1]];
        *yp += axp[2] * xp[p];
        *yp += axp[3] * xp[p+1];
        *yp += axp[4] * xp[p+2];
        *yp += axp[5] * xp[axi[5]];
        axp+=6;
        axi+=6;
    }

    void v2();

    inline void v2_() {
        auto p = *axi;
        *yp += axp[0] * xp[p];
        *yp += axp[1] * xp[p+1];
        axi+=2;
        axp+=2;
    }

    inline void v2_1() {
        auto p = *axi;
        *yp += axp[0] * xp[p];
        *yp += axp[1] * xp[p+1];

        *yp += axp[2] * xp[axi[2]];
        axi+=3;
        axp+=3;
    }

    void v3();

    inline void v3_() {
        auto p = *axi;
        *yp += axp[0] * xp[p];
        *yp += axp[1] * xp[p+1];
        *yp += axp[2] * xp[p+2];
        axi+=3;
        axp+=3;
    }

    inline void v3_1() {
        auto p = *axi;
        *yp += *axp++ * xp[p];
        *yp += *axp++ * xp[p+1];
        *yp += *axp++ * xp[p+2];
        axi+=3;
        *yp += *axp++ * xp[*axi++];
    }

    void v4();

    void v5();

    void v6();

    void v7();

    void v8();

    void v9();

    void v10();

    void v11();

    void v12();

    void v13();

    template<int nOps>
    inline void vs();

    template<int nOps>
    inline void vs() {
        *yp += *axp++ * xp[*axi++];
        vs<nOps-1>();
    }
    template<>
    inline void vs<0>() {}

    void build_templates();

    inline void executeSPMV() {
    for (int i = 0; i < m; ++i) {
        fp[i]();
        yp++;
    }
}

#endif//DDT_SERIALSPMVEXECUTOR_H
