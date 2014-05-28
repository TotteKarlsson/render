import numpy as np
cos = np.cos
sin = np.sin
sqrt = np.sqrt


def f_fprime(x1, y1, R1, Tx1, Ty1,
             x2, y2, R2, Tx2, Ty2,
             rotation_normalization, test=[0]):
    R1 = R1 / rotation_normalization
    R2 = R2 / rotation_normalization

    nx1 = cos(R1) * x1 - sin(R1) * y1 + Tx1
    ny1 = sin(R1) * x1 + cos(R1) * y1 + Ty1

    nx2 = cos(R2) * x2 - sin(R2) * y2 + Tx2
    ny2 = sin(R2) * x2 + cos(R2) * y2 + Ty2

    D = sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2 + 1)

    dDdR1 = ((nx1 - nx2) * ((-sin(R1) * x1 - cos(R1) * y1) / D) +
             (ny1 - ny2) * (( cos(R1) * x1 - sin(R1) * y1) / D)) / rotation_normalization

    test[0] = dDdR1 * D

    dDdTx1 =  (nx1 - nx2) / D
    dDdTy1 =  (ny1 - ny2) / D

    dDdR2 = ((nx2 - nx1) * ((-sin(R2) * x2 - cos(R2) * y2) / D) +
             (ny2 - ny1) * (( cos(R2) * x2 - sin(R2) * y2) / D)) / rotation_normalization

    dDdTx2 =  (nx2 - nx1) / D
    dDdTy2 =  (ny2 - ny1) / D

    return (D.sum(), 
            dDdR1.sum(), dDdTx1.sum(), dDdTy1.sum(),
            dDdR2.sum(), dDdTx2.sum(), dDdTy2.sum())


def Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
       x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
       rotation_normalization):
    R1 = R1 / rotation_normalization
    R2 = R2 / rotation_normalization
    vR1 = vR1 / rotation_normalization
    vR2 = vR2 / rotation_normalization

    # formulas are for R1 = R1 + e * vR1, etc., evaluated at e = 0
    nx1 = cos(R1) * x1 - sin(R1) * y1 + Tx1
    ny1 = sin(R1) * x1 + cos(R1) * y1 + Ty1
    nx2 = cos(R2) * x2 - sin(R2) * y2 + Tx2
    ny2 = sin(R2) * x2 + cos(R2) * y2 + Ty2

    d_nx1_d_e = vR1 * (-sin(R1) * x1 - cos(R1) * y1) + vTx1
    d_ny1_d_e = vR1 * ( cos(R1) * x1 - sin(R1) * y1) + vTy1
    d_nx2_d_e = vR2 * (-sin(R2) * x2 - cos(R2) * y2) + vTx2
    d_ny2_d_e = vR2 * ( cos(R2) * x2 - sin(R2) * y2) + vTy2

    Dsq = (nx1 - nx2)**2 + (ny1 - ny2)**2 + 1
    D = sqrt(Dsq)
    # F = (nx1 - nx2)**2 + (ny1 - ny2)**2 + 1
    # dD_de = d(sqrt(F)) / dF * (dF / dnx1 * dnx1 / de +
    #                            dF / dny1 * dny1 / de +
    #                            dF / dnx2 * dnx2 / de +
    #                            dF / dny2 * dny2 / de)
    d_D_d_e = (1 / 2.0) * (1 / D) * (2 * (nx1 - nx2) * d_nx1_d_e +
                                     2 * (ny1 - ny2) * d_ny1_d_e +
                                     2 * (nx2 - nx1) * d_nx2_d_e +
                                     2 * (ny2 - ny1) * d_ny2_d_e)

    dDdR1timesD = ((nx1 - nx2) * (-sin(R1) * x1 - cos(R1) * y1) +
                   (ny1 - ny2) * ( cos(R1) * x1 - sin(R1) * y1)) / rotation_normalization

    dDdR2timesD = ((nx2 - nx1) * (-sin(R2) * x2 - cos(R2) * y2) +
                   (ny2 - ny1) * ( cos(R2) * x2 - sin(R2) * y2)) / rotation_normalization

    # d(dDdR * D) / de
    d_dDdR1timesD_d_e = (((d_nx1_d_e - d_nx2_d_e) * ((-sin(R1) * x1 - cos(R1) * y1)) +
                          (nx1 - nx2) * vR1 *        (-cos(R1) * x1 + sin(R1) * y1)) +
                         ((d_ny1_d_e - d_ny2_d_e) * (( cos(R1) * x1 - sin(R1) * y1)) +
                          (ny1 - ny2) * vR1 *        (-sin(R1) * x1 - cos(R1) * y1))) / rotation_normalization

    d_dDdR2timesD_d_e = (((d_nx2_d_e - d_nx1_d_e) * ((-sin(R2) * x2 - cos(R2) * y2)) +
                          (nx2 - nx1) * vR2 *        (-cos(R2) * x2 + sin(R2) * y2)) +
                         ((d_ny2_d_e - d_ny1_d_e) * (( cos(R2) * x2 - sin(R2) * y2)) +
                          (ny2 - ny1) * vR2 *        (-sin(R2) * x2 - cos(R2) * y2))) / rotation_normalization

    # d (dDdr * D / D) / de
    d_dDdR1_d_e = (D * d_dDdR1timesD_d_e - dDdR1timesD * d_D_d_e) / Dsq
    d_dDdR2_d_e = (D * d_dDdR2timesD_d_e - dDdR2timesD * d_D_d_e) / Dsq


    d_dDdTx1_d_e = (D * (d_nx1_d_e - d_nx2_d_e) - (nx1 - nx2) * d_D_d_e) / Dsq
    d_dDdTy1_d_e = (D * (d_ny1_d_e - d_ny2_d_e) - (ny1 - ny2) * d_D_d_e) / Dsq

    d_dDdTx2_d_e = (D * (d_nx2_d_e - d_nx1_d_e) - (nx2 - nx1) * d_D_d_e) / Dsq
    d_dDdTy2_d_e = (D * (d_ny2_d_e - d_ny1_d_e) - (ny2 - ny1) * d_D_d_e) / Dsq

    return (d_dDdR1_d_e.sum(), d_dDdTx1_d_e.sum(), d_dDdTy1_d_e.sum(),
            d_dDdR2_d_e.sum(), d_dDdTx2_d_e.sum(), d_dDdTy2_d_e.sum())


if __name__ == '__main__':
    x1 = np.random.uniform(-100, 100, 100)
    y1 = np.random.uniform(-100, 100, 100)
    x2 = np.random.uniform(-100, 100, 100)
    y2 = np.random.uniform(-100, 100, 100)
    params = np.random.uniform(-1.0, 1.0, 6)

    normalization = 10000.0

    def call_f(p):
        R1, Tx1, Ty1, R2, Tx2, Ty2 = p
        return f_fprime(x1, y1, R1, Tx1, Ty1,
                        x2, y2, R2, Tx2, Ty2, normalization)

    def Dtrans(p):
        R1, Tx1, Ty1, R2, Tx2, Ty2 = p
        R1 /= normalization
        R2 /= normalization
        nx1 = cos(R1) * x1 - sin(R1) * y1 + Tx1
        ny1 = sin(R1) * x1 + cos(R1) * y1 + Ty1
        nx2 = cos(R2) * x2 - sin(R2) * y2 + Tx2
        ny2 = sin(R2) * x2 + cos(R2) * y2 + Ty2
        return sqrt((nx1 - nx2)**2 + (ny1-ny2)**2 + 1).sum()



    step = 0.00001
    print "sym"
    for v in call_f(params):
        print v,
    print

    print "step"
    print Dtrans(params),

    for idx in range(6):
        s = np.zeros(6)
        s[idx] = step
        upf = call_f(params + s)[0]
        downf = call_f(params - s)[0]
        print (upf - downf) / (2 * step),
    print

    print "Hsym D"
    v = np.random.uniform(-1, 1, (6,))
    R1, Tx1, Ty1, R2, Tx2, Ty2 = params
    for k in Hv(x1, y1, R1, Tx1, Ty1, v[0], v[1], v[2],
                x2, y2, R2, Tx2, Ty2, v[3], v[4], v[5], normalization):
        print k,
    print


    print "Hsym step"
    vals1 = f_fprime(x1, y1, R1 + step * v[0], Tx1 + step * v[1], Ty1 + step * v[2],
                     x2, y2, R2 + step * v[3], Tx2 + step * v[4], Ty2 + step * v[5], normalization)

    vals2 = f_fprime(x1, y1, R1 - step * v[0], Tx1 - step * v[1], Ty1 - step * v[2],
                     x2, y2, R2 - step * v[3], Tx2 - step * v[4], Ty2 - step * v[5], normalization)

    for idx, (v1, v2) in enumerate(zip(vals1[1:], vals2[1:])):
        print (v1 - v2) / (2 * step),
    print
