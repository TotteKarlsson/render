import numpy as np
cos = np.cos
sin = np.sin
sqrt = np.sqrt

def f_fprime(x1, y1, R1, Tx1, Ty1,
             x2, y2, R2, Tx2, Ty2):
    nx1 = cos(R1) * x1 - sin(R1) * y1 + Tx1
    ny1 = sin(R1) * x1 + cos(R1) * y1 + Ty1
    nx2 = cos(R2) * x2 - sin(R2) * y2 + Tx2
    ny2 = sin(R2) * x2 + cos(R2) * y2 + Ty2

    D = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2 + 1)
    dDdR = ((nx1 - nx2) * ((-sin(R1) * x1 - cos(R1) * y1) / D) + 
            (ny1 - ny2) * (( cos(R1) * x1 - sin(R1) * y1) / D))

    dDdTx =  (nx1 - nx2 + Tx1) / D
    dDdTy =  (ny1 - ny2 + Ty1) / D

    return D.sum(), dDdR.sum(), dDdTx.sum(), dDdTy.sum()


def Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
       x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
       e):

    nx1 = cos(R1 + e * vR1) * x1 - sin(R1 + e * vR1) * y1 + Tx1 + e * vTx1
    ny1 = sin(R1 + e * vR1) * x1 + cos(R1 + e * vR1) * y1 + Ty1 + e * vTy1
    nx2 = cos(R2 + e * vR2) * x2 - sin(R2 + e * vR2) * y2 + Tx2 + e * vTx2
    ny2 = sin(R2 + e * vR2) * x2 + cos(R2 + e * vR2) * y2 + Ty2 + e * vTy2

    d_nx1_d_e = vR1 * (-sin(R1 + e * vR1) * x1 - cos(R1 + e * vR1) * y1) + vTx1
    d_ny1_d_e = vR1 * ( cos(R1 + e * vR1) * x1 - sin(R1 + e * vR1) * y1) + vTy1
    d_nx2_d_e = vR2 * (-sin(R2 + e * vR2) * x2 - cos(R2 + e * vR2) * y2) + vTx2
    d_ny2_d_e = vR2 * ( cos(R2 + e * vR2) * x2 - sin(R2 + e * vR2) * y2) + vTy2

    Dsq = (nx1 - nx2)**2 + (ny1 - ny2)**2 + 1
    D = np.sqrt(Dsq)
    # F = (nx1 - nx2)**2 + (ny1 - ny2)**2 + 1
    # dD_de = d(sqrt(F)) / dF * (dF / dnx1 * dnx1 / de +
    #                            dF / dny1 * dny1 / de +
    #                            dF / dnx2 * dnx2 / de +
    #                            dF / dny2 * dny2 / de)
    d_D_d_e = (1 / 2.0) * (1 / D) * (2 * (nx1 - nx2) * d_nx1_d_e +
                                     2 * (ny1 - ny2) * d_ny1_d_e +
                                     2 * (nx2 - nx1) * d_nx2_d_e +
                                     2 * (ny2 - ny1) * d_ny2_d_e)
    
  
    dDdRtimesD = ((nx1 - nx2) * (-sin(R1) * x1 - cos(R1) * y1) + 
                  (ny1 - ny2) * ( cos(R1) * x1 - sin(R1) * y1))
  
    # d(dDdR * D) / de
    d_dDdRtimesD_d_e = (((d_nx1_d_e - d_nx2_d_e) * ((-sin(R1 + e * vR1) * x1 - cos(R1 + e * vR1) * y1)) +
                         (nx1 - nx2) * vR1 *        (-cos(R1 + e * vR1) * x1 + sin(R1 + e * vR1) * y1)) +
                        
                        ((d_ny1_d_e - d_ny2_d_e) * (( cos(R1 + e * vR1) * x1 - sin(R1 + e * vR1) * y1)) +
                         (ny1 - ny2) * vR1 *        (-sin(R1 + e * vR1) * x1 - cos(R1 + e * vR1) * y1)))
    

    # d (dDdr * D / D) / de
    d_dDdR_d_e = (D * d_dDdRtimesD_d_e - dDdRtimesD * d_D_d_e) / Dsq
    d_dDdTx_d_e = (D * (d_nx1_d_e - d_nx2_d_e + vTx1) - (nx1 - nx2 + Tx1 + e * vTx1) * d_D_d_e) / Dsq
    d_dDdTy_d_e = (D * (d_ny1_d_e - d_ny2_d_e + vTy1) - (ny1 - ny2 + Ty1 + e * vTy1) * d_D_d_e) / Dsq

    return d_dDdR_d_e.sum(), d_dDdTx_d_e.sum(), d_dDdTy_d_e.sum()






if __name__ == '__main__':
    x1 = np.random.uniform(0, 100, 100)
    y1 = np.random.uniform(0, 100, 100)
    x2 = np.random.uniform(0, 100, 100)
    y2 = np.random.uniform(0, 100, 100)
    R1 = 0.1 *0
    Tx1 = 15 *0
    Ty1 = 33 *0
    R2 = -.3 *0
    Tx2 = 66 *0
    Ty2 = -5 *0

    step = 0.00001
    print "sym"
    print f_fprime(x1, y1, R1, Tx1, Ty1,
                   x2, y2, R2, Tx2, Ty2)

    print
    print "step"
    print sqrt((x1 - x2)**2 + (y1-y2)**2 + 1).sum(),

    fm1 = f_fprime(x1, y1, R1 - step, Tx1, Ty1,
                   x2, y2, R2, Tx2, Ty2)
    fp1 = f_fprime(x1, y1, R1 + step, Tx1, Ty1,
                   x2, y2, R2, Tx2, Ty2)
    print (fp1[0] - fm1[0]) / (2 * step),

    fm1 = f_fprime(x1, y1, R1, Tx1 - step, Ty1,
                   x2, y2, R2, Tx2, Ty2)
    fp1 = f_fprime(x1, y1, R1, Tx1 + step, Ty1,
                   x2, y2, R2, Tx2, Ty2)
    print (fp1[0] - fm1[0]) / (2 * step),

    fm1 = f_fprime(x1, y1, R1, Tx1, Ty1 - step,
                   x2, y2, R2, Tx2, Ty2)
    fp1 = f_fprime(x1, y1, R1, Tx1, Ty1 + step,
                   x2, y2, R2, Tx2, Ty2)
    print (fp1[0] - fm1[0]) / (2 * step)

    print

    print "Hsym D"
    v = np.random.uniform(-1, 1, (6,))
    print Hv(x1, y1, R1, Tx1, Ty1, v[0], v[1], v[2],
             x2, y2, R2, Tx2, Ty2, v[3], v[4], v[5],
             0)

    print "Hsym step"
    vals1 = f_fprime(x1, y1, R1 + step * v[0], Tx1 + step * v[1], Ty1 + step * v[2],
                     x2, y2, R2 + step * v[3], Tx2 + step * v[4], Ty2 + step * v[5])

    vals2 = f_fprime(x1, y1, R1 - step * v[0], Tx1 - step * v[1], Ty1 - step * v[2],
                     x2, y2, R2 - step * v[3], Tx2 - step * v[4], Ty2 - step * v[5])

    for idx, (v1, v2) in enumerate(zip(vals1[1:], vals2[1:])):
        print (v1 - v2) / (2 * step),
    print
