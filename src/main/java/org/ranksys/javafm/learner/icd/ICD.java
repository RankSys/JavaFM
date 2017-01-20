package org.ranksys.javafm.learner.icd;

import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.SeparableFMData;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.PointWiseError;

import java.util.Arrays;
import java.util.logging.Logger;

public class ICD implements FMLearner<SeparableFMData> {

    private static final Logger LOG = Logger.getLogger(ICD.class.getName());

    private final PointWiseError error = PointWiseError.rmse();
    private final int numIter;
    private final double regB;
    private final double[] regW;
    private final double[] regM;

    public ICD(int numIter, double regB, double[] regW, double[] regM) {
        this.numIter = numIter;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    @Override
    public double error(FM fm, SeparableFMData test) {
        return test.stream()
                .mapToDouble(x -> error.error(fm, x))
                .average().getAsDouble();
    }

    @Override
    public void learn(FM fm, SeparableFMData train, SeparableFMData test) {
        throw new UnsupportedOperationException("WORK IN PROGRESS");

//        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));
//
//        int K = fm.getK();
//        double[] w = fm.getW();
//        double[][] m = fm.getM();
//
//        for (int t = 1; t <= numIter; t++) {
//            long time0 = System.nanoTime();
//
//            for (int k = 0; k < K; k++) {
//
//                // compute something?
//
//                double[] JIk_ = JIk_(k, fm, train);
//
//                train.streamPart(1).forEach(xzs -> {
//                    FMInstance x = xzs.v1;
//                    Stream<? extends FMInstance> zs = xzs.v2;
//
//                    double dL;
//                    double ddL;
//                    double dR;
//                    double ddR;
//
//
//                });
//
//
//
////                double[] Ju = Ju(k, fm, train);
//            }
//            // k = K
//
//            // k = K + 1
//
//            int iter = t;
//            long time1 = System.nanoTime() - time0;
//
//            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
//            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
//        }

    }

    private double[] JUk_(int k, FM fm, SeparableFMData train) {
        return Jk_(k, 0, fm, train);
    }

    private double[] JIk_(int k, FM fm, SeparableFMData train) {
        return Jk_(k, 1, fm, train);
    }

    private double[] Jk_(int k, int part, FM fm, SeparableFMData train) {
        int K = fm.getK();
        double[][] m = fm.getM();
        int N = (int) train.streamPart(part).count();
        double[] psik = psi(k, part, fm, train);

        double[] Jk_ = new double[K + 2];
        for (int k2 = 0; k2 < K; k2++) {
            Jk_[k2] = product(psik, psi(k2, part, fm, train));
        }

        return Jk_;
    }

    private double[] psi(int k, int part, FM fm, SeparableFMData train) {
        int K = fm.getK();
        double[][] m = fm.getM();
        int N = (int) train.streamPart(part).count();
        double[] psik = new double[N];

        if (k < K) {

        } else if (k == K) {
            if (part == 0) {

            } else {
                Arrays.fill(psik, 1.0);
            }
        } else {
            if (part == 0) {
                Arrays.fill(psik, 1.0);
            } else {

            }
        }

        return psik;
    }

    private double product(double[] x, double[] y) {
        double p = 0.0;
        for (int i = 0; i < x.length; i++) {
            p += x[i] * y[i];
        }

        return p;
    }

}
