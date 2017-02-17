/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.learner.FMLearner;

import java.util.Arrays;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Stochastic gradient descent learner.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ParallelPointWiseGradientDescent implements FMLearner<FMData> {

    private static final Logger LOG = Logger.getLogger(ParallelPointWiseGradientDescent.class.getName());

    private final double learnRate;
    private final int numIter;
    private final PointWiseError error;
    private final double regB;
    private final double[] regW;
    private final double[] regM;

    public ParallelPointWiseGradientDescent(double learnRate, int numIter, PointWiseError error, double regB, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.numIter = numIter;
        this.error = error;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    @Override
    public double error(FM fm, FMData test) {
        return test.stream()
                .mapToDouble(x -> error.error(fm, x))
                .average().getAsDouble();
    }

    @Override
    public void learn(FM fm, FMData train, FMData test) {
        LOG.info(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        int L = 10;

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            FM[] fms = new FM[L];
            for (int l = 0; l < L; l++) {
                fms[l] = fm.copy();
            }

            IntStream.range(0, L).parallel().forEach(l -> {
                train.stream().filter(x -> x.hashCode() % L == l).forEach(x -> {
//            int l = 0;
//            train.stream().forEach(x -> {
                    double[] b = fms[l].getB();
                    double[] w = fms[l].getW();
                    double[][] m = fms[l].getM();

                    double lambda = error.dError(fms[l], x);

                    b[0] -= learnRate * (lambda + regB * b[0]);

                    double[] xm = new double[m[0].length];
                    x.consume((i, xi) -> {
                        for (int j = 0; j < xm.length; j++) {
                            xm[j] += xi * m[i][j];
                        }

                        w[i] -= learnRate * (lambda * xi + regW[i] * w[i]);
                    });

                    x.consume((i, xi) -> {
                        for (int j = 0; j < m[i].length; j++) {
                            m[i][j] -= learnRate * (lambda * xi * xm[j]
                                    - lambda * xi * xi * m[i][j]
                                    + regM[i] * m[i][j]);
                        }
                    });
                });
            });

            mean(fm, fms);

//            double e0 = error(fm, train);
//            double[] es = Stream.of(fms).mapToDouble(fm1 -> error(fm1, train)).toArray();

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.info(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

    }

    private static void mean(FM fm0, FM[] fms) {
        int L = fms.length;

        fm0.getB()[0] = 0.0;
        Arrays.fill(fm0.getW(), 0.0);
        for (int i = 0; i < fm0.getM().length; i++) {
            Arrays.fill(fm0.getM()[i], 0.0);
        }

        for (FM fm1 : fms) {
            fm0.getB()[0] += fm1.getB()[0] / L;

            for (int i = 0; i < fm1.getW().length; i++) {
                fm0.getW()[i] += fm1.getW()[i] / L;

                for (int j = 0; j < fm1.getM()[i].length; j++) {
                    fm0.getM()[i][j] += fm1.getM()[i][j] / L;
                }
            }
        }
    }
}
