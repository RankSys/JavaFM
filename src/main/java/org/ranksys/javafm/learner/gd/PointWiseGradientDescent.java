/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import java.util.logging.Logger;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.data.FMData;

/**
 * Stochastic gradient descent learner.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class PointWiseGradientDescent implements FMLearner<FMData> {

    private static final Logger LOG = Logger.getLogger(PointWiseGradientDescent.class.getName());

    private final double learnRate;
    private final int numIter;
    private final PointWiseError error;
    private final double regB;
    private final double[] regW;
    private final double[] regM;

    public PointWiseGradientDescent(double learnRate, int numIter, PointWiseError error, double regB, double[] regW, double[] regM) {
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
        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            train.stream().forEach(x -> {
                double b = fm.getB();
                double[] w = fm.getW();
                double[][] m = fm.getM();

                double lambda = error.dError(fm, x);

                fm.setB(b - learnRate * (lambda + regB * b));

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

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

    }
}
