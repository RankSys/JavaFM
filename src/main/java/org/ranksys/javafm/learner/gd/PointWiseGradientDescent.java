/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import java.util.logging.Level;
import java.util.logging.Logger;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.error.PointWiseFMError;

/**
 * Stochastic gradient descent learner.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instances
 */
public class PointWiseGradientDescent<I extends FMInstance> implements FMLearner<I> {

    private static final Logger LOG = Logger.getLogger(PointWiseGradientDescent.class.getName());

    private final double learnRate;
    private final int numIter;
    private final PointWiseFMError<I> trainError;
    private final PointWiseFMError<I> testError;
    private final double regB;
    private final double[] regW;
    private final double[] regM;

    /**
     * Constructor.
     *
     * @param learnRate learning rate
     * @param numIter number of iterations
     * @param batchSize batch size
     * @param error prediction error and gradient
     */
    public PointWiseGradientDescent(double learnRate, int numIter, PointWiseFMError<I> trainError, PointWiseFMError<I> testError, double regB, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.numIter = numIter;
        this.trainError = trainError;
        this.testError = testError;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    private double error(FM<I> fm, PointWiseFMError<I> error, FMData<I> test) {
        double err = test.stream()
                .mapToDouble(x -> error.error(fm, x))
                .average().getAsDouble();

        return err;
    }

    @Override
    public void learn(FM<I> fm, FMData<I> train, FMData<I> test) {
            LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, trainError, train), error(fm, testError, test)));
            
        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.stream().forEach(x -> {
                double b = fm.getB();
                double[] w = fm.getW();
                double[][] m = fm.getM();

                double lambda = trainError.dError(fm, x);

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

            LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, trainError, train), error(fm, testError, test)));
        }

    }
}
