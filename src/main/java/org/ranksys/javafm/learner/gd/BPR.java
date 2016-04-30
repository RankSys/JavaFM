/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.logging.Logger;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FMInstance;
import org.ranksys.javafm.learner.FMLearner;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class BPR implements FMLearner<FMData> {

    private static final Logger LOG = Logger.getLogger(BPR.class.getName());

    private final double learnRate;
    private final int numIter;
    private final double[] regW;
    private final double[] regM;

    public BPR(double learnRate, int numIter, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.numIter = numIter;
        this.regW = regW;
        this.regM = regM;
    }

    private int[] uij(FMInstance x) {
        int[] uij = new int[3];
        x.consume((i, xi) -> {
            uij[(int) xi - 1] = i;
        });

        return uij;
    }

    @Override
    public double error(FM fm, FMData test) {
        return test.stream()
                .mapToDouble(x -> {
                    int[] uij = uij(x);
                    int i = uij[1];
                    int j = uij[2];

                    double sij = fm.prediction(x, i, 1.0) - fm.prediction(x, j, 1.0);
                    return log(1 / (1 + exp(-sij)));
                })
                .average().getAsDouble();
    }

    @Override
    public void learn(FM fm, FMData train, FMData test) {
        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            train.stream().forEach(x -> {
                double[] w = fm.getW();
                double[][] m = fm.getM();

                int[] uij = uij(x);
                int u = uij[0];
                int i = uij[1];
                int j = uij[2];

                double sij = fm.prediction(x, i, 1.0) - fm.prediction(x, j, 1.0);
                double lambda = 1 / (1 + exp(sij));

                w[i] -= learnRate * (-lambda + regW[i] * w[i]);
                w[j] -= learnRate * (+lambda + regW[j] * w[j]);

                for (int l = 0; l < m[u].length; l++) {
                    m[i][l] -= learnRate * (-lambda * m[u][l] + regM[i] * m[i][l]);
                    m[j][l] -= learnRate * (+lambda * m[u][l] + regM[j] * m[j][l]);
                    m[u][l] -= learnRate * (-lambda * m[i][l] + lambda * m[j][l] + regM[u] * m[u][l]);
                }
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }
    }
}
