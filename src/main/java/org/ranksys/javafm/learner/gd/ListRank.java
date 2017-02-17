/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap.Entry;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.FMInstance;
import org.ranksys.javafm.data.ListWiseFMData;
import org.ranksys.javafm.learner.FMLearner;

import java.util.List;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static java.lang.Math.log;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ListRank implements FMLearner<ListWiseFMData> {

    private static final Logger LOG = Logger.getLogger(ListRank.class.getName());

    private final double learnRate;
    private final int numIter;
    private final double regB;
    private final double[] regW;
    private final double[] regM;

    public ListRank(double learnRate, int numIter, double regB, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.numIter = numIter;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    private double[] getP(List<? extends FMInstance> group) {
        double[] p = group.stream()
                .mapToDouble(FMInstance::getTarget)
                .map(Math::exp)
                .toArray();

        double pNorm = DoubleStream.of(p).sum();
        for (int i = 0; i < p.length; i++) {
            p[i] /= pNorm;
        }

        return p;
    }

    private double[] getQ(FM fm, List<? extends FMInstance> group) {
        double[] q = group.stream()
                .mapToDouble(fm::predict)
                .map(Math::exp)
                .toArray();

        double qNorm = DoubleStream.of(q).sum();
        for (int i = 0; i < q.length; i++) {
            q[i] /= qNorm;
        }

        return q;
    }

    @Override
    public double error(FM fm, ListWiseFMData test) {
        return test.streamByGroup().map(Entry::getValue)
                .mapToDouble((List<? extends FMInstance> group) -> {
                    double[] p = getP(group);
                    double[] q = getQ(fm, group);

                    return IntStream.range(0, group.size())
                            .mapToDouble(i -> -p[i] * log(q[i]))
                            .sum();
                })
                .average().getAsDouble();
    }

    @Override
    public void learn(FM fm, ListWiseFMData train, ListWiseFMData test) {
        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            train.streamByGroup().map(Entry::getValue).forEach(group -> {
                double[] b = fm.getB();
                double[] w = fm.getW();
                double[][] m = fm.getM();

                double[] p = getP(group);
                double[] q = getQ(fm, group);

                for (int k = 0; k < group.size(); k++) {
                    FMInstance x = group.get(k);

                    double lambda = -p[k] + q[k];

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

                }
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }
    }

}
