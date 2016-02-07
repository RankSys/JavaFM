/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd.error;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.instance.NormFMInstance;

/**
 * Cross-entropy of the probability of selection of the instances in training having a common norm feature. It is based on the ListRankMF algorithm by Shi et al. 2010 @RecSys.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ListRankFMError implements FMError<NormFMInstance> {

    private final double lambdaB;
    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    /**
     * Constructor.
     *
     * @param lambda regularisation parameter
     */
    public ListRankFMError(double lambda) {
        this(lambda, i -> lambda, i -> lambda);
    }

    /**
     * Constructor.
     *
     * @param lambdaB regularisation parameter for the global bias
     * @param lambdaW regularisation parameters for the feature weights vector
     * @param lambdaM regularisation parameters for the feature interactions matrix
     */
    public ListRankFMError(double lambdaB, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        this.lambdaB = lambdaB;
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    private final Map<FMData<NormFMInstance>, Int2DoubleMap> targetNormMap = new HashMap<>();

    private static Int2DoubleMap computeTargetNorm(FMData<NormFMInstance> data) {
        int[] norms = data.stream()
                .mapToInt(x -> x.getNorm())
                .distinct()
                .toArray();

        Int2DoubleMap norm1 = new Int2DoubleOpenHashMap();
        for (int norm : norms) {
            double v = data.stream(norm)
                    .mapToDouble(x2 -> exp(x2.getTarget()))
                    .sum();
            norm1.put(norm, v);
        }

        return norm1;
    }

    private synchronized double getTargetNorm(FMData<NormFMInstance> train, int norm) {
        return targetNormMap.computeIfAbsent(train, data -> computeTargetNorm(data)).get(norm);
    }

    private static double getPredictionNorm(FMData<NormFMInstance> data, FM<NormFMInstance> fm, int norm) {
        return data.stream(norm)
                .mapToDouble(x2 -> exp(fm.prediction(x2)))
                .sum();
    }

    @Override
    public double localError(FM<NormFMInstance> fm, NormFMInstance x, FMData<NormFMInstance> test) {
        int norm = x.getNorm();

        double p = exp(x.getTarget()) / getTargetNorm(test, norm);
        double q = exp(fm.prediction(x)) / getPredictionNorm(test, fm, norm);
        return -p * log(q) + p * log(p);
    }

    @Override
    public void localGradient(DoubleAdder gradB, DoubleMatrix1D gradW, DoubleMatrix2D gradM, FM<NormFMInstance> fm, NormFMInstance x, FMData<NormFMInstance> train) {
        double b = fm.getB();
        DoubleMatrix1D w = fm.getW();
        DoubleMatrix2D m = fm.getM();

        int norm = x.getNorm();

        double p = exp(x.getTarget()) / getTargetNorm(train, norm);
        double q = exp(fm.prediction(x)) / getPredictionNorm(train, fm, norm);
        double err = (-p + q);

        gradB.add(err + lambdaB * b);

        DoubleMatrix1D xm = new DenseDoubleMatrix1D(m.columns());
        x.consume((i, xi) -> {
            double wi = w.getQuick(i);
            DoubleMatrix1D mi = m.viewRow(i);

            xm.assign(mi, (r, s) -> r + xi * s);

            gradW.setQuick(i, gradW.getQuick(i) + err * xi + lambdaW.applyAsDouble(i) * wi);
        });

        x.consume((i, xi) -> {
            DoubleMatrix1D mi = m.viewRow(i);

            gradM.viewRow(i)
                    .assign(xm, (r, s) -> r + err * xi * s)
                    .assign(mi, (r, s) -> r - err * xi * s)
                    .assign(mi, (r, s) -> r + lambdaM.applyAsDouble(i) * s);
        });
    }

}
