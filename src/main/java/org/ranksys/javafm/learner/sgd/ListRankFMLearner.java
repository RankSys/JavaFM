/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.sgd;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.HashMap;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.NormFMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ListRankFMLearner extends SGDFMLearner<NormFMInstance> {

    private final double lambdaB;
    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    public ListRankFMLearner(double alpha, double sampleFactor, double lambda) {
        this(alpha, sampleFactor, lambda, i -> lambda, i -> lambda);
    }

    public ListRankFMLearner(double alpha, double sampleFactor, double lambdaB, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        super(alpha, sampleFactor);
        this.lambdaB = lambdaB;
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    private final Map<FMData, Int2DoubleMap> norm1Map = new HashMap<>();

    private static Int2DoubleMap getNorm1(FMData<NormFMInstance> data, int[] norms) {
        Int2DoubleMap norm1 = new Int2DoubleOpenHashMap();
        for (int norm : norms) {
            double v = data.stream(norm)
                    .mapToDouble(x2 -> exp(x2.getTarget()))
                    .sum();
            norm1.put(norm, v);
        }
        
        return norm1;
    }

    @Override
    public void learn(FM<NormFMInstance> fm, FMData<NormFMInstance> train) {
        int[] trainNorms = train.stream()
                .mapToInt(x -> x.getNorm())
                .distinct()
                .toArray();
        norm1Map.put(train, getNorm1(train, trainNorms));
        
        super.learn(fm, train);
    }

    @Override
    public void learn(FM<NormFMInstance> fm, FMData<NormFMInstance> train, FMData<NormFMInstance> test) {
        int[] trainNorms = train.stream()
                .mapToInt(x -> x.getNorm())
                .distinct()
                .toArray();
        norm1Map.put(train, getNorm1(train, trainNorms));
        int[] testNorms = test.stream()
                .mapToInt(x -> x.getNorm())
                .distinct()
                .toArray();
        norm1Map.put(test, getNorm1(test, testNorms));
        
        super.learn(fm, train, test);
    }

    private synchronized double norm1(FMData<NormFMInstance> train, int norm) {
        return norm1Map.get(train).get(norm);
    }

    private static double norm2(FMData<NormFMInstance> train, FM fm, int norm) {
        return train.stream(norm)
                .mapToDouble(x2 -> exp(fm.prediction(x2)))
                .sum();
    }

    @Override
    protected double localError(FM<NormFMInstance> fm, NormFMInstance x, FMData<NormFMInstance> test) {
        int norm = x.getNorm();

        double p = exp(x.getTarget()) / norm1(test, norm);
        double q = exp(fm.prediction(x)) / norm2(test, fm, norm);
        return -p * log(q) + p * log(p);
    }

    @Override
    protected void gradientDescent(FM<NormFMInstance> fm, double alpha, NormFMInstance x, FMData<NormFMInstance> train) {
        DoubleMatrix1D w = fm.getW();
        DoubleMatrix2D m = fm.getM();

        int norm = x.getNorm();

        double p = exp(x.getTarget()) / norm1(train, norm);
        double q = exp(fm.prediction(x)) / norm2(train, fm, norm);
        double err = (-p + q);

        double b = fm.getB();
        fm.setB(b - alpha * (err + lambdaB * b));

        DoubleMatrix1D xm = new DenseDoubleMatrix1D(m.columns());
        x.consume((i, xi) -> {
            double wi = w.getQuick(i);
            DoubleMatrix1D mi = m.viewRow(i);

            xm.assign(mi, (r, s) -> r + xi * s);

            w.setQuick(i, wi - alpha * (err * xi + lambdaW.applyAsDouble(i) * wi));
        });

        x.consume((i, xi) -> {
            DoubleMatrix1D mi = m.viewRow(i).copy();

            m.viewRow(i)
                    .assign(xm, (r, s) -> r - alpha * err * xi * s)
                    .assign(mi, (r, s) -> r + alpha * err * xi * s)
                    .assign(mi, (r, s) -> r - alpha * lambdaM.applyAsDouble(i) * s);
        });
    }

}
