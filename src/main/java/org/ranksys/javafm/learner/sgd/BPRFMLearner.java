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
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.PairedFMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class BPRFMLearner extends SGDFMLearner<PairedFMInstance> {

    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    public BPRFMLearner(double alpha, double sampleFactor, double lambda) {
        this(alpha, sampleFactor, i -> lambda, i -> lambda);
    }

    public BPRFMLearner(double alpha, double sampleFactor, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        super(alpha, sampleFactor);
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    @Override
    protected double localError(FM<PairedFMInstance> fm, PairedFMInstance x, FMData<PairedFMInstance> test) {
        int p = x.getP();
        double xp = x.getXp();
        int n = x.getN();
        double xn = x.getXn();
        
        double pp = fm.prediction(x, p, xp);
        double np = fm.prediction(x, n, xn);

        return log(1 / (1 + exp(-(pp - np))));
    }

    @Override
    protected void gradientDescent(FM<PairedFMInstance> fm, double alpha, PairedFMInstance x, FMData<PairedFMInstance> train) {
        DoubleMatrix1D w = fm.getW();
        DoubleMatrix2D m = fm.getM();

        int p = x.getP();
        double xp = x.getXp();
        int n = x.getN();
        double xn = x.getXn();

        double pp = fm.prediction(x, p, xp);
        double np = fm.prediction(x, n, xn);

        double err = 1 / (1 + exp(pp - np));

        double wp = w.getQuick(p);
        double wn = w.getQuick(n);
        w.setQuick(p, wp + alpha * err * xp - alpha * lambdaW.applyAsDouble(p) * wp);
        w.setQuick(n, wn - alpha * err * xn - alpha * lambdaW.applyAsDouble(n) * wn);

        DoubleMatrix1D mp = m.viewRow(p).copy();
        DoubleMatrix1D mn = m.viewRow(n).copy();

        x.consume((i, xi) -> {
            DoubleMatrix1D mi = m.viewRow(i).copy();

            m.viewRow(p)
                    .assign(mi, (r, s) -> r + alpha * err * s * xi * xp)
                    .assign(mp, (r, s) -> r - alpha * lambdaM.applyAsDouble(p) * s);
            m.viewRow(n)
                    .assign(mi, (r, s) -> r - alpha * err * s * xi * xn)
                    .assign(mn, (r, s) -> r - alpha * lambdaM.applyAsDouble(n) * s);
            m.viewRow(i)
                    .assign(mp, (r, s) -> r + alpha * err * s * xi * xp)
                    .assign(mn, (r, s) -> r - alpha * err * s * xi * xn)
                    .assign(mi, (r, s) -> r - alpha * lambdaM.applyAsDouble(i) * s);
        });
    }

}
