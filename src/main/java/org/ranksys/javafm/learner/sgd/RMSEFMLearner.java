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
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;

/**
 * SGD learner that minimises the RMSE of the prediction with respect to
 * the target.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class RMSEFMLearner extends SGDFMLearner<FMInstance> {

    private final double lambdaB;
    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    /**
     * Constructor.
     *
     * @param alpha learning rate
     * @param sampleFactor proportion of training instance to be used for learning
     * @param lambda regularisation parameter
     */
    public RMSEFMLearner(double alpha, double sampleFactor, double lambda) {
        this(alpha, sampleFactor, lambda, i -> lambda, i -> lambda);
    }

    /**
     * Constructor.
     *
     * @param alpha learning rate
     * @param sampleFactor proportion of training instance to be used for learning
     * @param lambdaB regularisation parameter for the global bias
     * @param lambdaW regularisation parameters for the feature weights vector
     * @param lambdaM regularisation parameters for the feature interactions matrix
     */
    public RMSEFMLearner(double alpha, double sampleFactor, double lambdaB, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        super(alpha, sampleFactor);
        this.lambdaB = lambdaB;
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    @Override
    protected double localError(FM<FMInstance> fm, FMInstance x, FMData<FMInstance> test) {
        double e = fm.prediction(x) - x.getTarget();
        return e * e;
    }

    @Override
    protected void gradientDescent(FM<FMInstance> fm, double alpha, FMInstance x, FMData<FMInstance> train) {
        DoubleMatrix1D w = fm.getW();
        DoubleMatrix2D m = fm.getM();

        double err = fm.prediction(x) - x.getTarget();

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
