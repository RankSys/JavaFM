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
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.instance.FMInstance;

/**
 * RMSE of the prediction with respect to the target.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class RMSEFMError implements FMError<FMInstance> {

    private final double lambdaB;
    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    /**
     * Constructor.
     *
     * @param lambda regularisation parameter
     */
    public RMSEFMError(double lambda) {
        this(lambda, i -> lambda, i -> lambda);
    }

    /**
     * Constructor.
     *
     * @param lambdaB regularisation parameter for the global bias
     * @param lambdaW regularisation parameters for the feature weights vector
     * @param lambdaM regularisation parameters for the feature interactions matrix
     */
    public RMSEFMError(double lambdaB, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        this.lambdaB = lambdaB;
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    @Override
    public double localError(FM<FMInstance> fm, FMInstance x, FMData<FMInstance> test) {
        double e = fm.prediction(x) - x.getTarget();
        return e * e;
    }

    @Override
    public void localGradient(DoubleAdder gradB, DoubleMatrix1D gradW, DoubleMatrix2D gradM, FM<FMInstance> fm, FMInstance x, FMData<FMInstance> train) {
        double b = fm.getB();
        DoubleMatrix1D w = fm.getW();
        DoubleMatrix2D m = fm.getM();

        double err = fm.prediction(x) - x.getTarget();

        gradB.add((err + lambdaB * b));

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
