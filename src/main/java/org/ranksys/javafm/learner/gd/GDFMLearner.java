/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import static java.lang.Math.sqrt;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.error.FMError;

/**
 * Gradient descent learner for factorisation machines.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> instance type
 */
public abstract class GDFMLearner<I extends FMInstance> implements FMLearner<I> {
    
    protected final double alpha;
    protected final FMError<I> error;

    public GDFMLearner(double alpha, FMError<I> error) {
        this.alpha = alpha;
        this.error = error;
    }

    @Override
    public FM<I> learn(int K, FMData<I> train, FMData<I> test) {
        double b = 0.0;
        DenseDoubleMatrix1D w = new DenseDoubleMatrix1D(train.numFeatures());
        DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(train.numFeatures(), K);
        DoubleFunction init = (double x) -> sqrt(1.0 / K) * Math.random();
        m.assign(init);
        FM<I> fm = new FM<>(b, w, m);
        learn(fm, train, test);
        return fm;
    }
    
}
