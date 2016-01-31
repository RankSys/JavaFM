/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.sgd;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import static java.lang.Math.sqrt;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.FMLearner;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public abstract class SGDFMLearner<I extends FMInstance> implements FMLearner<I> {

    private static final Logger LOG = Logger.getLogger(SGDFMLearner.class.getName());
    private static final int NSTEPS = 20;

    private final double alpha;
    private final double sampleFactor;

    public SGDFMLearner(double alpha, double sampleFactor) {
        this.alpha = alpha;
        this.sampleFactor = sampleFactor;
    }

    @Override
    public double error(FM<I> fm, FMData<I> test) {

        double err = test.sample((int) (sampleFactor * test.numInstances() / NSTEPS))
                .mapToDouble(x -> localError(fm, x, test))
                .average().getAsDouble();

        return err;
    }

    @Override
    public FM<I> learn(int K, FMData<I> train, FMData<I> test) {
        double b = 0.0;
        DenseDoubleMatrix1D w = new DenseDoubleMatrix1D(train.numFeatures());
        DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(train.numFeatures(), K);
        DoubleFunction init = x -> sqrt(1.0 / K) * Math.random();
        m.assign(init);

        FM<I> fm = new FM(b, w, m);
        learn(fm, train, test);

        return fm;
    }

    @Override
    public void learn(FM<I> fm, FMData<I> train, FMData<I> test) {
        LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", 0, 0.0));
        LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        for (int t = 1; t <= NSTEPS; t++) {
            long time0 = System.nanoTime();

            train.sample((int) (sampleFactor * train.numInstances() / NSTEPS)).forEach(x -> {
                gradientDescent(fm, alpha, x, train);
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

    }

    protected abstract double localError(FM<I> fm, I x, FMData<I> test);

    protected abstract void gradientDescent(FM<I> fm, double alpha, I x, FMData<I> train);
}
