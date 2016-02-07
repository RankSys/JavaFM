/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.gd.error.FMError;

/**
 * Stochastic gradient descent learner.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instances
 */
public class StochasticGDFMLearner<I extends FMInstance> extends GDFMLearner<I> {

    private static final Logger LOG = Logger.getLogger(StochasticGDFMLearner.class.getName());
    private static final int NSTEPS = 20;

    private final double sampleFactor;

    /**
     * Constructor.
     *
     * @param alpha learning rate
     * @param sampleFactor proportion of training instance to be used for learning
     * @param error prediction error and gradient
     */
    public StochasticGDFMLearner(double alpha, double sampleFactor, FMError<I> error) {
        super(alpha, error);
        this.sampleFactor = sampleFactor;
    }

    @Override
    public double error(FM<I> fm, FMData<I> test) {
        int n = (int) (sampleFactor * test.numInstances() / NSTEPS);

        double err = test.sample(n)
                .mapToDouble(x -> error.localError(fm, x, test))
                .average().getAsDouble();

        return err;
    }


    @Override
    public void learn(FM<I> fm, FMData<I> train, FMData<I> test) {
        LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", 0, 0.0));
        LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        int n = (int) (sampleFactor * train.numInstances() / NSTEPS);

        DoubleAdder gradB = new DoubleAdder();
        DoubleMatrix1D gradW = new DenseDoubleMatrix1D(fm.getW().size());
        DoubleMatrix2D gradM = new DenseDoubleMatrix2D(fm.getM().rows(), fm.getM().columns());
        
        for (int t = 1; t <= NSTEPS; t++) {
            long time0 = System.nanoTime();

            train.sample(n).forEach(x -> {
                gradB.reset();
                gradW.assign(0.0);
                gradM.assign(0.0);

                error.localGradient(gradB, gradW, gradM, fm, x, train);

                fm.setB(fm.getB() - alpha * gradB.doubleValue());
                fm.getW().assign(gradW, (r, s) -> r - alpha * s);
                fm.getM().assign(gradM, (r, s) -> r - alpha * s);
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

    }
}
