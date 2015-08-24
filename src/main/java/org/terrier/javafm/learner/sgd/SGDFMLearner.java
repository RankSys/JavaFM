package org.terrier.javafm.learner.sgd;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import static java.lang.Math.sqrt;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.terrier.javafm.FMData;
import org.terrier.javafm.FM;
import org.terrier.javafm.FMInstance;
import org.terrier.javafm.learner.FMLearner;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public abstract class SGDFMLearner implements FMLearner {

    private static final Logger LOG = Logger.getLogger(SGDFMLearner.class.getName());

    private final double alpha;
    private final double sampleFactor;

    public SGDFMLearner(double alpha, double sampleFactor) {
        this.alpha = alpha;
        this.sampleFactor = sampleFactor;
    }

    @Override
    public double error(FM fm, FMData test) {

//        double err = test.sample((int) (sampleFactor * test.numInstances() / 20)).mapToDouble(x -> {
        double err = test.stream().mapToDouble(x -> {
            return localError(fm, x, test);
        }).average().getAsDouble();

        return err;
    }

    @Override
    public FM learn(int K, FMData train, FMData test) {
        double b = 0.0;
        DenseDoubleMatrix1D w = new DenseDoubleMatrix1D(train.numFeatures());
        DenseDoubleMatrix2D m = new DenseDoubleMatrix2D(train.numFeatures(), K);
        DoubleFunction init = x -> sqrt(1.0 / K) * Math.random();
        m.assign(init);

        FM fm = new FM(b, w, m);
        learn(fm, train, test);

        return fm;
    }

    @Override
    public void learn(FM fm, FMData train, FMData test) {
        LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", 0, 0.0));
        LOG.log(Level.INFO, () -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        for (int t = 1; t <= 20; t++) {
            long time0 = System.nanoTime();

            train.sample((int) (sampleFactor * train.numInstances() / 20)).forEach(x -> {
                gradientDescent(fm, alpha, x, train);
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.log(Level.INFO, () -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

    }

    protected abstract double localError(FM fm, FMInstance x, FMData test);

    protected abstract void gradientDescent(FM fm, double alpha, FMInstance x, FMData train);
}
