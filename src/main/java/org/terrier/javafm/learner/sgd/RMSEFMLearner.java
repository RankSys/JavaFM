package org.terrier.javafm.learner.sgd;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import java.util.function.IntToDoubleFunction;
import org.terrier.javafm.FMData;
import org.terrier.javafm.FM;
import org.terrier.javafm.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public class RMSEFMLearner extends SGDFMLearner {

    private final double lambdaB;
    private final IntToDoubleFunction lambdaW;
    private final IntToDoubleFunction lambdaM;

    public RMSEFMLearner(double alpha, double sampleFactor, double lambda) {
        this(alpha, sampleFactor, lambda, i -> lambda, i -> lambda);
    }

    public RMSEFMLearner(double alpha, double sampleFactor, double lambdaB, IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
        super(alpha, sampleFactor);
        this.lambdaB = lambdaB;
        this.lambdaW = lambdaW;
        this.lambdaM = lambdaM;
    }

    @Override
    protected double localError(FM fm, FMInstance x, FMData test) {
        double e = fm.prediction(x) - x.getTarget();
        return e * e;
    }

    @Override
    protected void gradientDescent(FM fm, double alpha, FMInstance x, FMData train) {
        DenseDoubleMatrix1D w = fm.getW();
        DenseDoubleMatrix2D m = fm.getM();

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
