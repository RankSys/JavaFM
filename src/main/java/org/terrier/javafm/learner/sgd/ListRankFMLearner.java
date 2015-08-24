package org.terrier.javafm.learner.sgd;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.HashMap;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import org.terrier.javafm.FMData;
import org.terrier.javafm.FM;
import org.terrier.javafm.FMInstance;
import org.terrier.javafm.NormFMInstance;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public class ListRankFMLearner extends SGDFMLearner {

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

    private synchronized double norm1(FMData train, int norm) {
        Int2DoubleMap norm1 = norm1Map.get(train);
        if (norm1 == null) {
            norm1 = new Int2DoubleOpenHashMap();
            for (int i = 0; i < train.numFeatures(); i++) {
                double v = train.stream(i)
                        .mapToDouble(x2 -> exp(x2.getTarget()))
                        .sum();
                norm1.put(i, v);
            }
            norm1Map.put(train, norm1);
        }

        return norm1.get(norm);
    }

    private static double norm2(FMData train, FM fm, int norm) {
        return train.stream(norm)
                .mapToDouble(x2 -> exp(fm.prediction(x2)))
                .sum();
    }

    @Override
    protected double localError(FM fm, FMInstance x, FMData test) {
        int norm = ((NormFMInstance) x).getNorm();

        double p = exp(x.getTarget()) / norm1(test, norm);
        double q = exp(fm.prediction(x)) / norm2(test, fm, norm);
        return -p * log(q) + p * log(p);
    }

    @Override
    protected void gradientDescent(FM fm, double alpha, FMInstance x, FMData train) {
        DenseDoubleMatrix1D w = fm.getW();
        DenseDoubleMatrix2D m = fm.getM();

        int norm = ((NormFMInstance) x).getNorm();

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
