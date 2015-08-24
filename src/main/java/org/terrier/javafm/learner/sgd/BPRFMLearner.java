package org.terrier.javafm.learner.sgd;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.function.IntToDoubleFunction;
import org.terrier.javafm.FMData;
import org.terrier.javafm.FM;
import org.terrier.javafm.FMInstance;
import org.terrier.javafm.PairedFMInstance;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public class BPRFMLearner extends SGDFMLearner {

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
    protected double localError(FM fm, FMInstance x, FMData test) {
        double pp = fm.prediction(x, ((PairedFMInstance) x).getP(), ((PairedFMInstance) x).getXp());
        double np = fm.prediction(x, ((PairedFMInstance) x).getN(), ((PairedFMInstance) x).getXn());

        return log(1 / (1 + exp(-(pp - np))));
    }

    @Override
    protected void gradientDescent(FM fm, double alpha, FMInstance x, FMData train) {
        DenseDoubleMatrix1D w = fm.getW();
        DenseDoubleMatrix2D m = fm.getM();

        int p = ((PairedFMInstance) x).getP();
        double xp = ((PairedFMInstance) x).getXp();
        int n = ((PairedFMInstance) x).getN();
        double xn = ((PairedFMInstance) x).getXn();

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
