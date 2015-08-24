package org.terrier.javafm;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public class PairedFMInstance extends FMInstance {

    private final int p;
    private final double xp;
    private final int n;
    private final double xn;

    public PairedFMInstance(int p, double xp, int n, double xn, double target, Int2DoubleMap features) {
        super(target, features);
        this.p = p;
        this.xp = xp;
        this.n = n;
        this.xn = xn;
    }

    public PairedFMInstance(int p, double xp, int n, double xn, double target, int[] k, double[] v) {
        super(target, k, v);
        this.p = p;
        this.xp = xp;
        this.n = n;
        this.xn = xn;
    }

    public int getP() {
        return p;
    }

    public double getXp() {
        return xp;
    }

    public int getN() {
        return n;
    }

    public double getXn() {
        return xn;
    }

}
