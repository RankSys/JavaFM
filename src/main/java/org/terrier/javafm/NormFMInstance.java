package org.terrier.javafm;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public class NormFMInstance extends FMInstance {

    public final int norm;

    public NormFMInstance(int norm, double target, Int2DoubleMap features) {
        super(target, features);
        this.norm = norm;
    }

    public NormFMInstance(int norm, double target, int[] k, double[] v) {
        super(target, k, v);
        this.norm = norm;
    }

    public int getNorm() {
        return norm;
    }

}
