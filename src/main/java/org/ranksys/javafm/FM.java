/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import java.util.Random;
import java.util.function.DoubleBinaryOperator;

/**
 * Factorisation machine.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class FM {

    private static final DoubleBinaryOperator SUM = (x, y) -> x + y;

    private final int K;
    private double b;
    private final double[] w;
    private final double[][] m;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FM(double b, double[] w, double[][] m) {
        // TODO: this line below could be dangerous
        this.K = m[0].length;
        this.b = b;
        this.w = w;
        this.m = m;
    }

    public FM(int numFeatures, int K, Random rnd, double sdev) {
        this.K = K;
        this.b = 0.0;
        this.w = new double[numFeatures];
        this.m = new double[numFeatures][K];
        for (double[] mi : m) {
            for (int j = 0; j < mi.length; j++) {
                mi[j] = rnd.nextGaussian() * sdev;
            }
        }
    }

    private double dotProduct(double[] x, double[] y) {
        double product = 0.0;
        for (int i = 0; i < x.length; i++) {
            product += x[i] * y[i];
        }

        return product;
    }

    /**
     * Predict the value of an instance.
     *
     * @param x instance
     * @return value of prediction
     */
    public double predict(FMInstance x) {
        double pred = b;

        double[] xm = new double[m[0].length];
        pred += x.operate((i, xi) -> {
            for (int j = 0; j < xm.length; j++) {
                xm[j] += xi * m[i][j];
            }

            return xi * w[i] - 0.5 * xi * xi * dotProduct(m[i], m[i]);
        }, SUM);

        pred += 0.5 * dotProduct(xm, xm);

        return pred;
    }

    public int getK() {
        return K;
    }

    /**
     * Get bias.
     *
     * @return bias
     */
    public double getB() {
        return b;
    }

    /**
     * Set bias.
     *
     * @param b bias
     */
    public void setB(double b) {
        this.b = b;
    }

    /**
     * Get feature weight vector.
     *
     * @return feature weight vector
     */
    public double[] getW() {
        return w;
    }

    /**
     * Get feature interaction matrix.
     *
     * @return feature interaction matrix
     */
    public double[][] getM() {
        return m;
    }
}
