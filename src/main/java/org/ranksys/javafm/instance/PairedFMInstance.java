/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.instance;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 * Paired instance. It contains actually two instance differing only in the
 * presence and absence of two features, called positive and negative feature.
 * This is useful for pair-wise learning.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class PairedFMInstance extends FMInstance {

    private final int p;
    private final double xp;
    private final int n;
    private final double xn;

    /**
     * Constructor.
     *
     * @param p index of positive feature
     * @param xp value of positive feature
     * @param n index of negative feature
     * @param xn value of negative feature
     * @param target target value
     * @param features sparse feature vector
     */
    public PairedFMInstance(int p, double xp, int n, double xn, double target, Int2DoubleMap features) {
        super(target, features);
        this.p = p;
        this.xp = xp;
        this.n = n;
        this.xn = xn;
    }

    /**
     * Constructor.
     *
     * @param p index of positive feature
     * @param xp value of positive feature
     * @param n index of negative feature
     * @param xn value of negative feature
     * @param target target value
     * @param k indices of non-zero features
     * @param v values of non-zero features
     */
    public PairedFMInstance(int p, double xp, int n, double xn, double target, int[] k, double[] v) {
        super(target, k, v);
        this.p = p;
        this.xp = xp;
        this.n = n;
        this.xn = xn;
    }

    /**
     * Get index of positive feature.
     *
     * @return index of positive feature
     */
    public int getP() {
        return p;
    }

    /**
     * Get value of positive feature.
     *
     * @return value of positive feature
     */
    public double getXp() {
        return xp;
    }

    /**
     * Get index of negative feature.
     *
     * @return index of negative feature
     */
    public int getN() {
        return n;
    }

    /**
     * Get value of negative feature.
     *
     * @return value of negative feature
     */
    public double getXn() {
        return xn;
    }

}
