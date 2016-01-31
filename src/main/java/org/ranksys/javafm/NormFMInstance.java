/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 * Normalised instance. For all instances having a non-zero value for the
 * indicated norm index, the target values are normalised in the learning
 * process.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class NormFMInstance extends FMInstance {

    private final int norm;

    /**
     * Constructor.
     *
     * @param norm index of norm feature
     * @param target target value
     * @param features sparse feature vector
     */
    public NormFMInstance(int norm, double target, Int2DoubleMap features) {
        super(target, features);
        this.norm = norm;
    }

    /**
     * Constructor.
     *
     * @param norm index of norm feature
     * @param target target value
     * @param k indices of non-zero features
     * @param v values of non-zero features
     */
    public NormFMInstance(int norm, double target, int[] k, double[] v) {
        super(target, k, v);
        this.norm = norm;
    }

    /**
     * Get index of norm feature.
     *
     * @return norm
     */
    public int getNorm() {
        return norm;
    }

}
