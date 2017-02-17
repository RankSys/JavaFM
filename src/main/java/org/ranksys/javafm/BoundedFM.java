/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import java.util.Arrays;
import java.util.Random;
import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class BoundedFM extends FM {

    private final double min;
    private final double max;

    public BoundedFM(double b, double[] w, double[][] m, double min, double max) {
        super(b, w, m);
        this.min = min;
        this.max = max;
    }

    public BoundedFM(double min, double max, int numFeatures, int K, Random rnd, double sdev) {
        super(numFeatures, K, rnd, sdev);
        this.min = min;
        this.max = max;
    }

    @Override
    public double predict(FMInstance x) {
        return min(max, max(min, super.predict(x)));
    }

    @Override
    public FM copy() {
        double[] copyW = Arrays.copyOf(getW(), getW().length);
        double[][] copyM = new double[getM().length][];
        for (int i = 0; i < getM().length; i++) {
            copyM[i] = Arrays.copyOf(getM()[i], getM()[i].length);
        }

        return new BoundedFM(getB()[0], copyW, copyM, min, max);
    }
}
