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
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
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
