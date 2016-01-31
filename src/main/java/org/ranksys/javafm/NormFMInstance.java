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
public class NormFMInstance extends FMInstance {

    private final int norm;

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
