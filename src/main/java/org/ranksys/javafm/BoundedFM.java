/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import static java.lang.Math.max;
import static java.lang.Math.min;
import org.ranksys.javafm.instance.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class BoundedFM<I extends FMInstance> extends FM<I> {

    private final double min;
    private final double max;
    
    public BoundedFM(double b, double[] w, double[][] m, double min, double max) {
        super(b, w, m);
        this.min = min;
        this.max = max;
    }

    @Override
    public double prediction(I x) {
        return min(max, max(min, super.prediction(x)));
    }

    @Override
    public double prediction(I x, int i, double xi) {
        return min(max, max(min, super.prediction(x, i, xi)));
    }

}
