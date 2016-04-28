/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd.error;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.NormFMData;
import org.ranksys.javafm.instance.NormFMInstance;

/**
 * Cross-entropy of the probability of selection of the instances in training having a common norm feature. It is based on the ListRankMF algorithm by Shi et al. 2010 @RecSys.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ListRankError implements PointWiseFMError<NormFMInstance> {

    private final NormFMData<NormFMInstance> data;
    private final IntToDoubleFunction targetNorm;

    public ListRankError(NormFMData<NormFMInstance> data) {
        this.data = data;
        
        Int2DoubleOpenHashMap targetNormMap = new Int2DoubleOpenHashMap();
        int[] norms = data.stream()
                .mapToInt(x -> x.getNorm())
                .distinct()
                .toArray();

        for (int norm : norms) {
            double v = data.stream(norm)
                    .mapToDouble(x -> exp(x.getTarget()))
                    .sum();
            targetNormMap.put(norm, v);
        }
        
        this.targetNorm = targetNormMap::get;
    }

    private double getPredictionNorm(FM<NormFMInstance> fm, int norm) {
        return data.stream(norm)
                .mapToDouble(x -> exp(fm.prediction(x)))
                .sum();
    }

    @Override
    public double error(FM<NormFMInstance> fm, NormFMInstance x) {
        double p = exp(x.getTarget()) / targetNorm.applyAsDouble(x.getNorm());
        double q = exp(fm.prediction(x)) / getPredictionNorm(fm, x.getNorm());

        return -p * log(q) + p * log(p);
    }

    @Override
    public double dError(FM<NormFMInstance> fm, NormFMInstance x) {
        double p = exp(x.getTarget()) / targetNorm.applyAsDouble(x.getNorm());
        double q = exp(fm.prediction(x)) / getPredictionNorm(fm, x.getNorm());

        return (-p + q);
    }
}
