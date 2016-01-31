/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import java.util.stream.DoubleStream;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class FMInstance {

    private final double target;
    private final Int2DoubleMap features;

    public FMInstance(double target, Int2DoubleMap features) {
        this.features = features;
        this.target = target;
    }

    public FMInstance(double target, int[] k, double[] v) {
        this.features = new Int2DoubleOpenHashMap(k, v);
        this.target = target;
    }

    public double getTarget() {
        return target;
    }

    public double get(int i) {
        return features.get(i);
    }

    public void consume(IntDoubleConsumer consumer) {
        features.int2DoubleEntrySet()
                .forEach(e -> consumer.accept(e.getIntKey(), e.getDoubleValue()));
    }

    public DoubleStream operate(IntDoubleFunction function) {
        return features.int2DoubleEntrySet().stream()
                .mapToDouble(e -> function.apply(e.getIntKey(), e.getDoubleValue()));
    }

    @FunctionalInterface
    public static interface IntDoubleConsumer {

        public void accept(int i, double v);
    }

    @FunctionalInterface
    public static interface IntDoubleFunction {

        public double apply(int i, double v);
    }

}
