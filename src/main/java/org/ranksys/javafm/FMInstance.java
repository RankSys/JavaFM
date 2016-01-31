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
import static java.lang.Double.NaN;
import java.util.function.DoubleBinaryOperator;

/**
 * Instance.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class FMInstance {

    private final double target;
    private final Int2DoubleMap features;

    /**
     * Constructor.
     *
     * @param target target value
     * @param features sparse feature vector
     */
    public FMInstance(double target, Int2DoubleMap features) {
        this.features = features;
        this.target = target;
    }

    /**
     * Constructor.
     *
     * @param target target value
     * @param k indices of non-zero features
     * @param v values of non-zero features
     */
    public FMInstance(double target, int[] k, double[] v) {
        this.features = new Int2DoubleOpenHashMap(k, v);
        this.target = target;
    }

    /**
     * Get target.
     *
     * @return target
     */
    public double getTarget() {
        return target;
    }

    /**
     * Get value of feature
     *
     * @param i feature index
     * @return value of feature
     */
    public double get(int i) {
        return features.get(i);
    }

    /**
     * Performs a consumer action on index-value pairs of the non-zero
     * instance features.
     *
     * @param consumer int-double consumer
     */
    public void consume(FeatureConsumer consumer) {
        features.int2DoubleEntrySet()
                .forEach(e -> consumer.accept(e.getIntKey(), e.getDoubleValue()));
    }

    /**
     * Performs an operation on index-value pairs of the non-zero instance
     * features.
     *
     * @param function int-double function
     * @param reducer aggregating operator of the results of the function
     * @return stream of double values resulting from applying the function
     */
    public double operate(FeatureDoubleFunction function, DoubleBinaryOperator reducer) {
        return features.int2DoubleEntrySet().stream()
                .mapToDouble(e -> function.apply(e.getIntKey(), e.getDoubleValue()))
                .reduce(reducer).orElse(NaN);
    }

    /**
     * Consumer of instance features.
     */
    @FunctionalInterface
    public static interface FeatureConsumer {

        /**
         * Consume index-value pair of feature.
         *
         * @param i index
         * @param v value
         */
        public void accept(int i, double v);
    }

    /**
     * Function of instance features.
     */
    @FunctionalInterface
    public static interface FeatureDoubleFunction {

        /**
         * Apply function to index-value pair of feature.
         *
         * @param i index
         * @param v value
         * @return result of function
         */
        public double apply(int i, double v);
    }

}
