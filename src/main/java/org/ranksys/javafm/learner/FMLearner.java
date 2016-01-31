/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner;

import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;

/**
 * Learner of factorisation machines.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instance
 */
public interface FMLearner<I extends FMInstance> {

    /**
     * Prediction error of a factorisation machine in a test set.
     *
     * @param fm factorisation machine
     * @param test test set
     * @return error
     */
    public double error(FM<I> fm, FMData<I> test);

    /**
     * Learns a factorisation machine. Reports the error on the test set.
     *
     * @param K dimension of the feature interaction matrix
     * @param train train set
     * @param test test set
     * @return factorisation machine learned
     */
    public FM<I> learn(int K, FMData<I> train, FMData<I> test);

    /**
     * Learns a factorisation machine. Reports the error on the train set.
     *
     * @param K dimension of the feature interaction matrix
     * @param train train set
     * @return factorisation machine learned
     */
    public default FM<I> learn(int K, FMData<I> train) {
        return learn(K, train, train);
    }

    /**
     * Learns a pre-initialised factorisation machine. Reports the error on the
     * test set.
     *
     * @param fm pre-initialised factorisation machine
     * @param train train set
     * @param test test set
     */
    public void learn(FM<I> fm, FMData<I> train, FMData<I> test);

    /**
     * Learns a pre-initialised factorisation machine. Reports the error on the
     * train set.
     *
     * @param fm pre-initialised factorisation machine
     * @param train train set
     */
    public default void learn(FM<I> fm, FMData<I> train) {
        learn(fm, train, train);
    }
}
