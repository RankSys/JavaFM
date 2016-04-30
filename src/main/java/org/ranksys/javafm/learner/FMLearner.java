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

/**
 * Learner of factorisation machines.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface FMLearner<D extends FMData> {

    public double error(FM fm, D test);

    /**
     * Learns a pre-initialised factorisation machine. Reports the error on the test set.
     *
     * @param fm pre-initialised factorisation machine
     * @param train train set
     * @param test test set
     */
    public void learn(FM fm, D train, D test);

    /**
     * Learns a pre-initialised factorisation machine. Reports the error on the train set.
     *
     * @param fm pre-initialised factorisation machine
     * @param train train set
     */
    public default void learn(FM fm, D train) {
        learn(fm, train, train);
    }
}
