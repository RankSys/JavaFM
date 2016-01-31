/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner;

import org.ranksys.javafm.FMData;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface FMLearner<I extends FMInstance> {

    public double error(FM<I> fm, FMData<I> test);

    public FM<I> learn(int K, FMData<I> train, FMData<I> test);

    public default FM<I> learn(int K, FMData<I> train) {
        return learn(K, train, train);
    }

    public void learn(FM<I> fm, FMData<I> train, FMData<I> test);

    public default void learn(FM<I> fm, FMData<I> train) {
        learn(fm, train, train);
    }
}
