/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd.error;

import org.ranksys.javafm.FM;
import org.ranksys.javafm.instance.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface PointWiseFMError<I extends FMInstance> {

    /**
     * Local prediction error of an instance.
     *
     * @param fm factorisation machine
     * @param x instance
     * @param test test set
     * @return local prediction error
     */
    public abstract double error(FM<I> fm, I x);
    
    public abstract double dError(FM<I> fm, I x);

}
