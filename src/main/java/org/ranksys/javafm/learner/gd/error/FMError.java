/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd.error;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import java.util.concurrent.atomic.DoubleAdder;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.instance.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface FMError<I extends FMInstance> {

    /**
     * Local prediction error of an instance.
     *
     * @param fm factorisation machine
     * @param x instance
     * @param test test set
     * @return local prediction error
     */
    public abstract double localError(FM<I> fm, I x, FMData<I> test);

    /**
     * Adds the local gradient of an instance to the input gradient.
     *
     * @param gradB global bias gradient
     * @param gradW feature weight vector gradient
     * @param gradM feature interaction matrix gradient
     * @param fm factorisation machine
     * @param x instance
     * @param train training set
     */
    public abstract void localGradient(DoubleAdder gradB, DoubleMatrix1D gradW, DoubleMatrix2D gradM, FM<I> fm, I x, FMData<I> train);

}
