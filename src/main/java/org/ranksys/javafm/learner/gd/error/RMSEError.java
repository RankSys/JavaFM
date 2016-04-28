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
 * RMSE of the prediction with respect to the target.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class RMSEError<I extends FMInstance> implements PointWiseFMError<I> {

    @Override
    public double error(FM<I> fm, I x) {
        double e = fm.prediction(x) - x.getTarget();
        return e * e;
    }

    @Override
    public double dError(FM<I> fm, I x) {
        return fm.prediction(x) - x.getTarget();
    }

}
