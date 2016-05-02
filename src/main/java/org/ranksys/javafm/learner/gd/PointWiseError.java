/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import java.util.function.DoubleBinaryOperator;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.FMInstance;
import static java.lang.Math.abs;
import static java.lang.Math.signum;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class PointWiseError {

    private final DoubleBinaryOperator error;
    private final DoubleBinaryOperator dError;

    public PointWiseError(DoubleBinaryOperator error, DoubleBinaryOperator dError) {
        this.error = error;
        this.dError = dError;
    }

    public double error(FM fm, FMInstance x) {
        return error.applyAsDouble(fm.predict(x), x.getTarget());
    }

    public double dError(FM fm, FMInstance x) {
        return dError.applyAsDouble(fm.predict(x), x.getTarget());
    }
    
    public static PointWiseError rmse() {
        return new PointWiseError((y, x) -> (y - x) * (y - x), (y, x) -> y - x);
    }

    public static PointWiseError mae() {
        return new PointWiseError((y, x) -> abs(y - x), (y, x) -> signum(y - x));
    }

}
