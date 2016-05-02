/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap.Entry;
import java.util.List;
import java.util.stream.Stream;
import org.ranksys.javafm.FMInstance;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface ListWiseFMData extends FMData {

    public Stream<Entry<List<? extends FMInstance>>> streamByGroup();
    
}
