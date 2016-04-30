/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import org.ranksys.javafm.FMInstance;
import java.util.stream.Stream;

/**
 * Collection of instances.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface FMData {
    
    /**
     * Returns number of instances.
     *
     * @return number of instances
     */
    public int numInstances();
    
    /**
     * Returns Number of features of the instances.
     *
     * @return number of features
     */
    public int numFeatures();
    
    public void shuffle();
    
    /**
     * Returns a stream of all instances.
     *
     * @return stream of all instances
     */
    public Stream<? extends FMInstance> stream();
    
}
