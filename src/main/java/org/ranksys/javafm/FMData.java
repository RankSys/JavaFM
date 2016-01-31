/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import java.util.stream.Stream;

/**
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public interface FMData<I extends FMInstance> {
    
    public int numInstances();
    
    public int numFeatures();
    
    public Stream<I> stream();
    
    public Stream<I> stream(int i);
    
    public Stream<I> sample(int n);
    
}
