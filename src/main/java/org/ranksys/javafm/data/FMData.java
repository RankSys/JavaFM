/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import java.util.Random;
import org.ranksys.javafm.instance.FMInstance;
import java.util.stream.Stream;

/**
 * Collection of instances.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instances
 */
public interface FMData<I extends FMInstance> {
    
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
    
    /**
     * Returns a stream of all instances.
     *
     * @return stream of all instances
     */
    public Stream<I> stream();
    
    /**
     * Returns a stream of instances having non-zero values for the index of
     * a feature.
     *
     * @param i index of the feature.
     * @return stream of instances
     */
    public Stream<I> stream(int i);
    
    /**
     * Returns a sample of instances of size n
     *
     * @param n sample size
     * @return stream of instances
     */
    public Stream<I> sample(int n);
    
    /**
     * Returns a sample of instances of size n
     *
     * @param n sample size
     * @param rnd random number generator
     * @return stream of instances
     */
    public Stream<I> sample(int n, Random rnd);
    
}
