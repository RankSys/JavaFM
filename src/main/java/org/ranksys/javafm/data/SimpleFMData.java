/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.ranksys.javafm.FMInstance;

/**
 * Subclass of ArrayList implementing the FMData interface.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class SimpleFMData implements FMData {

    private final List<FMInstance> list;
    private final int numFeatures;
    private final Random rnd;

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     */
    public SimpleFMData(int numFeatures, Random rnd, List<FMInstance> instances) {
        this.numFeatures = numFeatures;
        this.rnd = rnd;
        this.list = new ArrayList<>(instances);
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     */
    public SimpleFMData(int numFeatures) {
        this(numFeatures, new Random(), new ArrayList<>());
    }
    
    public void add(FMInstance x) {
        list.add(x);
    }

    @Override
    public int numInstances() {
        return list.size();
    }

    @Override
    public int numFeatures() {
        return numFeatures;
    }

    @Override
    public void shuffle() {
        Collections.shuffle(list, rnd);
    }

    @Override
    public Stream<? extends FMInstance> stream() {
        return list.stream();
    }

}
