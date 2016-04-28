/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.stream.Stream;
import org.ranksys.javafm.instance.NormFMInstance;

/**
 * Subclass of ArrayList implementing the FMData interface.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instance
 */
public class NormFMData<I extends NormFMInstance> implements FMData<I> {

    private final Int2ObjectOpenHashMap<Collection<I>> map = new Int2ObjectOpenHashMap<>();
    private final ArrayList<I> list = new ArrayList<>();
    private final int numFeatures;
    private final Random rnd;

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     */
    public NormFMData(int numFeatures, Random rnd) {
        this.numFeatures = numFeatures;
        this.rnd = rnd;
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     */
    public NormFMData(int numFeatures) {
        this(numFeatures, new Random());
    }

    public boolean add(I x) {
        boolean b1 = map.computeIfAbsent(x.getNorm(), i_ -> new ArrayList<>()).add(x);
        boolean b2 = list.add(x);
        
        return b1 && b2;
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
    public Stream<I> stream() {
        return list.stream();
    }

    @Override
    public Stream<I> stream(int i) {
        return map.get(i).stream();
    }

    @Override
    public Stream<I> sample(int n) {
        return rnd.ints(n, 0, numInstances())
                .mapToObj(i -> list.get(i));
    }

}
