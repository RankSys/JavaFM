/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.stream.Stream;
import org.ranksys.javafm.instance.FMInstance;

/**
 * Subclass of ArrayList implementing the FMData interface.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instance
 */
public class ArrayListFMData<I extends FMInstance> extends ArrayList<I> implements FMData<I> {

    private final int numFeatures;
    private final Random rnd;

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     */
    public ArrayListFMData(int numFeatures, Random rnd) {
        this.numFeatures = numFeatures;
        this.rnd = rnd;
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     * @param c collection of instances
     */
    public ArrayListFMData(int numFeatures, Random rnd, Collection<? extends I> c) {
        super(c);
        this.numFeatures = numFeatures;
        this.rnd = rnd;
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     */
    public ArrayListFMData(int numFeatures) {
        this(numFeatures, new Random());
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param c collection of instances
     */
    public ArrayListFMData(int numFeatures, Collection<? extends I> c) {
        this(numFeatures, new Random(), c);
    }

    @Override
    public int numInstances() {
        return size();
    }

    @Override
    public int numFeatures() {
        return numFeatures;
    }

    @Override
    public Stream<I> stream() {
        return ((ArrayList<I>) this).stream();
    }

    @Override
    public Stream<I> stream(int i) {
        return stream().filter(instance -> instance.get(i) > 0);
    }

    @Override
    public Stream<I> sample(int n, Random rnd) {
        return rnd.ints(n, 0, numInstances())
                .mapToObj(i -> get(i));
    }

    @Override
    public Stream<I> sample(int n) {
        return sample(n, rnd);
    }

}
