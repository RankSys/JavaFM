/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import it.unimi.dsi.fastutil.ints.AbstractInt2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap.Entry;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.ranksys.javafm.instance.FMInstance;

/**
 * Subclass of ArrayList implementing the FMData interface.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instance
 */
public class GroupFMData implements FMData {

    private final IntList groupList = new IntArrayList();
    private final IntSet groupSet = new IntOpenHashSet();
    private final Int2ObjectOpenHashMap<List<FMInstance>> map = new Int2ObjectOpenHashMap<>();
    private final int numFeatures;
    private final Random rnd;

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     */
    public GroupFMData(int numFeatures, Random rnd) {
        this.numFeatures = numFeatures;
        this.rnd = rnd;
    }

    /**
     * Constructor.
     *
     * @param numFeatures number of features
     */
    public GroupFMData(int numFeatures) {
        this(numFeatures, new Random());
    }

    public void add(FMInstance x, int group) {
        if (groupSet.add(group)) {
            groupList.add(group);
        }
        map.computeIfAbsent(group, i -> new ArrayList<>()).add(x);
    }

    @Override
    public int numInstances() {
        return map.values().stream().mapToInt(List::size).sum();
    }

    @Override
    public int numFeatures() {
        return numFeatures;
    }

    @Override
    public void shuffle() {
        Collections.shuffle(groupList, rnd);
    }

    @Override
    public Stream<? extends FMInstance> stream() {
        return groupList.stream()
                .flatMap(i -> map.get(i).stream());
    }

    public Stream<Entry<List<? extends FMInstance>>> streamByGroup() {
        return groupList.stream()
                .map(i -> new AbstractInt2ObjectMap.BasicEntry<>(i, map.get(i)));
    }
}
