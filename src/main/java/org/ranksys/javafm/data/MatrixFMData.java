/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.data;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.ranksys.javafm.instance.FMInstance;

/**
 * Collection of instances backed by a matrix.
 *
 * @author Saúl Vargas (Saul.Vargas@mendeley.com)
 */
public class MatrixFMData implements FMData<FMInstance> {

    private final DenseDoubleMatrix1D targets;
    private final SparseDoubleMatrix2D features;
    private final Random rnd;

    /**
     * Constructor.
     *
     * @param targets target vector
     * @param matrix feature matrix
     */
    public MatrixFMData(DenseDoubleMatrix1D targets, SparseDoubleMatrix2D matrix) {
        this(targets, matrix, new Random());
    }

    /**
     * Constructor.
     *
     * @param targets target vector
     * @param matrix feature matrix
     * @param rnd random number generator for sampling
     */
    public MatrixFMData(DenseDoubleMatrix1D targets, SparseDoubleMatrix2D matrix, Random rnd) {
        this.targets = targets;
        this.features = matrix;
        this.rnd = rnd;
    }

    @Override
    public int numInstances() {
        return features.rows();
    }

    @Override
    public int numFeatures() {
        return features.columns();
    }

    @Override
    public Stream<FMInstance> stream() {
        return IntStream.range(0, numInstances())
                .mapToObj(i -> convert(targets.getQuick(i), features.viewRow(i)));
    }

    @Override
    public Stream<FMInstance> stream(int j) {
        DoubleMatrix1D col = features.viewColumn(j);
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        col.getNonZeros(indexList, valueList);

        return IntStream.range(0, indexList.size())
                .map(i -> indexList.getQuick(i))
                .mapToObj(i -> convert(targets.getQuick(i), features.viewRow(i)));
    }

    @Override
    public Stream<FMInstance> sample(int n) {
        return rnd.ints(n, 0, numInstances())
                .mapToObj(i -> convert(targets.getQuick(i), features.viewRow(i)));
    }

    private FMInstance convert(double target, DoubleMatrix1D row) {
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        row.getNonZeros(indexList, valueList);

        Int2DoubleMap f = new Int2DoubleOpenHashMap();
        for (int i = 0; i < indexList.size(); i++) {
            f.put(indexList.getQuick(i), valueList.getQuick(i));
        }

        return new FMInstance(target, f);
    }
}
