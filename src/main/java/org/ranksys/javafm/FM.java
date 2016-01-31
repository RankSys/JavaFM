/* 
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm;

import org.ranksys.javafm.instance.FMInstance;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Factorisation machine.
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 * @param <I> type of instance
 */
public class FM<I extends FMInstance> {

    private double b;
    private final DoubleMatrix1D w;
    private final DoubleMatrix2D m;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FM(double b, DoubleMatrix1D w, DoubleMatrix2D m) {
        this.b = b;
        this.w = w;
        this.m = m;
    }

    /**
     * Predict the value of an instance.
     *
     * @param x instance
     * @return value of prediction
     */
    public double prediction(I x) {
        double pred = b;

        DoubleMatrix1D xm = new DenseDoubleMatrix1D(m.columns());
        pred += x.operate((i, xi) -> {
            double wi = w.getQuick(i);
            DoubleMatrix1D mi = m.viewRow(i);

            xm.assign(mi, (r, s) -> r + xi * s);

            return xi * wi - 0.5 * xi * xi * mi.zDotProduct(mi);
        }, (v1, v2) -> v1 + v2);

        pred += 0.5 * xm.zDotProduct(xm);

        return pred;
    }

    /**
     * Feature-specific contribution to the prediction of the value of an
     * instance.
     *
     * @param x instance
     * @param i index of the feature of interest
     * @param xi value of the feature of interest
     * @return value of the contribution of the feature to the prediction
     */
    public double prediction(I x, int i, double xi) {
        double wi = w.getQuick(i);
        DoubleMatrix1D mi = m.viewRow(i);

        double pred = 0.0;

        pred += xi * wi;

        pred += x.operate((j, xj) -> {
            DoubleMatrix1D mj = m.viewRow(j);

            return xi * xj * mi.zDotProduct(mj);
        }, (v1, v2) -> v1 + v2);

        return pred;
    }

    /**
     * Get bias.
     *
     * @return bias
     */
    public double getB() {
        return b;
    }

    /**
     * Set bias.
     *
     * @param b bias
     */
    public void setB(double b) {
        this.b = b;
    }

    /**
     * Get feature weight vector.
     *
     * @return feature weight vector
     */
    public DoubleMatrix1D getW() {
        return w;
    }

    /**
     * Get feature interaction matrix.
     *
     * @return feature interaction matrix
     */
    public DoubleMatrix2D getM() {
        return m;
    }

    private static void saveDenseDoubleMatrix1D(OutputStream stream, DoubleMatrix1D vector) throws IOException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(stream));
        double[] v = vector.toArray();
        for (int j = 0; j < v.length; j++) {
            out.write(Double.toString(v[j]));
            out.newLine();
        }
        out.flush();
    }

    private static void saveDenseDoubleMatrix2D(OutputStream stream, DoubleMatrix2D matrix) throws IOException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(stream));
        double[][] m = matrix.toArray();
        for (double[] pu : m) {
            for (int j = 0; j < pu.length; j++) {
                out.write(Double.toString(pu[j]));
                if (j < pu.length - 1) {
                    out.write('\t');
                }
            }
            out.newLine();
        }
        out.flush();
    }

    private static DoubleMatrix1D loadDenseDoubleMatrix1D(InputStream stream, int rows) throws IOException {
        double[] v = new double[rows];

        BufferedReader in = new BufferedReader(new InputStreamReader(stream));
        for (int i = 0; i < rows; i++) {
            v[i] = parseDouble(in.readLine());
        }

        return new DenseDoubleMatrix1D(v);
    }

    private static DoubleMatrix2D loadDenseDoubleMatrix2D(InputStream stream, int rows, int columns) throws IOException {
        double[][] m = new double[rows][columns];

        BufferedReader in = new BufferedReader(new InputStreamReader(stream));
        for (double[] mi : m) {
            String[] tokens = in.readLine().split("\t", mi.length);
            for (int j = 0; j < mi.length; j++) {
                mi[j] = parseDouble(tokens[j]);
            }
        }

        return new DenseDoubleMatrix2D(m);
    }

    /**
     * Save factorisation machine in a compressed, human readable file.
     *
     * @param out output
     * @throws IOException when I/O error
     */
    public void save(OutputStream out) throws IOException {
        int N = m.rows();
        int K = m.columns();
        try (ZipOutputStream zip = new ZipOutputStream(out)) {
            zip.putNextEntry(new ZipEntry("info"));
            PrintStream ps = new PrintStream(zip);
            ps.println(N);
            ps.println(K);
            ps.flush();
            zip.closeEntry();

            zip.putNextEntry(new ZipEntry("b"));
            ps = new PrintStream(zip);
            ps.println(b);
            ps.flush();
            zip.closeEntry();

            zip.putNextEntry(new ZipEntry("w"));
            saveDenseDoubleMatrix1D(zip, w);
            zip.closeEntry();

            zip.putNextEntry(new ZipEntry("m"));
            saveDenseDoubleMatrix2D(zip, m);
            zip.closeEntry();
        }
    }

    /**
     * Loads a factorisation machine from a compressed, human readable file.
     *
     * @param in input
     * @return factorisation machine
     * @throws IOException when I/O error
     */
    public static FM load(InputStream in) throws IOException {
        int N;
        int K;
        double b;
        DoubleMatrix1D w;
        DoubleMatrix2D m;
        try (ZipInputStream zip = new ZipInputStream(in)) {
            zip.getNextEntry();
            BufferedReader reader = new BufferedReader(new InputStreamReader(zip));
            N = parseInt(reader.readLine());
            K = parseInt(reader.readLine());
            zip.closeEntry();

            zip.getNextEntry();
            reader = new BufferedReader(new InputStreamReader(zip));
            b = parseDouble(reader.readLine());
            zip.closeEntry();

            zip.getNextEntry();
            w = loadDenseDoubleMatrix1D(zip, N);
            zip.closeEntry();

            zip.getNextEntry();
            m = loadDenseDoubleMatrix2D(zip, N, K);
            zip.closeEntry();
        }

        return new FM(b, w, m);
    }

}
