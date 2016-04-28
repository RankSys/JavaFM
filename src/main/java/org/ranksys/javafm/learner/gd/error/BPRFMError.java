///*
// * Copyright (C) 2016 RankSys http://ranksys.org
// *
// * This Source Code Form is subject to the terms of the Mozilla Public
// * License, v. 2.0. If a copy of the MPL was not distributed with this
// * file, You can obtain one at http://mozilla.org/MPL/2.0/.
// */
//package org.ranksys.javafm.learner.gd.error;
//
//import cern.colt.matrix.DoubleMatrix1D;
//import cern.colt.matrix.DoubleMatrix2D;
//import static java.lang.Math.exp;
//import static java.lang.Math.log;
//import java.util.concurrent.atomic.DoubleAdder;
//import java.util.function.IntToDoubleFunction;
//import org.ranksys.javafm.FM;
//import org.ranksys.javafm.data.FMData;
//import org.ranksys.javafm.instance.PairedFMInstance;
//
///**
// * Pair-wise error based on the Bayesian Personalized Ranking framework
// * by Rendle et al. 2009 (@UAI).
// *
// * @author Saúl Vargas (Saul@VargasSandoval.es)
// */
//public class BPRFMError implements FMError<PairedFMInstance> {
//
//    private final IntToDoubleFunction lambdaW;
//    private final IntToDoubleFunction lambdaM;
//
//    /**
//     * Constructor.
//     *
//     * @param lambda regularisation parameter
//     */
//    public BPRFMError(double lambda) {
//        this(i -> lambda, i -> lambda);
//    }
//
//    /**
//     * Constructor.
//     *
//     * @param lambdaW regularisation parameters for the feature weights vector
//     * @param lambdaM regularisation parameters for the feature interactions matrix
//     */
//    public BPRFMError(IntToDoubleFunction lambdaW, IntToDoubleFunction lambdaM) {
//        this.lambdaW = lambdaW;
//        this.lambdaM = lambdaM;
//    }
//
//    @Override
//    public double error(FM<PairedFMInstance> fm, PairedFMInstance x, FMData<PairedFMInstance> test) {
//        int p = x.getP();
//        double xp = x.getXp();
//        int n = x.getN();
//        double xn = x.getXn();
//
//        double pp = fm.prediction(x, p, xp);
//        double np = fm.prediction(x, n, xn);
//
//        return log(1 / (1 + exp(-(pp - np))));
//    }
//
//    @Override
//    public void localGradient(DoubleAdder gradB, DoubleMatrix1D gradW, DoubleMatrix2D gradM, FM<PairedFMInstance> fm, PairedFMInstance x, FMData<PairedFMInstance> train) {
//        DoubleMatrix1D w = fm.getW();
//        DoubleMatrix2D m = fm.getM();
//
//        int p = x.getP();
//        double xp = x.getXp();
//        int n = x.getN();
//        double xn = x.getXn();
//
//        double pp = fm.prediction(x, p, xp);
//        double np = fm.prediction(x, n, xn);
//
//        double err = 1 / (1 + exp(pp - np));
//
//        double wp = w.getQuick(p);
//        double wn = w.getQuick(n);
//        gradW.setQuick(p, gradW.getQuick(p) - err * xp + lambdaW.applyAsDouble(p) * wp);
//        gradW.setQuick(n, gradW.getQuick(n) + err * xn + lambdaW.applyAsDouble(n) * wn);
//
//        DoubleMatrix1D mp = m.viewRow(p);
//        DoubleMatrix1D mn = m.viewRow(n);
//
//        x.consume((i, xi) -> {
//            DoubleMatrix1D mi = m.viewRow(i);
//
//            gradM.viewRow(p)
//                    .assign(mi, (r, s) -> r - err * s * xi * xp)
//                    .assign(mp, (r, s) -> r + lambdaM.applyAsDouble(p) * s);
//            gradM.viewRow(n)
//                    .assign(mi, (r, s) -> r + err * s * xi * xn)
//                    .assign(mn, (r, s) -> r + lambdaM.applyAsDouble(n) * s);
//            gradM.viewRow(i)
//                    .assign(mp, (r, s) -> r - err * s * xi * xp)
//                    .assign(mn, (r, s) -> r + err * s * xi * xn)
//                    .assign(mi, (r, s) -> r + lambdaM.applyAsDouble(i) * s);
//        });
//    }
//
//}
