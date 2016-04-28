/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.example;

import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.PointWiseGradientDescent;
import java.util.Random;
import java.util.Arrays;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.NormFMData;
import static org.ranksys.javafm.example.ML100kRatingPredictionExample.getRecommendationDataset;
import org.ranksys.javafm.instance.NormFMInstance;
import org.ranksys.javafm.learner.gd.error.ListRankError;

/**
 * Example with rating prediction (not real recommendation) with the MovieLens 100K dataset. Note that this type of rating prediction is of little use for generating useful recommendations. This is just a example of how JavaFM works.<br>
 *
 * http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ML100kRatingRankingExample {

    public static void main(String[] args) throws Exception {
        NormFMData<NormFMInstance> train = getRecommendationDataset("u1.base");
        NormFMData<NormFMInstance> test = getRecommendationDataset("u1.test");

        double learnRate = 0.01;
        int numIter = 200;
        double sdev = 0.1;
        double lambdaB = 0.1;
        double[] lambdaW = new double[train.numFeatures()];
        Arrays.fill(lambdaW, 0.1);
        double[] lambdaM = new double[train.numFeatures()];
        Arrays.fill(lambdaM, 0.1);
        int K = 100;

        FMLearner<NormFMInstance> learner = new PointWiseGradientDescent<>(learnRate, numIter,
                new ListRankError(train), new ListRankError(test), lambdaB, lambdaW, lambdaM);

        double b = 0.0;
        double[] w = new double[train.numFeatures()];
        double[][] m = new double[train.numFeatures()][K];
        Random rnd = new Random();
        for (double[] mi : m) {
            for (int j = 0; j < mi.length; j++) {
                mi[j] = rnd.nextGaussian() * sdev;
            }
        }
        FM<NormFMInstance> fm = new FM<>(b, w, m);

        learner.learn(fm, train, test);
    }

}
