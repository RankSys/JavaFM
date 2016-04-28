/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.example;

import java.util.Arrays;
import java.util.Random;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.GroupFMData;
import static org.ranksys.javafm.example.ML100kRatingPredictionExample.getRecommendationDataset;
import org.ranksys.javafm.learner.gd.ListRank;

/**
 * Example with rating prediction (not real recommendation) with the MovieLens 100K dataset. Note that this type of rating prediction is of little use for generating useful recommendations. This is just a example of how JavaFM works.<br>
 *
 * http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ML100kRatingRankingExample {

    public static void main(String[] args) throws Exception {
        GroupFMData train = getRecommendationDataset("u1.base");
        GroupFMData test = getRecommendationDataset("u1.test");

        double learnRate = 0.01;
        int numIter = 200;
        double sdev = 0.1;
        double regB = 0.01;
        double[] regW = new double[train.numFeatures()];
        Arrays.fill(regW, 0.01);
        double[] regM = new double[train.numFeatures()];
        Arrays.fill(regM, 0.01);
        int K = 100;
        
        FM fm = new FM(train.numFeatures(), K, new Random(), sdev);

        new ListRank(learnRate, numIter, regB, regW, regM)
                .learn(fm, train, test);
    }

}
