/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.example;

import org.ranksys.javafm.BoundedFM;
import org.ranksys.javafm.FMInstance;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.data.SimpleListWiseFMData;
import org.ranksys.javafm.learner.gd.ParallelPointWiseGradientDescent;

import java.io.*;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.Arrays;
import java.util.Random;

import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;
import static org.ranksys.javafm.learner.gd.PointWiseError.rmse;

/**
 * Example with rating prediction (not real recommendation) with the MovieLens
 * 100K dataset. Note that this type of rating prediction is of little use for
 * generating real-world recommendations. This is just a example of how JavaFM
 * works.<br>
 *
 * http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ML100kRatingPredictionExample {

    private static final int NUM_USERS = 6040;
    private static final int NUM_ITEMS = 3952;

    public static void main(String[] args) throws Exception {
//        FMData train = getRecommendationDataset("u1.base");
//        FMData test = getRecommendationDataset("u1.test");
        FMData train = getRecommendationDataset("/Users/vargass/Datasets/ml1M_fold1/train.data");
        FMData test = getRecommendationDataset("/Users/vargass/Datasets/ml1M_fold1/test.data");

        double learnRate = 0.01;
        int numIter = 200;
        double sdev = 0.1;
        double regB = 0.1;
        double[] regW = new double[train.numFeatures()];
        Arrays.fill(regW, 0.1);
        double[] regM = new double[train.numFeatures()];
        Arrays.fill(regM, 0.1);
        int K = 1000;

        BoundedFM fm = new BoundedFM(1.0, 5.0, train.numFeatures(), K, new Random(), sdev);

        new ParallelPointWiseGradientDescent(learnRate, numIter, rmse(), regB, regW, regM)
                .learn(fm, train, test);
    }

    public static SimpleListWiseFMData getRecommendationDataset(String file) throws IOException {
        SimpleListWiseFMData dataset = new SimpleListWiseFMData(NUM_USERS + NUM_ITEMS);

        if (!new File(file).exists()) {
            URL url = new URL("http://files.grouplens.org/datasets/movielens/ml-100k/" + file);
            ReadableByteChannel rbc = Channels.newChannel(url.openStream());
            FileOutputStream fos = new FileOutputStream(file);
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        }

        InputStream is = new FileInputStream(file);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            reader.lines().forEach(line -> {
                String[] tokens = line.split("\t");
                int u = parseInt(tokens[0]) - 1;
                int i = parseInt(tokens[1]) - 1 + NUM_USERS;
                double r = parseDouble(tokens[2]);

                dataset.add(new FMInstance(r, new int[]{u, i}, new double[]{1.0, 1.0}), u);
            });
        }

        return dataset;
    }

}
