/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import static java.lang.Double.parseDouble;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.error.RMSEError;
import org.ranksys.javafm.learner.gd.PointWiseGradientDescent;
import java.util.Random;
import org.ranksys.javafm.BoundedFM;
import static java.lang.Integer.parseInt;
import java.util.Arrays;
import org.ranksys.javafm.data.NormFMData;
import org.ranksys.javafm.instance.NormFMInstance;
import static java.lang.Integer.parseInt;
import static java.lang.Integer.parseInt;
import static java.lang.Integer.parseInt;

/**
 * Example with rating prediction (not real recommendation) with the MovieLens 100K dataset. Note that this type of rating prediction is of little use for generating useful recommendations. This is just a example of how JavaFM works.<br>
 *
 * http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ML100kRatingPredictionExample {

    private static final int NUM_USERS = 943;
    private static final int NUM_ITEMS = 1682;

    public static void main(String[] args) throws Exception {
        FMData<NormFMInstance> train = getRecommendationDataset("u1.base");
        FMData<NormFMInstance> test = getRecommendationDataset("u1.test");

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
                new RMSEError<>(), new RMSEError<>(), lambdaB, lambdaW, lambdaM);

        double b = 0.0;
        double[] w = new double[train.numFeatures()];
        double[][] m = new double[train.numFeatures()][K];
        Random rnd = new Random();
        for (double[] mi : m) {
            for (int j = 0; j < mi.length; j++) {
                mi[j] = rnd.nextGaussian() * sdev;
            }
        }
        BoundedFM<NormFMInstance> fm = new BoundedFM<>(b, w, m, 1.0, 5.0);

        learner.learn(fm, train, test);
    }

    public static NormFMData<NormFMInstance> getRecommendationDataset(String file) throws MalformedURLException, IOException {
        NormFMData<NormFMInstance> dataset = new NormFMData<>(NUM_USERS + NUM_ITEMS);

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

                dataset.add(new NormFMInstance(u, r, new int[]{u, i}, new double[]{1.0, 1.0}));
            });
        }
        
//        Collections.shuffle(dataset);

        return dataset;
    }

}
