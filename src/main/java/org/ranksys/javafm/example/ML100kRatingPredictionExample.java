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
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.ArrayListFMData;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.sgd.RMSEFMLearner;
import static java.lang.Integer.parseInt;

/**
 * Example with rating prediction (not real recommendation) with the MovieLens 100K
 * dataset. Note that this type of rating prediction is of little use for generating useful
 * recommendations. This is just a example of how JavaFM works.<br>
 * 
 * http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class ML100kRatingPredictionExample {

    private static final int NUM_USERS = 943;
    private static final int NUM_ITEMS = 1682;

    public static void main(String[] args) throws Exception {
        FMData<FMInstance> train = getRecommendationDataset("u1.base");
        FMData<FMInstance> test = getRecommendationDataset("u1.test");

        double alpha = 0.01;
        double sample = 10.0;
        double lambdaB = 0.1;
        IntToDoubleFunction lambdaW = i -> 0.1;
        IntToDoubleFunction lambdaM = i -> 0.1;
        int K = 100;

        FMLearner<FMInstance> learner = new RMSEFMLearner(alpha, sample, lambdaB, lambdaW, lambdaM);

        FM<FMInstance> fm = learner.learn(K, train, test);

        fm.save(new FileOutputStream("fm.zip"));
    }

    private static FMData<FMInstance> getRecommendationDataset(String file) throws MalformedURLException, IOException {
        ArrayListFMData<FMInstance> dataset = new ArrayListFMData<>(NUM_USERS + NUM_ITEMS);

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

                dataset.add(new FMInstance(r, new int[]{u, i}, new double[]{1.0, 1.0}));
            });
        }

        return dataset;
    }

}
