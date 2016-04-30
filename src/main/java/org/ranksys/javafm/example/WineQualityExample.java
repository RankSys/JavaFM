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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.ranksys.javafm.data.ListFMData;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.FMInstance;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import org.ranksys.javafm.learner.gd.PointWiseGradientDescent;
import java.util.DoubleSummaryStatistics;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.IntStream.range;
import java.util.stream.Stream;
import org.ranksys.javafm.BoundedFM;
import static org.ranksys.javafm.learner.gd.PointWiseError.rmse;

/**
 * Regression example with the Wine Quality dataset.<br>
 *
 * https://archive.ics.uci.edu/ml/datasets/Wine+Quality
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class WineQualityExample {

    public static void main(String[] args) throws Exception {
        FMData dataset = getWineQualityDataset();
        List<FMData> partition = getRandomPartition(dataset, 0.6, new Random(1L));
        FMData train = partition.get(0);
        FMData test = partition.get(1);

        double learnRate = 0.001;
        int numIter = 200;
        double sdev = 1.0;
        double regB = 0.01;
        double[] regW = new double[train.numFeatures()];
        Arrays.fill(regW, 0.01);
        double[] regM = new double[train.numFeatures()];
        Arrays.fill(regM, 0.01);
        int K = 10;

        BoundedFM fm = new BoundedFM(3.0, 9.0, train.numInstances(), K, new Random(), sdev);

        new PointWiseGradientDescent(learnRate, numIter, rmse(), regB, regW, regM)
                .learn(fm, train, test);
    }

    private static FMData getWineQualityDataset() throws IOException {
        int columns = 11;

        ListFMData data = new ListFMData(columns);

        String filePath = "winequality-white.csv";
        if (!new File(filePath).exists()) {
            URL url = new URL("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv");
            ReadableByteChannel rbc = Channels.newChannel(url.openStream());
            FileOutputStream fos = new FileOutputStream(filePath);
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        }

        InputStream is = new FileInputStream(filePath);

        try (BufferedReader in = new BufferedReader(new InputStreamReader(is))) {
            in.readLine();
            String instance;
            while ((instance = in.readLine()) != null) {
                String[] tokens = instance.split(";");
                double target = parseDouble(tokens[columns]);
                int[] k = range(0, columns).toArray();
                double[] v = Stream.of(tokens)
                        .limit(columns)
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                data.add(new FMInstance(target, k, v));
            }
        }

        for (int _col = 0; _col < columns; _col++) {
            int col = _col;
            
            DoubleSummaryStatistics stats = data.stream()
                    .mapToDouble(x -> x.get(col))
                    .summaryStatistics();
            double max = stats.getMax();
            double min = stats.getMin();

            if (max == min) {
                data.stream().forEach(x -> x.set(col, 0.0));
            } else {
                data.stream().forEach(x -> x.set(col, (x.get(col) - min) / (max - min)));
            }
        }

        return data;
    }

    private static List<FMData> getRandomPartition(FMData dataset, double trainProp, Random rnd) {
        Map<Boolean, List<FMInstance>> partition = dataset.stream()
                .collect(groupingBy(instance -> rnd.nextDouble() < trainProp));
        FMData train = new ListFMData(dataset.numFeatures(), new Random(), partition.get(true));
        FMData test = new ListFMData(dataset.numFeatures(), new Random(), partition.get(false));

        return Arrays.asList(train, test);
    }

}
