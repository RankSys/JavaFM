package org.terrier.javafm.learner;

import org.terrier.javafm.FMData;
import org.terrier.javafm.FM;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public interface FMLearner {

    public double error(FM fm, FMData test);

    public FM learn(int K, FMData train, FMData test);

    public default FM learn(int K, FMData train) {
        return learn(K, train, train);
    }

    public void learn(FM fm, FMData train, FMData test);

    public default void learn(FM fm, FMData train) {
        learn(fm, train, train);
    }
}
