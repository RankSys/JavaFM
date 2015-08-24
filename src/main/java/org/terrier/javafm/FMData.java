package org.terrier.javafm;

import java.util.stream.Stream;

/**
 *
 * @author Saúl Vargas (Saul.Vargas@glasgow.ac.uk)
 */
public interface FMData {
    
    public int numInstances();
    
    public int numFeatures();
    
    public Stream<? extends FMInstance> stream();
    
    public Stream<? extends FMInstance> stream(int i);
    
    public Stream<? extends FMInstance> sample(int n);
    
}
