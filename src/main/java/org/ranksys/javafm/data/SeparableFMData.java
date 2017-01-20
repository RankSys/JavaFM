package org.ranksys.javafm.data;

import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.javafm.FMInstance;

import java.util.stream.Stream;

public interface SeparableFMData extends FMData {

    public int numFeaturesPart(int part);

    public Stream<Tuple2<? extends FMInstance, Stream<? extends FMInstance>>> streamPart(int part);

}
