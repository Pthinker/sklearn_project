#!/bin/sh
scriptdir=`dirname $0`

java -mx1000m -cp "$scriptdir/stanford-ner.jar" edu.stanford.nlp.ie.NERServer -loadClassifier $scriptdir/classifiers/english.all.3class.distsim.crf.ser.gz -port 8080 -outputFormat inlineXML
