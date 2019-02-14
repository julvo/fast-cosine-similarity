/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.staysense.fastcosinesimilarity;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.NullPointerException;
import java.util.Collection;
import java.util.Map;
import java.util.ArrayList;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

import com.elasticsearch.staysense.fastcosinesimilarity.Util;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.store.ByteArrayDataInput;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;
import org.elasticsearch.script.ScoreScript;


public final class FastCosineSimilarityPlugin extends Plugin implements ScriptPlugin{

    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new FastCosineSimilarityEngine();
    }

  private static class FastCosineSimilarityEngine implements ScriptEngine {

    //
    //The normalized vector score from the query
    //
    double queryVectorNorm;

    @Override
    public String getType() {
        return "fast_cosine";
    }

    @Override
    public <T> T compile(String scriptName, String scriptSource, ScriptContext<T> context, Map<String, String> params) {
        if (context.equals(ScoreScript.CONTEXT) == false) {
            throw new IllegalArgumentException(getType() + " scripts cannot be used for context [" + context.name + "]");
        }
        // we use the script "source" as the script identifier
        if ("staysense".equals(scriptSource)) {
            ScoreScript.Factory factory = (p, lookup) -> new ScoreScript.LeafFactory() {
                // The field to compare against
                final String field;
                // Whether this search should be cosine similarity or dot product
                final boolean l1, l2, cosine, dot;
                // score = (score + scoreOffset) * scoreScale
                final Double scoreOffset, scoreScale;

                // The query embedded vector
                final Object vector;
                Boolean exclude;
                // The final comma delimited vector representation of the query vector
                double[] inputVector;
                {
                    if (p.containsKey("field") == false) {
                        throw new IllegalArgumentException("Missing parameter [field]");
                    }

                    final String measure = p.get("measure").toString();
                    cosine = measure.equals("cosine");
                    l2     = measure.equals("l2");
                    l1     = measure.equals("l1");
                    dot    = !(cosine || l2 || l1);

                    final Object scoreOffsetObj = p.get("scoreOffset");
                    scoreOffset = scoreOffsetObj != null ? Double.valueOf(scoreOffsetObj.toString()) : 0.0d;
                    final Object scoreScaleObj = p.get("scoreScale");
                    scoreScale = scoreScaleObj != null ? Double.valueOf(scoreScaleObj.toString()) : 1.0d;

                    //Get the field value from the query
                    field = p.get("field").toString();

                    final Object excludeBool = p.get("exclude");
                    exclude = excludeBool != null ? (boolean) excludeBool : true;

                    //Get the query vector embedding
                    vector = p.get("vector");

                    //Determine if raw comma-delimited vector or embedding was passed
                    if(vector != null) {
                        final ArrayList<Double> tmp = (ArrayList<Double>) vector;
                        inputVector = new double[tmp.size()];
                        for (int i = 0; i < inputVector.length; i++) {
                            inputVector[i] = tmp.get(i);
                        }
                    } else {
                        final Object encodedVector = p.get("encoded_vector");
                        if(encodedVector == null) {
                            throw new IllegalArgumentException("Must have 'vector' or 'encoded_vector' as a parameter");
                        }
                        inputVector = Util.convertBase64ToArray((String) encodedVector);
                    }

                    //If cosine calculate the query vec norm
                    if(cosine) {
                        queryVectorNorm = 0d;
                        // compute query inputVector norm once
                        for (double v : inputVector) {
                            queryVectorNorm += Math.pow(v, 2.0);
                        }
                    }
                }

                @Override
                public ScoreScript newInstance(LeafReaderContext context) throws IOException {

                    return new ScoreScript(p, lookup, context) {
                          Boolean is_value = false;

                          // Use Lucene LeafReadContext to access binary values directly.
                          BinaryDocValues accessor = context.reader().getBinaryDocValues(field);

                          @Override
                          public void setDocument(int docId) {
                              // advance has undefined behavior calling with a docid <= its current docid
                              try {
                                  accessor.advanceExact(docId);
                                  is_value = true;
                              } catch (NullPointerException | IOException e) {
                                  is_value = false;
                              }
                          }

                          @Override
                          public double execute() {

                            //If there is no field value return 0 rather than fail.
                            if (!is_value) return 0.0d;

                            final int inputVectorSize = inputVector.length;
                            final byte[] bytes;

                            try {
                                 bytes = accessor.binaryValue().bytes;
                            } catch (NullPointerException | IOException e) {
                                 return 0d;
                            }


                            final ByteArrayDataInput docVectorData = new ByteArrayDataInput(bytes);

                            docVectorData.readVInt();

                            final int docVectorLength = docVectorData.readVInt(); // returns the number of bytes to read

                            if(docVectorLength != inputVectorSize * 8) {
                                return 0d;
                            }

                            final int position = docVectorData.getPosition();

                            final DoubleBuffer doubleBuffer = ByteBuffer.wrap(bytes, position, docVectorLength).asDoubleBuffer();

                            final double[] docVector = new double[inputVectorSize];
                            doubleBuffer.get(docVector);

                            double score = 0d;

                            if (cosine) {
                                double docVectorNorm = 0d;

                                for (int i = 0; i < inputVectorSize; i++) {
                                    score += docVector[i] * inputVector[i];
                                    docVectorNorm += Math.pow(docVector[i], 2.0);
                                }

                                if (docVectorNorm == 0 || queryVectorNorm == 0) return scoreOffset * scoreScale;
                                score = score / Math.sqrt(docVectorNorm) / Math.sqrt(queryVectorNorm);

                            } else if (l2) {
                                for (int i = 0; i < inputVectorSize; i++) {
                                    score += Math.pow(docVector[i] - inputVector[i], 2.0);
                                }
                                score = -Math.sqrt(score);

                            } else if (l1) {
                                for (int i = 0; i < inputVectorSize; i++) {
                                    score -= Math.abs(docVector[i] - inputVector[i]);
                                }
                                
                            } else if (dot) {
                                for (int i = 0; i < inputVectorSize; i++) {
                                    score += docVector[i] * inputVector[i];
                                }
                            }

                            score = (score + scoreOffset) * scoreScale;

                            return score;
                          }
                    };
                }

                @Override
                public boolean needs_score() {
                    return false;
                }
            };
            return context.factoryClazz.cast(factory);
        }
        throw new IllegalArgumentException("Unknown script name " + scriptSource);
    }

    @Override
    public void close() {
        // optionally close resources
    }
  }
}
