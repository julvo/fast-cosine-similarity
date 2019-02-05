![alt text](http://staysense.com/img/staysense-logo.png "StaySense Orchestrator")

StaySense - Fast Cosine Similarity ElasticSearch Plugin
================================
Extremely fast vector scoring on ElasticSearch 6.4.x+ using vector embeddings.

<b>About StaySense</b>: [StaySense](http://staysense.com) is a revolutionary software company creating the most advanced marketing software ever made publicly available for Hospitality Managers in the Vacation Rental and Hotel Industries. 

Company Website: <http://staysense.com>

# Fast Elasticsearch Vector Scoring

This Plugin allows you to score Elasticsearch documents based on embedding-vectors, using dot-product or cosine-similarity at break neck speeds.

## General
* This plugin was ported from [This elasticsearch 5.x vector scoring plugin](https://github.com/MLnick/elasticsearch-vector-scoring) and [this discussion](https://discuss.elastic.co/t/vector-scoring/85227/6) and  [lior-k](https://github.com/lior-k)'s original contribution for ElasticSearch 5.5+ to achieve lightning fast result times when searching across millions of documents.
* This port is for ElasticSearch 6.4+ utilizing the ScoreScript class which was officially split from SearchScript and thus incompatible < 6.4.x

## Improvements
* lior-k's implementation had some confusing variable assignments that did not consistently match with Cosine-Sim's [mathematical definition](https://en.wikipedia.org/wiki/Cosine_similarity#Definition). This has been updated in the code to more accurately reflect the mathematical definition.
* Null pointer exceptions are now skipped (e.g. a document doesn't have a vector to compare against) allowing queries to complete successfully even in sparse datasets.
* Ported for latest version of ElasticSearch.
* Issues and Pull-Requests welcomed!


## Elasticsearch version
* Currently designed for Elasticsearch 6.4.x+
* Plugin is NOT backwards compatible (see note above about ScoreScript class)
* Will succesfully build for 6.4.0 and 6.4.1 (latest). Simply modify pom.xml with the correct version then follow maven build steps below.


## Maven Build Steps
* Clone the project
* `mvn package` to compile the plugin as a zip file
* In Elasticsearch run `elasticsearch-plugin install file:/PATH_TO_ZIP` to install plugin

## Why embeddings?
* Ultimately, by defining the field mapping as a binary value, by storing an embedded version of the vector you are able to take advantage of Lucene's direct API to achieve direct byte access without transformation.
* When creating the document, Lucene encodes the embedding directly to binary, making read access blazing fast on the search side.
* Does Lucene do the same with non-embedded vectors? Unsure, but the plugin supports that too if you want to store in [1.2934, -2.0349, ...., .039] format and try!

## Usage

### Documents
* Each document you score should have a field containing the base64 representation of your vector. for example:
```
   {
   	"_id": 1,
   	....
   	"embeddedVector": "v7l48eAAAAA/s4VHwAAAAD+R7I5AAAAAv8MBMAAAAAA/yEI3AAAAAL/IWkeAAAAAv7s480AAAAC/v6DUgAAAAL+wJi0gAAAAP76VqUAAAAC/sL1ZYAAAAL/dyq/gAAAAP62FVcAAAAC/tQRvYAAAAL+j6ycAAAAAP6v1KcAAAAC/bN5hQAAAAL+u9ItAAAAAP4ckTsAAAAC/pmkjYAAAAD+cYpwAAAAAP5renEAAAAC/qY0HQAAAAD+wyYGgAAAAP5WrCcAAAAA/qzjTQAAAAD++LBzAAAAAP49wNKAAAAC/vu/aIAAAAD+hqXfAAAAAP4FfNCAAAAA/pjC64AAAAL+qwT2gAAAAv6S3OGAAAAC/gfMtgAAAAD/If5ZAAAAAP5mcXOAAAAC/xYAU4AAAAL+2nlfAAAAAP7sCXOAAAAA/petBIAAAAD9soYnAAAAAv5R7X+AAAAC/pgM/IAAAAL+ojI/gAAAAP2gPz2AAAAA/3FonoAAAAL/IHg1AAAAAv6p1SmAAAAA/tvKlQAAAAD/I2OMAAAAAP3FBiCAAAAA/wEd8IAAAAL94wI9AAAAAP2Y1IIAAAAA/rnS4wAAAAL9vriVgAAAAv1QxoCAAAAC/1/qu4AAAAL+inZFAAAAAv7aGA+AAAAA/lqYVYAAAAD+kNP0AAAAAP730BiAAAAA="
   }
   ```
* Use this field mapping:
```
      "embeddedVector": {
        "type": "binary",
        "doc_values": true
      }
```
* The vector can be of any dimension

### Converting a vector to Base64
to convert an array of doubles to a base64 string we use these example methods:

**Java**
```
public static final String convertArrayToBase64(double[] array) {
	final int capacity = 8 * array.length;
	final ByteBuffer bb = ByteBuffer.allocate(capacity);
	for (int i = 0; i < array.length; i++) {
		bb.putDouble(array[i]);
	}
	bb.rewind();
	final ByteBuffer encodedBB = Base64.getEncoder().encode(bb);
	return new String(encodedBB.array());
}

public static double[] convertBase64ToArray(String base64Str) {
	final byte[] decode = Base64.getDecoder().decode(base64Str.getBytes());
	final DoubleBuffer doubleBuffer = ByteBuffer.wrap(decode).asDoubleBuffer();

	final double[] dims = new double[doubleBuffer.capacity()];
	doubleBuffer.get(dims);
	return dims;
}
```
**Python**
```
import base64
import numpy as np

dbig = np.dtype('>f8')

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=dbig).tolist()

def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dbig)).decode("utf-8")
    return base64_str
```
**Go**
```
import(
    "math"
    "encoding/binary"
    "encoding/base64"
)
func convertArrayToBase64(array []float64) string {
	bytes := make([]byte, 0, 8*len(array))
	for _, a := range array {
		bits := math.Float64bits(a)
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, bits)
		bytes = append(bytes, b...)
	}
 	encoded := base64.StdEncoding.EncodeToString(bytes)
	return encoded
}
func convertBase64ToArray(base64Str string) ([]float64, error) {
	decoded, err := base64.StdEncoding.DecodeString(base64Str)
	if err != nil {
		return nil, err
	}
 	length := len(decoded)
	array := make([]float64, 0, length/8)
 	for i := 0; i < len(decoded); i += 8 {
		bits := binary.BigEndian.Uint64(decoded[i : i+8])
		f := math.Float64frombits(bits)
		array = append(array, f)
	}
	return array, nil
}
```


### Querying

## Querying with encodings
* Query for documents based on their cosine similarity:


    For ES 6.4.x:
```
{
  "query": {
    "function_score": {
    "boost_mode" : "replace",
        "functions": [
          {
            "script_score": {
              "script": {
                  "source": "staysense",
                  "lang" : "fast_cosine",
                  "params": {
                      "field": "embeddedVector",
                      "measure": "cosine",
                      "encoded_vector" : "v+kopYAAAAA/wivkYAAAAD+wfJeAAAAAv8DL4QAAAAA/waYiwAAAAL+zAmvAAAAAv8c+aiAAAAC/07MyQAAAAL+ccr9AAAAAP9feCOAAAAC/y+ivYAAAAL/R34XgAAAAv+G8nuAAAAA/09hlwAAAAL/MkSWAAAAAP9EXn4AAAAC/zBBxYAAAAD/UY+3AAAAAP7zQSkAAAAC/zRijgAAAAA=="
                  }
              }
            }
          }
        ]
    }
  }
}
```

* The example above shows a vector of 64 dimensions
* Parameters:
   1. `field`: The document field containing the base64 vector to compare against.
   2. `measure`: String. Specifies the similarity measure to use, either "dot", "cosine", "l1" or "l2"
   3. `encoded_vector`: The encoded vector to compare to.

## Querying with vectors
* Query for documents based on their cosine similarity:


    For ES 6.4.x:
```
{
  "query": {
    "function_score": {
    "boost_mode" : "replace",
        "functions": [
          {
            "script_score": {
              "script": {
                  "source": "staysense",
                  "lang" : "fast_cosine",
                  "params": {
                      "field": "embeddedVector",
                      "measure": "l2",
                      "vector" : [
                      -0.09217305481433868, 0.010635560378432274, -0.02878434956073761, ... , 0.08279753476381302
                      ]
                  }
              }
            }
          }
        ]
    }
  }
}
```
* The example above shows a vector of 64 dimensions
* Parameters:
   1. `field`: The document field containing the base64 vector to compare against.
   2. `measure`: String. Specifies the similarity measure to use, either "dot", "cosine", "l1" or "l2"
   3. `vector`: The comma separated non-encoded vector to compare to.
