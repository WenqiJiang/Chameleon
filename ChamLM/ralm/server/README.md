# Retriever Server Prototype

Consists of a server and a client in separate files sharing a utility module to ensure they both follow the same protocol.
For quick testing of the protocol run `python ralm/retriever/serialization_utils.py`.
It contains a simple test which prints the original data encodes and decodes it and prints it again.
Currently it is up to the user to verifiy that they match.

To test the socket connection run first `python server/server.py` and then in a new terminal run `python ralm/retriever/client.py`.
The client as well as the server will print what values they "see" again for verification.
Currently most parameters are defined `ralm/retriever/serialization_utils.py` and can be changed.

* BYTE_ORDER
* BATCH_SIZE
* HIDDEN_DIM

Are safe to change.

The N_BYTES* parameters might require changes in the types conversion/interpretation and should currently be left as they are.

## Layout of Serialized Data

* $k$: 32bit integer - number of requested nearest neighbours
* $q_{i,j}$: 32bit floating point - embeeding of query
* $D$: number of hidden dimensions of Encoder
* $B$: Batchsize
* $idx$: 64bit integer - index of this nearest neighbours for retrieval in database
* $dist$: 32 bit floating point - distance from query to this neighbour

For both request and answer the values are vectorized and the vectors are concatenated.

So for a batch of queries the layout is defined as:

$$
[k|q_{1,1}|q_{1,2}|\dots|q_{1,D}|q_{2,1}|q_{2,2}|\dots|q_{2,D}|q_{3,1}|\dots|q_{B,D}]
$$

$$
[idx_{1,1}|idx_{1,2}|\dots|idx_{1,k}|idx_{2,1}|idx_{2,2}|\dots|idx_{2,k}|idx_{3,1}|\dots|idx_{B,k}|
dist_{1,1}|dist_{1,2}|\dots|dist_{1,k}|dist_{2,1}|dist_{2,2}|\dots|dist_{2,k}|dist_{3,1}|\dots|dist_{B,k}]
$$



