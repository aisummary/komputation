package com.komputation.cpu.functions

/*
    word^(1)_1   word^(2)_1   ...   word^(T)_1
    word^(1)_2   word^(2)_2   ...   word^(T)_2
    ...          ...                ....
    word^(1)_d   word^(2)_d   ...   word^(T)_d
*/


fun lookup(vectors: Array<FloatArray>, dimension: Int, length: Int, ids: IntArray, result: FloatArray) {

    for (index in 0..length - 1) {

        val id = ids[index]

        val start = index * dimension

        val vector = vectors[id]

        System.arraycopy(vector, 0, result, start, dimension)

    }

}