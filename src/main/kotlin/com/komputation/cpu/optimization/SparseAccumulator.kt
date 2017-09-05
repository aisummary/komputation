package com.komputation.cpu.optimization

import com.komputation.matrix.constantIntArray
import java.util.*

class SparseAccumulator(numberVectors : Int, maximumBatchSize : Int, maximumLength : Int, private val dimension : Int) {

    private val hashTable = constantIntArray(numberVectors, -1)
    private val visited = BooleanArray(numberVectors)

    private val reverseHashTable = constantIntArray(maximumBatchSize * maximumLength, -1)
    private val counts = FloatArray(maximumBatchSize * maximumLength)
    private val sums = Array(maximumBatchSize * maximumLength) { FloatArray(dimension) }

    private var lastNewId = -1

    fun accumulate(ids: IntArray, numberIds : Int, gradient: FloatArray) {

        for (indexId in 0..numberIds-1) {

            val currentId = ids[indexId]

            val hashedId = this.hashId(currentId)

            this.addToSum(indexId, hashedId, gradient)

            // Is this the first occurrence of the vector?
            if (!this.visited[currentId]) {

                // Increment the count
                this.counts[hashedId] += 1.0f

                // Avoid further increments for the current example.
                this.visited[currentId] = true

            }

        }

        // Reset the visit flag
        for (currentId in ids) {

            this.visited[currentId] = false

        }

    }

    private fun hashId(id: Int): Int {

        val existingHash = this.hashTable[id]

        if (existingHash == -1) {

            val newHash = ++this.lastNewId

            this.hashTable[id] = newHash
            this.reverseHashTable[newHash] = id

            return newHash

        }
        else {

            return existingHash

        }

    }

    private fun addToSum(indexId: Int, hashedId: Int, gradient: FloatArray) {

        val sum = this.sums[hashedId]

        val start = indexId * this.dimension
        for (indexDimension in 0..this.dimension - 1) {

            sum[indexDimension] += gradient[start + indexDimension]

        }

    }

    fun getSize() = this.lastNewId + 1

    fun getIds() = this.reverseHashTable

    fun getCounts() = this.counts

    fun getSums() = this.sums

    fun reset() {

        for (indexId in 0..this.lastNewId) {

            val id = this.reverseHashTable[indexId]

            this.reverseHashTable[indexId] = -1
            this.hashTable[id] = -1

            this.counts[indexId] = 0.0f

            Arrays.fill(this.sums[indexId], 0.0f)

        }

        this.lastNewId = -1

    }

}