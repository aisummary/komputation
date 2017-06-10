package shape.konvolution.layers.continuation

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix

class SoftmaxLayerTest {

    @Test
    fun test() {

        val softmaxLayer = SoftmaxLayer()

        val input = createRealMatrix(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))

        val actual = softmaxLayer.forward(input).single()
        val expected = createRealMatrix(
            doubleArrayOf(Math.exp(1.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(2.0) / (Math.exp(2.0) + Math.exp(4.0))),
            doubleArrayOf(Math.exp(3.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(4.0) / (Math.exp(2.0) + Math.exp(4.0)))
        )

        assertMatrixEquality(expected, actual, 0.0001)

    }


}