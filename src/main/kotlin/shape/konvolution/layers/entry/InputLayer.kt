package shape.konvolution.layers.entry

import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix

class InputLayer : EntryPoint {

    override fun forward(input : Matrix) =

        input as RealMatrix

}