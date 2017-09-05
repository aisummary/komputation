package com.komputation.cpu.optimization

import com.komputation.cpu.functions.scale

fun updateDensely(parameters: FloatArray, gradient : FloatArray, dimension : Int, batchSize : Int, rule : UpdateRule) {

    scale(gradient, 1f.div(batchSize.toFloat()), gradient, dimension)

    rule.updateDensely(parameters, gradient, dimension)

}