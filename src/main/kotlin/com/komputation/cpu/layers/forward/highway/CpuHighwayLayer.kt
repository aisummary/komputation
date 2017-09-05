package com.komputation.cpu.layers.forward.highway

import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.combination.AdditionCombination
import com.komputation.cpu.layers.combination.HadamardCombination
import com.komputation.cpu.layers.forward.CpuCounterProbabilityLayer
import com.komputation.cpu.layers.forward.dense.CpuDenseLayer
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuHighwayLayer internal constructor(
    name : String?,
    inputDimension : Int,
    private val transformation : CpuDenseLayer,
    private val transformationFraction : CpuDenseLayer,
    private val transformationHadamard : HadamardCombination,
    private val counterProbability: CpuCounterProbabilityLayer,
    private val carryHadamard : HadamardCombination,
    private val addition : AdditionCombination) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    override val numberOutputRows = inputDimension
    override val numberOutputColumns = 1
    private val numberOutputEntries = this.numberOutputRows * this.numberOutputColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows = inputDimension
    override val numberInputColumns = 1
    private val numberInputEntries = this.numberInputRows * this.numberInputColumns
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberOutputEntries)
        this.backwardResult = FloatArray(this.numberInputEntries)

    }

    override fun release() {

    }

    private val gradientAccumulator = DenseAccumulator(inputDimension)

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        // H(x)
        val transformed = this.transformation.forward(withinBatch, 1, input, isTraining)

        // T(x)
        val transformationFraction = this.transformationFraction.forward(withinBatch, 1, input, isTraining)

        // H(x) (.) T(x)
        val transformationComponent = this.transformationHadamard.forward(transformed, transformationFraction)

        // 1 - T(x)
        val carryFraction = this.counterProbability.forward(withinBatch, 1, transformationFraction, isTraining)

        // x (.) (1 - T(x))
        val carryComponent = this.carryHadamard.forward(input, carryFraction)

        // H(x) (.) T(x) + x (.) (1 - T(x))
        this.forwardResult = this.addition.forward(transformationComponent, carryComponent)

        return this.forwardResult

    }

    private fun backwardTransformation(withinBatch : Int, chain: FloatArray) {

        // d chain / d H(x) (.) T(x)
        val backwardChainWrtTransformationComponent = this.addition.backwardFirst(chain)

        // d H(x) (.) T(x) / d H(x)
        val backwardTransformationComponentWrtTransformation = this.transformationHadamard.backwardFirst(backwardChainWrtTransformationComponent)

        // d H(x) / d x
        val backwardTransformation = this.transformation.backward(withinBatch, backwardTransformationComponentWrtTransformation)

        this.gradientAccumulator.accumulate(backwardTransformation)

        // d H(x) (.) T(x) / d T(x)
        val backwardTransformationComponentWrtTransformationFraction = this.transformationHadamard.backwardSecond(backwardChainWrtTransformationComponent)

        // d T(x) / d x
        this.backwardResult = this.transformationFraction.backward(withinBatch, backwardTransformationComponentWrtTransformationFraction)

        this.gradientAccumulator.accumulate(this.backwardResult)

    }

    private fun backwardCarry(withinBatch : Int, chain: FloatArray) {

        // d chain / d x (.) (1 - T(x))
        val backwardChainWrtCarryComponent = this.addition.backwardSecond(chain)

        // d x (.) (1 - T(x)) / d x
        val backwardCarryComponentWrtInput = this.carryHadamard.backwardFirst(backwardChainWrtCarryComponent)

        this.gradientAccumulator.accumulate(backwardCarryComponentWrtInput)

        // d x (.) (1 - T(x)) / d (1 - T(x))
        val backwardCarryComponentWrtCarryFraction = this.carryHadamard.backwardSecond(backwardChainWrtCarryComponent)

        // d (1 - T(x)) / d T(x)
        val backwardCounterProbability = this.counterProbability.backward(withinBatch, backwardCarryComponentWrtCarryFraction)

        // d T(x) / d x
        val backwardTransformationFraction = this.transformationFraction.backward(withinBatch, backwardCounterProbability)

        this.gradientAccumulator.accumulate(backwardTransformationFraction)

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        this.backwardTransformation(withinBatch, chain)

        this.backwardCarry(withinBatch, chain)

        System.arraycopy(this.gradientAccumulator.getAccumulation(), 0, this.backwardResult, 0, this.numberInputEntries)

        this.gradientAccumulator.reset()

        return this.backwardResult

    }

    override fun optimize(batchSize : Int) {

        this.transformation.optimize(batchSize)
        this.transformationFraction.optimize(batchSize)

    }

}