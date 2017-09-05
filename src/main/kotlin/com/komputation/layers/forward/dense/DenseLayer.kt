package com.komputation.layers.forward.dense

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.activation.cpuActivationLayer
import com.komputation.cpu.layers.forward.dense.CpuDenseLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.CudaForwardLayer
import com.komputation.cuda.layers.forward.activation.cudaActivationLayer
import com.komputation.cuda.layers.forward.dense.CudaDenseLayer
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.projection.projectionLayer
import com.komputation.optimization.OptimizationInstruction

class DenseLayer internal constructor(
    private val name : String?,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val numberOutputRows : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    private val projectionName = concatenateNames(this.name, "projection")
    private val activationName = concatenateNames(this.name, "activation")

    private val projection = projectionLayer(
        this.projectionName,
        this.numberInputRows,
        this.numberInputColumns,
        true,
        this.numberOutputRows,
        this.weightInitialization,
        this.biasInitialization,
        this.optimization)

    override fun buildForCpu(): CpuDenseLayer {

        val projectionLayer = this.projection.buildForCpu()

        val activationLayer = cpuActivationLayer(this.activationName, this.activationFunction, this.numberOutputRows, 1).buildForCpu()

        return CpuDenseLayer(this.name, projectionLayer, activationLayer)

    }

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaForwardLayer {

        val projectionLayer = this.projection.buildForCuda(context, cublasHandle)

        val activationLayer = cudaActivationLayer(this.activationName, this.activationFunction, this.numberOutputRows).buildForCuda(context, cublasHandle)

        return CudaDenseLayer(this.name, projectionLayer, activationLayer)

    }


}

fun denseLayer(
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction? = null) =

    denseLayer(
        null,
        inputDimension,
        1,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy
    )

fun denseLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    activationFunction: ActivationFunction,
    optimization: OptimizationInstruction? = null) =

    denseLayer(
        name,
        inputDimension,
        1,
        outputDimension,
        weightInitialization,
        biasInitialization,
        activationFunction,
        optimization)

fun denseLayer(
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    activationFunction: ActivationFunction,
    optimization: OptimizationInstruction? = null) =

    denseLayer(
        null,
        numberInputRows,
        numberInputColumns,
        numberOutputRows,
        weightInitialization,
        biasInitialization,
        activationFunction,
        optimization)

fun denseLayer(
    name : String?,
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    activationFunction: ActivationFunction,
    optimization: OptimizationInstruction? = null) =

    DenseLayer(
        name,
        numberInputRows,
        numberInputColumns,
        numberOutputRows,
        weightInitialization,
        biasInitialization,
        activationFunction,
        optimization)