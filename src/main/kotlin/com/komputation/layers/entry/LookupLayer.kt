package com.komputation.layers.entry

import com.komputation.cpu.functions.concatenate
import com.komputation.cpu.layers.entry.CpuLookupLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.EntryKernels
import com.komputation.cuda.kernels.FillKernels
import com.komputation.cuda.kernels.HashtableKernels
import com.komputation.cuda.layers.entry.CudaGroupSum
import com.komputation.cuda.layers.entry.CudaHashing
import com.komputation.cuda.layers.entry.CudaLookupLayer
import com.komputation.layers.CpuEntryPointInstruction
import com.komputation.layers.CudaEntryPointInstruction
import com.komputation.optimization.OptimizationInstruction

class LookupLayer(
    private val name : String? = null,
    private val vectors: Array<FloatArray>,
    private val maximumLength: Int,
    private val hasFixedLength: Boolean,
    private val dimension : Int,
    private val optimization : OptimizationInstruction?) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu(): CpuLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCpu().invoke(this.vectors.size, this.dimension)

        }
        else {

            null
        }

        val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

        return CpuLookupLayer(this.name, this.vectors, minimumLength, this.maximumLength, this.dimension, updateRule)

    }

    override fun buildForCuda(context: CudaContext): CudaLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCuda(context).invoke(this.vectors.size, this.dimension, 1)

        }
        else {

            null
        }

        val concatenation = FloatArray(this.vectors.size * this.dimension)
        for ((index, vector) in this.vectors.withIndex()) {

            concatenate(vector, index * this.dimension, this.dimension, concatenation)

        }

        val hashing = CudaHashing(
            this.maximumLength,
            2,
            { context.createKernel(HashtableKernels.hash()) },
            { context.createKernel(FillKernels.twoIntegerArrays()) })

        val groupSum = CudaGroupSum(
            this.dimension,
            this.maximumLength,
            2 * this.maximumLength,
            { context.createKernel(HashtableKernels.groupSum()) },
            { context.createKernel(FillKernels.oneFloatArray()) })

        return CudaLookupLayer(
            this.name,
            concatenation,
            this.maximumLength,
            this.hasFixedLength,
            this.dimension,
            updateRule,
            { context.createKernel(EntryKernels.lookup())},
            hashing,
            groupSum,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun lookupLayer(
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    lookupLayer(
        null,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        optimization
    )

fun lookupLayer(
    name: String? = null,
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    LookupLayer(
        name,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        optimization
    )