package com.komputation.cuda.optimization

typealias CudaOptimizationStrategy = (numberParameters : Int, numberRows : Int, numberColumns : Int) -> BaseCudaUpdateRule