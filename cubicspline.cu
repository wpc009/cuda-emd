// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>

#define USE_CUSPARSE   true
// includes, project
#if defined(USE_CUDPP)
#include "cudpp.h"
#elif defined(USE_CUSPARSE)
#include "cusparse.h"
#endif

#include <cuda_runtime_api.h>
#include "cuda_util.h"

#define CUDPP_APP_COMMON_IMPL
#include "common_config.h"
#include "stopwatch.h"

#include <string>


using namespace cudpp_app;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
void cubicSpline();
void testing(int,int);
void testCases();

template <typename T> __global__ void _prepare_systems(T* a,T* b,T* c,T* d,const T* x,const T* y,T* h,const int n);
template <typename T> void preparing_parameters_gpu(T*,T*,T*,T*,const T*,T*,const T*,const int*,int,int,int);
template <typename T> void preparing_parameters_cpu(T*,T*,T*,T*,const T*,T*,const T*,const int*,int,int,int);

  
#define printArray(array,len) printArrayFmt(array,len,%f)


#define printArrayFmt(array,len,fmt) printf(#array"=["); \
    for(int i =0;i<len;i++){ \
        if(i % 10 == 0){ \
            printf("\n");   \
        }   \
        printf(#fmt", ",array[i]); \
    } \
    printf("]\n")
    
#define println(fmt,...) printf(fmt,##__VA_ARGS__);printf("\n")

#define blockalign(x,block_size) (x + block_size -1 )/ block_size

struct AddressPair_F
{
    float * host;
    float * device;
    /* data */
};

#if defined(USE_CUSPARSE)
void testCusparse(){
    cusparseHandle_t handle=0;
    cusparseStatus_t status;
    int version;

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS){
        println("CUSPARSE Library initializing failed.");
    }

    status = cusparseGetVersion(handle,&version);
    if ( status != CUSPARSE_STATUS_SUCCESS){
        println("CUSPRSE get version failed");
    }else{
        println("CUSPARSE Version:%d",version);
    }


}
#endif
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    // int numSMs;
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz; multi processor count:%d\n",
               prop.name, (unsigned int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate,prop.multiProcessorCount);
    }

#if defined(USE_CUSPARSE)
    testCusparse();
#endif

    // runTest( argc, argv);
    // cubicSpline();
    // testing(512,16);
    testCases();
}

//cuda kernel's thread and block can not expand without limit. So, can not only using num of block and thread 
    //to represent system scale. stride trick must be used.
    //this testing is used to check whether stride logic is correct.

    //by design, each block handle one interplation segment. block stride through all segments.
    //thread num of controled by the system size parameter. thread stride through all point inside a segment.

void testCases(){
    /* 
        
        Case 1. thread stride testing
    */
    testing(
        100000, //num of check points, determin the num of segments. < total blocks
        1024 //system size,  the size of a single system. determin the num of threads. < segment size(20)
    );
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void _prepare_systems(T* a,T* b,T* c,T* d,const T* x,const T* y,const int* check_point_idx,T* h,const int n)
{
    //blockDim == systemSize
    //gridDim = numOfSystem
    //blockIdx == systemIdx
    int ti_base = blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ T h[blockDim+2];

    //grid-stride
    for(int ti = ti_base;ti < n;ti += blockDim.x * gridDim.x){



    if (ti >= n)
        return;

    if(ti == 0){
        //head

        // h[1] = 0.0f;
        d[ti] = 6.0f*(y[check_point_idx[1]] - y[check_point_idx[0]])/(x[check_point_idx[1]] - x[check_point_idx[0]])/ (x[check_point_idx[1]] - x[check_point_idx[0]]);
        h[ti] = 0.0f;
    }else if (ti < n -1){
        //middle
        /* shared mem version
        if( threadIdx.x == 0){
            //head thread
            h[0] = ti > 2 ? x[ti-1] - x[ti -2]:0.0f;
        }else if(threadIdx.x == blockDim.x -1){
            //last thread
            h[threadIdx.x + 2] = x[ti +1 ] - x[ti];
        }
        */
        
        for(int j=0;j<3;j++){
            d[ti] += 6.0f * y[check_point_idx[ti-1 + j]] / (x[check_point_idx[ ti-1 + j]] - x[check_point_idx[ti-1 + (j+1)%3]]) / (x[check_point_idx[ ti-1+j]] - x[check_point_idx[ti-1 + (j+2)%3]]);
        }
        h[ti] = x[check_point_idx[ti]] - x[check_point_idx[ti-1]]; 
        
    }else{
        //last ti = n-1
        
        d[ti] = 6.0f * (0.0f - (y[check_point_idx[ ti]] - y[check_point_idx[ti-1]])/(x[check_point_idx[ti]] - x[check_point_idx[ti-1]]) ) / (x[check_point_idx[ti]] - x[check_point_idx[ti-1]]);
        h[ti] = x[check_point_idx[ ti]] - x[check_point_idx[ti-1]];
    }

    if(threadIdx.x == blockDim.x -1 && ti < n - 1){
        //block can not be synchronized, so calculate the h[ti + 1] for the last ti in block. do not exceed n -1.
        h[ti + 1]  = x[check_point_idx[ti + 1]] - x[check_point_idx[ti]];
    }
    
    __syncthreads();        
    b[ti] = 2.0f;
    a[ti] = ti < n - 1 ? h[ti] / (h[ti] + h[ti +1]) : 1.0f;
    // c[ti] = h[threadIdx.x+2] / (h[threadIdx.x+1] + h[threadIdx.x +2]);
    c[ti] = ti < n - 1 ? h[ti +1 ] / (h[ti+1] + h[ti]):0.0f;
    }
}

//deprecated
template <typename T>
__global__ void _cubic_spline_segment(const T* x,const T* y ,const T* m,const T* h,const T* spline_x,T* spline_y, int segment_len){
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if ( ti >= segment_len)
        return;

    float temp1 = x[1] - spline_x[ti];
    float temp2 = spline_x[ti] - x[0];
    spline_y[ti] = m[0] * powf(temp1,3) / (6 * h[0])
    + m[1] * powf(temp2,3) / (6*h[0])
    + (y[0] - m[0]*powf(h[0],2)/6) * (temp1) / h[0]
    + (y[1] - m[1]*powf(h[0],2)/6) * (temp2) / h[0]
    ;
}

template <typename T>
__global__ void _cubic_spline(const T* x,T* y,const int* check_point_idx,const T* m,const T* h,int totalSize){
    //grid-stride
    for( int i = blockIdx.x; i < totalSize; i += gridDim.x){
        int to_idx = check_point_idx[i];
        int from_idx = 0;
        if ( i > 0){
            from_idx = check_point_idx[i-1];
        }
        //block-stride
        for( int j = from_idx + threadIdx.x + 1; j < to_idx; j += blockDim.x){
            y[j] = m[i-1] * powf(x[to_idx] - x[j],3) / (6*h[i])
                + m[i] * powf(x[j] - x[from_idx],3) / (6*h[i])
                + (y[from_idx] - m[i-1] * powf(h[i],2)/6) * (x[to_idx] - x[j]) / h[i]
                + (y[to_idx] - m[i] * powf(h[i],2)/6) * (x[j] - x[from_idx]) / h[i]
                ;
        }

    }
}
template <typename T>
__global__ void _cubic_spline2(const T* x,T* y,const int* check_point_idx,const T* m,const T* h,int totalSize)
{
    for ( int i = threadIdx.x; i < totalSize; i += blockDim.x)
    {
        int to_idx = check_point_idx[i];
        int from_idx = 0;
        if ( i > 0 ){
            from_idx = check_point_idx[i -1 ];
        }

        for( int j = from_idx + blockIdx.x + 1; j < to_idx; j+= gridDim.x)
        {
            y[j] = m[i-1] * powf(x[to_idx] - x[j],3) / (6*h[i])
                + m[i] * powf(x[j] - x[from_idx],3) / (6*h[i])
                + (y[from_idx] - m[i-1] * powf(h[i],2)/6) * (x[to_idx] - x[j]) / h[i]
                + (y[to_idx] - m[i] * powf(h[i],2)/6) * (x[j] - x[from_idx]) / h[i]
                ;   
        }
    }
}

template <typename T>
void preparing_parameters_gpu(
    T* d_a,
    T* d_b,
    T* d_c,
    T* d_d,
    const T* d_x,
    T* d_diff,
    const T* d_y,
    const int* d_check_point_idx,
    const int len,
    const int systemSize,
    const int numOfSystem
    )
{
    if(systemSize * numOfSystem < len){
        println("invalid systemSize:%d and numOfSystem:%d parameter, less than total size:%d",systemSize,numOfSystem,len);
        return;
    }
    
    _prepare_systems<T><<<28*4,1024>>>(d_a,d_b,d_c,d_d,d_x,d_y,d_check_point_idx,d_diff,len);
}

template <typename T>
void preparing_parameters_cpu(
    T* a,
    T* b,
    T* c,
    T* d,
    const T* x,
    T* diff,
    const T* y,
    const int * check_point_idx,
    const int len,
    const int systemSize,
    const int numOfSystem
    )
{
    diff[0] = 0.0f;
    for(int i =1;i<len;i++)
    {
        diff[i] = x[check_point_idx[i] ] - x[check_point_idx[i-1]];
    }

    for(int s=0;s<numOfSystem;s++)
    {
        int base = s*systemSize;
        float* sub_b = b + base;
        float* sub_a = a + base;
        float* sub_c = c + base;
        float* sub_d = d + base;
        float* sub_diff = diff + base;
        // const float* sub_x = x + base;
        const int* sub_x_idx = check_point_idx + base;
        const float* sub_y = y + base;

        int last = len -1;
        for(int i=0;i<systemSize && i+base < len;i++)
        {

            sub_b[i] = 2.0f;
            if( i + base ==0){
                sub_a[i] = 0.0f;
                sub_c[i] = 1.0f;
                sub_d[i] = 6.0f*(y[sub_x_idx[1]] - y[sub_x_idx[0]])/(x[sub_x_idx[1]] - x[sub_x_idx[0]])/ (x[sub_x_idx[1]] - x[sub_x_idx[0]]);
                // sub_d[i] = 0.0f;
            }else if (i + base < last){
                sub_a[i] = sub_diff[i] / (sub_diff[i] + sub_diff[i+1]);
                sub_c[i] = sub_diff[i+1] / (sub_diff[i] + sub_diff[i+1]);
                sub_d[i] = 0.0f;
                // println("=============================");
                for(int j=0;j<3;j++){
                    sub_d[i] += 6.0f * y[sub_x_idx[i-1 + j]] / (x[sub_x_idx[i-1 + j]] - x[sub_x_idx[i-1 + (j+1)%3]]) / (x[sub_x_idx[i-1+j]] - x[sub_x_idx[i-1 + (j+2)%3]]);
                    // printf("y[%d] / (x[%d] - x[%d]) / (x[%d] - x[%d])\n",i-1+j,i-1+j,i-1+(j+1)%3,i-1+j,i-1+(j+2)%3);
                }
                // printf("=============================\n");
                // println("=============================");

            }else{
                // println("last i=%d y[i]=%f y[i-1]=%f",i,y[i],y[i-1]);
                sub_d[i] = 6.0f * (0.0f - (y[sub_x_idx[i]] - y[sub_x_idx[i-1]])/(x[sub_x_idx[i]] - x[sub_x_idx[i-1]]) ) / (x[sub_x_idx[i]] - x[sub_x_idx[i-1]]);
                // sub_d[i] = 0.0f;
                // printf("last element:%d@%f\n",i,sub_d[i]);
                sub_a[i] = 1.0f;
                sub_c[i] = 0.0f;

            }

            if(sub_d[i] != sub_d[i]){
                println("d[i+base] = %f, i+base=%d,i=%d",sub_d[i],i+base,i);
            }
        }
    }

}

void testing(int len,int SYSTEM_SIZE){
    //GPU
    // int len = 64;
    // int SYSTEM_SIZE = 32; // one system per block, so system size should be equal to block size (num of threads per block).
    int numOfblocks = blockalign(len,SYSTEM_SIZE);
    int numOfSystem = max(len / SYSTEM_SIZE,1);
    int memSize = len * sizeof(float);
    int spline_len;
    int part_i;
    cusparseHandle_t handle=0;
    cusparseStatus_t status;

    float a_error= 0.0f;
    float b_error = 0.0f;
    float c_error= 0.0f;
    float d_error = 0.0f;
    float m_error = 0.0f;

    int segment_start = 1;
    int segment_end = 1;
    int offset = 0;
    int residual = 0;



    cudpp_app::StopWatch timer;
    


    float * d_a;
    float * d_b;
    float * d_c;
    float * d_d;
    // float * d_x;
    // float * d_y;
    int * d_check_point_idx;
    float * d_diff;
    float * d_spline_x;
    float * d_spline_y;
    float * d_m;

    /*
    Malloc memories
    */
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_y,memSize));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_m,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_diff,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_check_point_idx,memSize) );


    CUDA_SAFE_CALL( cudaMemset(d_a,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_b,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_c,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_d,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_diff,0,memSize) );

    
    int * x = (int*) malloc(len * sizeof(int));
    float * y = (float*) malloc(memSize);
    float * spline_x;
    float * spline_y;
/*
For Testing
*/
    float * a2 = (float*) malloc(memSize);
    float * b2 = (float*) malloc(memSize);
    float * c2 = (float*) malloc(memSize);
    float * d2 = (float*) malloc(memSize);
    float * h2 = (float*) malloc(memSize);
    float * m2 = (float*) malloc(memSize);

    float * a = (float*) malloc(memSize);
    float * b = (float*) malloc(memSize);
    float * c = (float*) malloc(memSize);
    float * d = (float*) malloc(memSize);
    float * h = (float*) malloc(memSize);
    float * m = (float*) malloc(memSize);

    

    AddressPair_F addressPairs[] = {
        {a2,d_a},
        {b2,d_b},
        {c2,d_c},
        {d2,d_d},
        {h2,d_diff}
    };

/*==========================
Fake data
===========================*/
    spline_len = (len -1) * 20 + 1;
    spline_x = (float*) malloc(spline_len * sizeof(float));
    spline_y = (float*) malloc(spline_len * sizeof(float));
    memset(spline_y,0,spline_len * sizeof(float));

    for(int i=0;i<spline_len;i++){
        spline_x[i] = ((float)i) * 0.05;
    }


    for(int i = 0;i<len;i++){
        x[i] = i * 20;
        spline_y[x[i]] = cos(spline_x[x[i]]);
    }

    

    

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_x,spline_len*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_y,spline_len*sizeof(float)));


    CUDA_SAFE_CALL( cudaMemcpy( d_spline_x, spline_x, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_spline_y, spline_y, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_check_point_idx, x, memSize, cudaMemcpyHostToDevice));


#if defined(USE_CUDPP)
    result = cudppCreate(&theCudpp);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        goto end;
    }
#endif

    /* ================================
    GPU preparing parameters
    ==================================*/
    timer.reset();
    timer.start();
    preparing_parameters_gpu<float>(d_a,d_b,d_c,d_d,d_spline_x,d_diff,d_spline_y,d_check_point_idx,len,SYSTEM_SIZE,numOfblocks);
    cudaThreadSynchronize();
    timer.stop();
    println("preparing_parameters_gpu costs:%f",timer.getTime());

    

    

    for(int i =0;i<5;i++){
        AddressPair_F addr = addressPairs[i];
        cudaError_t result = cudaMemcpy(addr.host , addr.device, memSize, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(result));
            goto end;
        }
    }

    /* ===============================
    CPU preparing parameters
    =================================*/
#if 1
    
    timer.reset();
    timer.start();
    preparing_parameters_cpu<float>(a,b,c,d,spline_x,h,spline_y,x,len,SYSTEM_SIZE,numOfblocks);
    
    timer.stop();
    println("preparing_parameters_cpu costs:%f",timer.getTime());
    

#endif
    
    /*
    printArray(a,len);
    printArray(a2,len);
    printArray(b,len);
    printArray(b2,len);
    printArray(c,len);
    printArray(c2,len);
    printArray(d,len);
    printArray(d2,len);
    */
    // printArray(y,len);

/*============================
    GPU Solve Tridiagonal Matrix
============================*/
#if defined(USE_CUSPARSE)
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS){
        println("CUSPARSE Library initializing failed.");
    }
    d_m = d_d;
    timer.reset();
    timer.start();
    status = cusparseSgtsv(
        handle,
        len,1,
        d_a,
        d_b,
        d_c,
        d_d,
        len
    );
    cudaThreadSynchronize();
    timer.stop();
    if( status != CUSPARSE_STATUS_SUCCESS){
        println("solve tridiagonal system failed.%d",status);
    }
    println("solve tridiagonal system cost:%f ms",timer.getTime());
    CUDA_SAFE_CALL( cudaMemcpy( m, d_m, memSize, cudaMemcpyDeviceToHost));
    //printArray(m,len);
#endif

#if defined(USE_CUDPP)
    result = cudppPlan(theCudpp, &tridiagonalPlan, config, 0, 0, 0);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan here\n");
        goto end;
    }
    println("system size:%d num of system:%d",min(SYSTEM_SIZE,len),numOfSystem);

    err = cudppTridiagonal(tridiagonalPlan, 
                               d_a, 
                               d_b, 
                               d_c, 
                               d_d, 
                               d_m, 
                               min(SYSTEM_SIZE,len), 
                               numOfSystem);

    cudaThreadSynchronize();
    if (err != CUDPP_SUCCESS) 
    {
        printf("Error running cudppTridiagonal\n");
        goto end;
    }
    
    if( numOfSystem * SYSTEM_SIZE < len){
        offset = numOfSystem * SYSTEM_SIZE;
        println("residual system size: %d",len - offset);
        err = cudppTridiagonal(tridiagonalPlan,
                d_a + offset,
                d_b + offset,
                d_c + offset,
                d_d + offset,
                d_m + offset,
                len - offset,
                1
            );
        cudaThreadSynchronize();
        if (err != CUDPP_SUCCESS) 
        {
            printf("Error running cudppTridiagonal\n");
            goto end;
        }
    }
    CUDA_SAFE_CALL( cudaMemcpy(m2,d_m,memSize,cudaMemcpyDeviceToHost) );

    // printArray(m2,len);
#endif
//Cubic Spline
#if 1
    part_i = 0;

    
    


    // CUDA_SAFE_CALL( cudaMemcpy( d_spline_x, spline_x, spline_len * sizeof(float), cudaMemcpyHostToDevice));

    
    segment_start = 1;
    segment_end =1;
    part_i = 0;
    timer.reset();
    timer.start();
    /* =============================
        plan 1, kernel only handle single segment, repeatly call kernel to finish all spline. Slow
        =============================== */
    // for(int i =0;i<spline_len;i++){
    //     // println("spline_x:%f, x:%f",spline_x[i],x[part_i]);
    //     if( spline_x[i] == x[part_i] || i == spline_len -1) {
    //         segment_end = i;
    //         // println("segment [%d,%d) - %d",segment_start,segment_end,part_i);
    //         if(segment_end - segment_start > 0 && part_i > 0){
    //             _cubic_spline_segment<<< blockalign(segment_end - segment_start,512),512 >>>(
    //             d_x + part_i - 1,
    //             d_y + part_i - 1,
    //             d_m + part_i - 1,
    //             d_diff + part_i,
    //             d_spline_x + segment_start,
    //             d_spline_y + segment_start,
    //             segment_end - segment_start
    //             );
                
    //         }
    //         part_i ++;
    //         segment_start = i;
            
    //     }
    // }
    /* ==============================
        Plan 2, block handle entire segment, stride tought all segments.
    =================================*/
    println("before spline");       
    // printArray(spline_y,spline_len);
    // printArrayFmt(x,len,%d);
    _cubic_spline2<<< 28,SYSTEM_SIZE>>>(d_spline_x,d_spline_y,d_check_point_idx,d_m,d_diff,len);
    cudaThreadSynchronize();
    timer.stop();
    println("gpu cubic spline costs: %f",timer.getTime());

    

    CUDA_SAFE_CALL( cudaMemcpy( spline_y,d_spline_y, spline_len * sizeof(float), cudaMemcpyDeviceToHost));
    // spline_y[spline_len-1] = y[len -1];
    // printArray(spline_y,spline_len);
/* =============================
CPU Cubic Spline Interplation
===============================*/
#if 0

    

    memset(spline_y,0,spline_len * sizeof(float));

    for(int i=0;i<spline_len;i++){
        spline_x[i] = ((float)i) * 0.05;
    }


    for(int i = 0;i<len;i++){
        x[i] = i * 20;
        spline_y[x[i]] = cos(spline_x[x[i]]);
    }


    printf("preparing data done.\n");
    residual = len - min(SYSTEM_SIZE,len) * numOfSystem;

    testTridiagonalDataType<float>(a,b,c,d,m,min(SYSTEM_SIZE,len),numOfSystem,config);
    if(residual > 0){
        offset = min(SYSTEM_SIZE,len) * numOfSystem;
        testTridiagonalDataType<float>(a+offset,b+offset,c+offset,d+offset,m+offset,residual,1,config);
    }

    // printArray(spline_y,spline_len);
    part_i = 0;
    timer.reset();
    timer.start();
    for(int i =0;i<spline_len;i++){
        float x_i = spline_x[i];
        if ( x_i >= spline_x[x[part_i]] ){
            part_i ++;
        }
        float x_part_i = spline_x[x[part_i]];
        float y_part_i = spline_y[x[part_i]];
        float x_part_i_1 = spline_x[x[part_i -1]];
        float y_part_i_1 = spline_y[x[part_i -1]];

        spline_y[i] = m[part_i-1] * powf(x_part_i - x_i,3) / (6*h[part_i]) 
        + m[part_i] * powf(x_i - x_part_i_1,3) / (6 *h[part_i]) 
        + (y_part_i_1  - m[part_i -1 ] * powf(h[part_i],2) / 6) * (x_part_i - x_i) / h[part_i] 
        + (y_part_i     - m[part_i]     * powf(h[part_i],2) / 6) * (x_i - x_part_i_1) / h[part_i]
        ;
        // printf("%f,",spline[i]);
        // println("i=%d,part_i=%d,x_i=%f",i,part_i,x_i);
    }
    timer.stop();
    println("cpu cubic spline costs: %f",timer.getTime());
#endif
    // printArray(m,len);
    // printArray(m2,len);

    for(int i =0;i<len;i++){
        a_error += fabs(a[i] - a2[i]);
        b_error += fabs(b[i] - b2[i]);
        c_error += fabs(c[i] - c2[i]);
        d_error += fabs(d[i] - d2[i]);
        m_error += fabs(m[i] - m2[i]);
    }
    
    println("errors: \na:%f \nb:%f\nc:%f\nd:%f\nm:%f",a_error,b_error,c_error,d_error,m_error);
    
    // printArray(spline_x,spline_len);
    // printArray(spline_y,spline_len);
    // printArrayFmt(x,len,%d);
#endif
end:
    /*
    Free Memories
    */
#ifdef USE_CUDPP 
    cudppDestroy(theCudpp);
#endif
    println("free host memories");
    free(a);
    free(b);
    free(c);
    free(d);
    free(h);
    free(m);

    free(a2);
    free(b2);
    free(c2);
    free(d2);
    free(h2);
    free(m2);

    free(x);
    free(y);
    free(spline_x);
    free(spline_y);
    println("free device memories");
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFree(d_d));
    // CUDA_SAFE_CALL(cudaFree(d_x));
    // CUDA_SAFE_CALL(cudaFree(d_y));
    // CUDA_SAFE_CALL(cudaFree(d_m));
    CUDA_SAFE_CALL(cudaFree(d_diff));
    CUDA_SAFE_CALL(cudaFree(d_spline_x));
    CUDA_SAFE_CALL(cudaFree(d_spline_y));


}
