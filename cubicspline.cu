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
// #if defined(USE_CUDPP)
#include "cudpp.h"
// #elif defined(USE_CUSPARSE)
#include "cusparse.h"
// #endif

#include <cuda_runtime_api.h>
#include "cuda_util.h"

#define CUDPP_APP_COMMON_IMPL
#include "common_config.h"
#include "stopwatch.h"

#include <string>

#include "emd.cuh"
#include "cubic_spline.cuh"

using namespace cudpp_app;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
void cubicSpline();
void testing(int,int);
void testing2(int,int);
void testCases();

// template <typename T> __global__ void _prepare_systems(T* a,T* b,T* c,T* d,const T* x,const T* y,T* h,const int n);
// template <typename T> void preparing_parameters_gpu(T*,T*,T*,T*,const T*,T*,const T*,const int*,int,int,int);
// template <typename T> void preparing_parameters_cpu(T*,T*,T*,T*,const T*,T*,const T*,const int*,int,int,int);

  


#define TIME(t,msg,code) do {\
    t.reset(); \
    t.start(); \
    code;\
    t.stop();\
    printf(#msg" cost:%f ms\n",t.getTime());\
} while(0)

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


void testMergeSort(){
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEYS_ONLY;
    CUDPPResult result = CUDPP_SUCCESS;  
    CUDPPHandle theCudpp;
    CUDPPHandle plan; 
    int test[]  = { 1,5,20,2,600,30};
    int * d_test = NULL;
    int * d_values = NULL;
    result = cudppCreate(&theCudpp);
    result = cudppPlan(theCudpp,&plan,config,10,1,0);

    automallocD(d_test,10,int);
    automallocD(d_values,10,int);
    CUDA_SAFE_CALL( cudaMemcpy(d_test,test,6 * sizeof(int),cudaMemcpyHostToDevice) );

    result =  cudppRadixSort(plan, d_test, NULL, 6);
    if( result != CUDPP_SUCCESS){
        println("error in merge sort");
    }
    cudaThreadSynchronize();
    CUDA_SAFE_CALL( cudaMemcpy(test,d_test,6 * sizeof(int),cudaMemcpyDeviceToHost) );
    printArrayFmt(test,6,%d);

    autofreeD(d_test);
    cudppDestroyPlan(plan);
    cudppDestroy(theCudpp);

}

void testSum(){
    // int data[] = {7,2,3,4,5};
    float data[] = {
1.000000, 1.001534, 1.006192, 1.014093, 1.025414, 1.040391, 1.059334, 1.082637, 1.110798, 1.144443,
1.184355, 1.231522, 1.287195, 1.352969, 1.430900, 1.523672, 1.634833, 1.769163, 1.933214, 2.136175,
2.391275, 2.718141, 3.146970, 3.726274, 4.538268, 5.731900, 7.601384, 10.799616, 17.037558, 32.133091,
88.189651, 962.687073, 459.412811, 58.715717, 20.718317, 10.099625, 5.781237, 3.642375, 2.445008, 1.716237,
1.244916, 0.925744, 0.701687, 0.539801, 0.420056, 0.329745, 0.260522, 0.206739, 0.164474, 0.130944,
0.104136, 0.082568, 0.065135, 0.050998, 0.039517, 0.030196, 0.022649, 0.016570, 0.011720, 0.007907,
0.004978, 0.002812, 0.001311, 0.000400, 0.000019, 0.000121, 0.000675, 0.001664, 0.003083, 0.004944,
0.007269, 0.010100, 0.013495, 0.017536, 0.022330, 0.028023, 0.034805, 0.042930, 0.052740, 0.064697,
0.079438, 0.097860, 0.121249, 0.151506, 0.191520, 0.245858, 0.322068, 0.433334, 0.604356, 0.885793,
1.395234, 2.458854, 5.280423, 17.603279, 458.042633, 50.944889, 9.615420, 3.988515, 2.194571, 1.398465,
0.975213, 0.722858, 0.559960, 0.448493, 0.368742, 0.309638, 0.264574, 0.229402, 0.201408, 0.178756,
0.160166, 0.144724, 0.131762, 0.120783, 0.111411, 0.103357, 0.096396, 0.090350, 0.085079, 0.080471,
0.076433, 0.072891, 0.069786, 0.067066, 0.064693, 0.062631, 0.060856, 0.059345, 0.058082, 0.057055,
0.056255, 0.055678, 0.055324, 0.055197, 0.055305, 0.055662, 0.056288, 0.057210, 0.058465, 0.060101,
0.062185, 0.064801, 0.068067, 0.072139, 0.077236, 0.083669, 0.091885, 0.102557, 0.116731, 0.136108,
0.163621, 0.204698, 0.270392, 0.386333, 0.624412, 1.260242, 4.300251, 726.236511, 4.991086, 1.051353,
0.416455, 0.211455, 0.122681, 0.077344, 0.051598, 0.035857, 0.025696, 0.018859, 0.014107, 0.010717,
0.008247, 0.006415, 0.005037, 0.003988, 0.003181, 0.002556, 0.002067, 0.001684, 0.001382, 0.001144,
0.000955, 0.000806, 0.000689, 0.000598, 0.000527, 0.000474, 0.000435, 0.000410, 0.000396, 0.000393,
0.000402, 0.000422, 0.000455, 0.000502, 0.000565, 0.000648, 0.000755, 0.000892, 0.001066, 0.001286,
0.001566, 0.001922, 0.002377, 0.002960, 0.003713, 0.004693, 0.005983, 0.007698, 0.010014, 0.013193,
0.017654, 0.024075, 0.033625, 0.048428, 0.072657, 0.115361, 0.199078, 0.392766, 0.994788, 4.770679,
551.039795, 3.964813, 1.170807, 0.581505, 0.360130, 0.252133, 0.190864, 0.152518, 0.126811, 0.108687,
0.095415, 0.085408, 0.077692, 0.071639, 0.066828, 0.062972, 0.059865, 0.057360, 0.055348, 0.053747,
0.052497, 0.051551, 0.050874, 0.050439, 0.050227, 0.050223, 0.050419, 0.050809, 0.051392, 0.052168,
0.053144, 0.054328, 0.055731, 0.057371, 0.059267, 0.061445, 0.063938, 0.066784, 0.070031, 0.073738,
0.077975, 0.082833, 0.088420, 0.094877, 0.102379, 0.111152, 0.121489, 0.133779, 0.148540, 0.166483,
0.188601, 0.216314, 0.251721, 0.298023, 0.360315, 0.447111, 0.573563, 0.768854, 1.095411, 1.707999,
3.085589, 7.406201, 39.441910, 319.908752, 12.883609, 3.850798, 1.778736, 0.999404, 0.627488, 0.422916,
0.299216, 0.219168, 0.164660, 0.126040, 0.097796, 0.076600, 0.060353, 0.047677, 0.037644, 0.029608,
0.023114, 0.017831, 0.013519, 0.010001, 0.007144, 0.004850, 0.003049, 0.001691, 0.000740, 0.000179,
0.000000, 0.000210, 0.000826, 0.001879, 0.003410, 0.005480, 0.008162, 0.011550, 0.015755, 0.020917,
0.027207, 0.034840, 0.044083, 0.055271, 0.068829, 0.085301, 0.105386, 0.129995, 0.160331, 0.197999,
0.245178, 0.304874, 0.381319, 0.480595, 0.611685, 0.788238, 1.031759, 1.377588, 1.886829, 2.672290,
3.959980, 6.256157, 10.887075, 22.273178, 63.125969, 501.552368, 973.458435, 91.533119, 33.411629, 17.696648,
11.196737, 7.864154, 5.916808, 4.674214, 3.829395, 3.227066, 2.781438, 2.441910, 2.177039, 1.966341,
1.796056, 1.656603, 1.541153, 1.444745, 1.363674, 1.295156, 1.237047, 1.187682, 1.145764, 1.110259,
1.080350, 1.055381, 1.034827, 1.018273, 1.005386, 0.995911, 0.989653, 0.986472, 0.986279, 0.989683,
1.000000
    };
    int len = 381;
    float gpu_res = float();
    float cpu_res = float();
    float * d_data = NULL;
for(int j=1;j<len;j++){
    automallocD(d_data,j,float);
    CUDA_SAFE_CALL( cudaMemcpy(d_data,data,j* sizeof(float),cudaMemcpyHostToDevice) );
    gpu_res = sum<float,28,512>(d_data,j);
    cpu_res = data[0];
    for(int i =1;i<j;i++)
    {
        cpu_res += data[i];
    }

    println("gpu sum error = %f,res = %f",(gpu_res - cpu_res) / cpu_res,gpu_res);

    autofreeD(d_data);
}

}
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
    // testMergeSort();
    // testSum();
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
    testing2(
        20 , //num of check points, determin the num of segments. < total blocks
        512 //system size,  the size of a single system. determin the num of threads. < segment size(20)
    );

    // testing(
    //     20,
    //     512
    // );
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void generateFakeData(T* data_x,T* data_y,int len,int * check_points,int check_points_len)
{
    for(int i =0;i< len;i++)
    {
        data_x[i] = ((float)i) * 0.05;
        data_y[i] = cos(data_x[i]);
    }

    for(int i = 0;i<check_points_len;i++)
    {        
        check_points[i] = i * 20;
    }
}

void testing2(int len,int SYSTEM_SIZE){

    float * d_diff;
    float * d_spline_x;
    // float * d_spline_y;
    float * d_spline_out;
    float * d_residual;
    float * d_prev;
    int * d_check_point_idx;

    float * spline_x;
    float * spline_y;
    float * spline_out;

    int * check_points;

    int spline_len = (len -1) * 20 + 1;
    int memSize = len * sizeof(float);
    int error = 0;

    spline_x = (float*) malloc(spline_len * sizeof(float));
    spline_y = (float*) malloc(spline_len * sizeof(float));
    spline_out = (float*) malloc(spline_len * sizeof(float));
    check_points = (int*) malloc(memSize);

    generateFakeData(spline_x,spline_y,spline_len,check_points,len);

    automallocD(d_diff,spline_len,float);
    // automallocD(d_spline_y,spline_len,float);
    automallocD(d_spline_x,spline_len,float);
    automallocD(d_residual,spline_len,float);
    automallocD(d_prev,spline_len,float);
    automallocD(d_spline_out,spline_len,float);
    automallocD(d_check_point_idx,len,int);



    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_diff,spline_len * sizeof(float)));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_x,spline_len * sizeof(float)));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_y,spline_len * sizeof(float)));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_out,spline_len * sizeof(float)));
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &d_check_point_idx,memSize));

    CUDA_SAFE_CALL( cudaMemcpy( d_spline_x, spline_x, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL( cudaMemcpy( d_spline_y, spline_y, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL( cudaMemcpy( d_residual, spline_y, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_prev, spline_y, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_check_point_idx, check_points, len * sizeof(int), cudaMemcpyHostToDevice));

    printArrayFmt(check_points,len,%d);
    printArray(spline_x,spline_len);
    // _diff<float><<<28*2,512 >>>(d_spline_y,d_diff,spline_len);

    // cubic_spline_gpu<float,512>(d_spline_x,d_spline_y,d_spline_out,spline_len,d_check_point_idx,len);

    // CUDA_SAFE_CALL( cudaMemcpy( spline_out, d_spline_out, spline_len * sizeof(float), cudaMemcpyDeviceToHost));

    // printArray(spline_out,spline_len);
    // printArray(spline_y,spline_len);

    // sifting<float>(d_prev,spline_len,d_residual);
    printArray(spline_y,spline_len);

    emd<float>(d_prev,spline_len);


finally:
    free(spline_x);
    free(spline_y);
    free(spline_out);
    free(check_points);
    CUDA_SAFE_CALL( cudaFree(d_diff) );
    CUDA_SAFE_CALL( cudaFree(d_spline_x) );
    // CUDA_SAFE_CALL( cudaFree(d_spline_y) );
    CUDA_SAFE_CALL( cudaFree(d_check_point_idx) );
    CUDA_SAFE_CALL( cudaFree(d_spline_out) );
    autofreeD(d_residual);
    autofreeD(d_prev);
    
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
    int counter[2];



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
    float * d_spline_x_diff;
    int * d_maxima;
    int * d_minima;
    int * d_counter;
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
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_counter,2*sizeof(int)) );



    CUDA_SAFE_CALL( cudaMemset(d_a,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_b,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_c,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_d,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_diff,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_counter,0,2 * sizeof(int)) );

    
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

    int * maxima;
    int * minima;

    

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
    maxima = (int*) malloc((spline_len >> 1) * sizeof(int));
    minima = (int*) malloc((spline_len >> 1) * sizeof(int));

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
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_spline_x_diff,spline_len*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_maxima,(spline_len>>1)*sizeof(int)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_minima,(spline_len>>1)*sizeof(int)));

    CUDA_SAFE_CALL( cudaMemset(d_maxima,-1,(spline_len>>1) * sizeof(int)));
    CUDA_SAFE_CALL( cudaMemset(d_minima,-1,(spline_len>>1) * sizeof(int)));

    CUDA_SAFE_CALL( cudaMemcpy( d_spline_x, spline_x, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_spline_y, spline_y, spline_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_check_point_idx, x, memSize, cudaMemcpyHostToDevice));

    // printArray(spline_x,spline_len);
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
    
    
    printArray(a,len);
    printArray(a2,len);
    printArray(b,len);
    printArray(b2,len);
    printArray(c,len);
    printArray(c2,len);
    printArray(d,len);
    printArray(d2,len);
    printArrayFmt(x,len,%d);
    printArray(spline_x,spline_len);
    printArray(spline_y,spline_len);
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
    printArray(m,len);
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
    // _cubic_spline2<<< 28,SYSTEM_SIZE>>>(d_spline_x,d_spline_y,d_spline_y,d_check_point_idx,d_m,d_diff,len);
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
    

    // timer.reset();
    // timer.start();
    // _diff<float><<<28*2,512 >>>(d_spline_y,d_spline_x_diff,spline_len);
    // CUDA_SAFE_CALL( cudaMemset(d_counter,0,2 * sizeof(int)) );
    // _extrema<float> <<< 28*2,512>>>(d_spline_x_diff,spline_len,d_maxima,d_minima,d_counter);
    // cudaThreadSynchronize();
    // timer.stop();

    // CUDA_SAFE_CALL( cudaMemcpy(counter,d_counter,(2) * sizeof(int),cudaMemcpyDeviceToHost) );
    // printArrayFmt(counter,2,%d);


    // println("_diff kernel costs:%f",timer.getTime());
    // CUDA_SAFE_CALL( cudaMemcpy(maxima,d_maxima,(spline_len>>1) * sizeof(int),cudaMemcpyDeviceToHost) );
    // CUDA_SAFE_CALL( cudaMemcpy(minima,d_minima,(spline_len>>1) * sizeof(int),cudaMemcpyDeviceToHost) );
    // CUDA_SAFE_CALL( cudaMemcpy(spline_y,d_spline_x_diff,(spline_len) * sizeof(int),cudaMemcpyDeviceToHost) );
    // printArrayFmt(maxima,spline_len>>1,%d);
    printArray(spline_y,spline_len);


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

    free(maxima);
    free(minima);

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
    CUDA_SAFE_CALL(cudaFree(d_spline_x_diff));
    CUDA_SAFE_CALL(cudaFree(d_maxima));
    CUDA_SAFE_CALL(cudaFree(d_minima));
    CUDA_SAFE_CALL(cudaFree(d_counter));




}
