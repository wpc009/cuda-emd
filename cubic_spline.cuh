#ifndef _CUBIC_SPLINE_
#define _CUBIC_SPLINE_
#include "cusparse.h"
#include "common.cuh"



// #define DEBUG 

template <typename T>
__global__ void _prepare_systems(T* a, T* b, T* c, T* d, const T* x, const T* y, T* h, const int n)
{
    int ti_base = blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ T h[blockDim+2];

    //grid-stride
    for (int ti = ti_base; ti < n; ti += blockDim.x * gridDim.x) {
        // if (ti >= n)
            // return;

        if (ti == 0) {
            //head
            d[ti] = 6.0f * (y[1] - y[0]) / (x[1] - x[0]) / (x[1] - x[0]);
            h[ti] = 0.0f;
        } else if (ti < n - 1) {
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

            for (int j = 0; j < 3; j++) {
                d[ti] += 6.0f * y[ti - 1 + j] / (x[ ti - 1 + j] - x[ti - 1 + (j + 1) % 3]) / (x[ ti - 1 + j] - x[ti - 1 + (j + 2) % 3]);
            }
            h[ti] = x[ti] - x[ti - 1];

        } else {
            //last ti = n-1

            d[ti] = 6.0f * (0.0f - (y[ ti] - y[ti - 1]) / (x[ti] - x[ti - 1]) ) / (x[ti] - x[ti - 1]);
            h[ti] = x[ ti] - x[ti - 1];
        }

        if (threadIdx.x == blockDim.x - 1 && ti < n - 1) {
            //block can not be synchronized, so calculate the h[ti + 1] for the last ti in block. do not exceed n -1.
            h[ti + 1]  = x[ti + 1] - x[ti];
        }

        __syncthreads();
        b[ti] = 2.0f;
        a[ti] = ti < n - 1 ? h[ti] / (h[ti] + h[ti + 1]) : 1.0f;
        c[ti] = ti < n - 1 ? h[ti + 1 ] / (h[ti + 1] + h[ti]) : 0.0f;
    }
}

template <typename T>
__global__ void _prepare_systems(T* a,T* b,T* c,T* d,const int* check_point_idx,const T* y,T* h,const int n)
{
    int ti_base = blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ T h[blockDim+2];

    //grid-stride
    for(int ti = ti_base;ti < n;ti += blockDim.x * gridDim.x){



    if (ti >= n)
        return;

    if(ti == 0){
        //head

        d[ti] = 6.0f*(y[1] - y[0])/(check_point_idx[1] - check_point_idx[0])/ (check_point_idx[1] - check_point_idx[0]);
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
            d[ti] += 6.0f * y[ti-1 + j] / (check_point_idx[ ti-1 + j] - check_point_idx[ti-1 + (j+1)%3]) / (check_point_idx[ ti-1+j] - check_point_idx[ti-1 + (j+2)%3]);
        }
        h[ti] = check_point_idx[ti] - check_point_idx[ti-1]; 
        
    }else{
        //last ti = n-1
        
        d[ti] = 6.0f * (0.0f - (y[ti] - y[ti-1])/(check_point_idx[ti] - check_point_idx[ti-1]) ) / (check_point_idx[ti] - check_point_idx[ti-1]);
        h[ti] = check_point_idx[ ti] - check_point_idx[ti-1];
    }

    if(threadIdx.x == blockDim.x -1 && ti < n - 1){
        //block can not be synchronized, so calculate the h[ti + 1] for the last ti in block. do not exceed n -1.
        h[ti + 1]  = check_point_idx[ti + 1] - check_point_idx[ti];
    }
    
    __syncthreads();        
    b[ti] = 2.0f;
    a[ti] = ti < n - 1 ? h[ti] / (h[ti] + h[ti +1]) : 1.0f;
    c[ti] = ti < n - 1 ? h[ti +1 ] / (h[ti+1] + h[ti]):0.0f;
    }
}

//deprecated
template <typename T>
__global__ void _cubic_spline_segment(const T* x,const T* y ,const T* m,const T* h,const T* spline_x,T* spline_y, int segment_len){
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if ( ti >= segment_len)
        return;

    T temp1 = x[1] - spline_x[ti];
    T temp2 = spline_x[ti] - x[0];
    spline_y[ti] = m[0] * powf(temp1,3) / (6 * h[0])
    + m[1] * powf(temp2,3) / (6*h[0])
    + (y[0] - m[0]*powf(h[0],2)/6) * (temp1) / h[0]
    + (y[1] - m[1]*powf(h[0],2)/6) * (temp2) / h[0]
    ;
}

/*
 * cubic spline version 1
 * Each block handle one segmentation a time.
 * when num of threads in a block is larger than points in a segment. The larger indexed threads will be wasted.
 */
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

/*
 * Cubci Spline Version 2
 * Revert the role of block and thread. use the assumption of the num of blocks much smaller then num of threads.
 * Much faster than version 1. More efficient compare to version 1.
 */
template <typename T>
__global__ void _cubic_spline2(const T* x,const T* y,T* spline_out,const T* m,const T* h,int totalSize)
{
    
    for ( int i = threadIdx.x; i < totalSize; i += blockDim.x)
    {   

        int to_idx = i;
        int from_idx = 0;
        if ( i > 0 ){
            from_idx = i - 1;
        }
        spline_out[to_idx] = y[to_idx];
        spline_out[from_idx] = y[from_idx];

        for( int j = (int)x[from_idx] + blockIdx.x + 1; j < (int)x[to_idx]; j+= gridDim.x)
        {            
            spline_out[j] = m[i-1] * powf(x[to_idx] - (T)j,3) / (6*h[i])
                + m[i] * powf((T)j - x[from_idx],3) / (6*h[i])
                + (y[from_idx] - m[i-1] * powf(h[i],2)/6) * (x[to_idx] - (T)j) / h[i]
                + (y[to_idx] - m[i] * powf(h[i],2)/6) * ((T)j - x[from_idx]) / h[i]
                ;   
        }
    }
}

template <typename T>
__global__ void _cubic_spline2(const int* check_point_idx,const T* y,T* spline_out,const T* m,const T* h,int totalSize)
{
    for ( int i = threadIdx.x; i < totalSize; i += blockDim.x)
    {
        int to_idx = check_point_idx[i];
        int from_idx = 0;
        
        spline_out[to_idx] = y[i];
        if ( i > 0) {
            from_idx = check_point_idx[i - 1] ;
            // spline_out[i - 1] = y[i - 1];

            for ( int j = from_idx + blockIdx.x + 1 ; j < to_idx; j += gridDim.x)
            {

                spline_out[j] = m[i - 1] * powf(to_idx - j, 3) / (6 * h[i])
                + m[i] * powf(j - from_idx, 3) / (6 * h[i])
                + (y[i - 1] - m[i - 1] * powf(h[i], 2) / 6) * (to_idx - j) / h[i]
                + (y[i] - m[i] * powf(h[i], 2) / 6) * (j - from_idx) / h[i]
                ;
            }
        }

    }
}

template <typename T>
__global__ void _get_group_idx(const int* check_point_idx,int totalSize,int * group_index)
{
    uint ti = threadIdx.x;
    uint i = ti + blockIdx.x * blockDim.x;
    int middle = 0;
    int start = 0;
    int end = totalSize - 1;
    int num_of_blocks = (check_point_idx[end] + blockDim.x -1) / blockDim.x;
    if( i > check_point_idx[end])
        return;

    for(uint i = ti + blockIdx.x * blockDim.x;i < num_of_blocks;i += gridDim.x * blockDim.x)
    {
            start = 0;
            end = totalSize - 1;
            do {
                middle = ((end - start) >> 1) + start;
                if (check_point_idx[middle] <= i * blockDim.x)
                {
                    start = middle;
                } else {
                    end = middle;
                }
            } while (end - start > 1);

            group_index[i] = start;
    }
    
}

template <typename T>
__global__ void _cubic_spline3(const int* check_point_idx,const T* y,T* spline_out,const T* m,const T* h,int totalSize,int* group_index)
{
    extern __shared__ T sdata[];
    uint tid = threadIdx.x;
    uint i;
    int middle = 0;
    int start = 0;
    int end = totalSize - 1;
    int num_of_blocks = (check_point_idx[end] + blockDim.x -1) / blockDim.x;
    int tail = check_point_idx[end];

    // sdata[ti] = totalSize - 1;

    for (uint bid = blockIdx.x; bid < num_of_blocks; bid += gridDim.x)
    {
        i = tid + bid * blockDim.x; 

        if( i > tail)
            return;

        if ( bid  == num_of_blocks - 1) {
            end = totalSize - 1;
        } else {
            end = min(group_index[bid + 1] + 1, totalSize - 1);
        }
        start = group_index[bid];

        do {
            middle = ((end - start) >> 1) + start;
            if (check_point_idx[middle] <= i)
            {
                start = middle;
            } else {
                end = middle;
            }
        } while (end - start > 1);

        int to_idx = end;
        int from_idx = start;
        sdata[tid * 4 ] = check_point_idx[start];
        sdata[tid * 4 + 1] = check_point_idx[end];
        sdata[tid * 4 + 2] = h[to_idx]; // 6
        sdata[tid * 4 + 3] = sdata[tid * 4 + 2] * sdata[tid* 4 + 2];
        //m[from_idx] //2
        //m[to_idx] //2
        //y[from_idx] //1
        //y[to_idx] //1



        if ( i == sdata[tid]) {
            spline_out[i] = y[start];
        } else if (i == sdata[tid + blockDim.x])
        {
            spline_out[i] = y[end];
        } else
        {

            spline_out[i] =  m[from_idx] * powf(sdata[tid*4 + 1] - i, 3) / (6 * sdata[tid * 4 + 2])
                             + m[to_idx] * powf(i - sdata[tid * 4], 3) / (6 * sdata[tid * 4 + 2])
                             + (y[from_idx] - m[from_idx] * sdata[tid * 4 + 3] / 6) * (sdata[tid * 4 + 1] - i) / sdata[tid * 4 + 2]
                             + (y[to_idx] - m[to_idx] * sdata[tid * 4 + 3] / 6) * (i - sdata[tid * 4]) / sdata[tid * 4 + 2]
                             ;
        }
    }

}




/**
 *   deprecated
 */
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
    
    // _prepare_systems<T><<<28*4,1024>>>(d_a,d_b,d_c,d_d,d_x,d_y,d_check_point_idx,d_diff,len);
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
        T* sub_b = b + base;
        T* sub_a = a + base;
        T* sub_c = c + base;
        T* sub_d = d + base;
        T* sub_diff = diff + base;
        // const T* sub_x = x + base;
        const int* sub_x_idx = check_point_idx + base;
        const T* sub_y = y + base;
        int last = len -1;
        for(int i=0;i<systemSize && i+base < len;i++)
        {

            sub_b[i] = 2.0f;
            if( i + base ==0){
                sub_a[i] = 0.0f;
                sub_c[i] = 1.0f;
                sub_d[i] = 6.0f*(y[sub_x_idx[1]] - y[sub_x_idx[0]])/(x[sub_x_idx[1]] - x[sub_x_idx[0]])/ (x[sub_x_idx[1]] - x[sub_x_idx[0]]);
                
            }else if (i + base < last){
                sub_a[i] = sub_diff[i] / (sub_diff[i] + sub_diff[i+1]);
                sub_c[i] = sub_diff[i+1] / (sub_diff[i] + sub_diff[i+1]);
                sub_d[i] = 0.0f;
                for(int j=0;j<3;j++){
                    sub_d[i] += 6.0f * y[sub_x_idx[i-1 + j]] / (x[sub_x_idx[i-1 + j]] - x[sub_x_idx[i-1 + (j+1)%3]]) / (x[sub_x_idx[i-1+j]] - x[sub_x_idx[i-1 + (j+2)%3]]);
                }
            }else{                
                sub_d[i] = 6.0f * (0.0f - (y[sub_x_idx[i]] - y[sub_x_idx[i-1]])/(x[sub_x_idx[i]] - x[sub_x_idx[i-1]]) ) / (x[sub_x_idx[i]] - x[sub_x_idx[i-1]]);
                sub_a[i] = 1.0f;
                sub_c[i] = 0.0f;
            }
#ifdef DEBUG
            if(sub_d[i] != sub_d[i]){
                println("d[i+base] = %f, i+base=%d,i=%d",sub_d[i],i+base,i);
            }
#endif
        }
    }

}

/*

Memory Demand: n x 5 x sizeof(T) + min(n,8) ×(4)×sizeof(T) + N x sizeof(T)

n is the num of check points.
N is the num of spline points.
usally n << N.
*/
template <typename T,int const systemSize>
int cubic_spline_gpu(const T* d_data_x,const T* d_data_y,int check_point_len,T* d_spline_out,int data_len)
{
    T * d_a = NULL;
    T * d_b = NULL;
    T * d_c = NULL;
    T * d_m = NULL; 
    T * d_diff = NULL;
    T * d_group_idx = NULL;



    int memSize = check_point_len * sizeof(T);
    int numOfBlocks = blockalign(check_point_len,systemSize);
    cusparseHandle_t handle = 0;
    cusparseStatus_t status;
    int res = 0;

#ifdef DEBUG
    T * m = (T*) malloc(memSize);
    T * a = (T*) malloc(memSize);
    T * b = (T*) malloc(memSize);
    T * c = (T*) malloc(memSize);
    T * h = (T*) malloc(memSize);
#endif

    
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_diff,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_a,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_b,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_c,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_m,memSize) );

    automallocD(d_group_idx,30,int);

    CUDA_SAFE_CALL( cudaMemset(d_a,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_b,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_c,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_m,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_diff,0,memSize) );

    CUDA_SAFE_CALL( cudaMemset(d_spline_out,0,data_len * sizeof(T)) );

    
    
    // preparing_parameters_gpu<T>(d_a,d_b,d_c,d_m,d_data_x,d_diff,d_data_y,d_check_point_idx,check_point_len,
    //     systemSize,numOfBlocks);
    _prepare_systems<T><<<28*2,512>>>(d_a,d_b,d_c,d_m,d_data_x,d_data_y,d_diff,check_point_len);
    cudaThreadSynchronize();
#ifdef DEBUG
    CUDA_SAFE_CALL( cudaMemcpy(m,d_m, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(a,d_a, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(b,d_b, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(c,d_c, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(h,d_diff, memSize,cudaMemcpyDeviceToHost) );
    

    printArray(m,check_point_len);
    printArray(a,check_point_len);
    printArray(b,check_point_len);
    printArray(c,check_point_len);
    printArray(h,check_point_len);
#endif

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS){
        println("CUSPARSE Library initializing failed.");
        res = -1;
        goto finalize;
    }

    
    status = cusparseSgtsv(
        handle,
        check_point_len,1,
        d_a,
        d_b,
        d_c,
        d_m,
        check_point_len
    );
    cudaThreadSynchronize();

#ifdef DEBUG
    CUDA_SAFE_CALL( cudaMemcpy(m,d_m, memSize,cudaMemcpyDeviceToHost) );
    printArray(m,check_point_len);
#endif
    
    if( status != CUSPARSE_STATUS_SUCCESS){
        println("solve tridiagonal system failed.%d",status);
        res = -1;
        goto finalize;
    }
    
    // _cubic_spline2<T><<< 28*2,systemSize >>>(d_data_x,d_data_y,d_spline_out,d_m,d_diff,check_point_len);

    _cubic_spline3<T><<< 28,512 >>>(d_data_x,d_data_y,d_spline_out,d_m,d_diff,check_point_len,d_group_idx);
    cudaThreadSynchronize();


/*
 * Finalize
 */
 finalize:
    cusparseDestroy(handle);
    
    autofreeD(d_diff);
#ifdef DEBUG
    free(m);
    free(a);
    free(b);
    free(c);
#endif
    autofreeD(d_a);
    autofreeD(d_b);
    autofreeD(d_c);
    autofreeD(d_m);
    autofreeD(d_group_idx);

    return res;
}


template <typename T,int const systemSize>
int cubic_spline_gpu(const int * d_check_point_idx,const T* d_data_y,const int check_point_len,T* d_spline_out,const int data_len)
{
    T * d_a = NULL;
    T * d_b = NULL;
    T * d_c = NULL;
    T * d_m = NULL; 
    T * d_diff = NULL;
    int * d_group_idx = NULL;



    int memSize = check_point_len * sizeof(T);
    int numOfBlocks = blockalign(check_point_len,systemSize);
    cusparseHandle_t handle = 0;
    cusparseStatus_t status;
    int res = 0;

#ifdef DEBUG
    T * m = (T*) malloc(memSize);
    T * a = (T*) malloc(memSize);
    T * b = (T*) malloc(memSize);
    T * c = (T*) malloc(memSize);
    T * h = (T*) malloc(memSize);
#endif

    
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_diff,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_a,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_b,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_c,memSize) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_m,memSize) );

    automallocD(d_group_idx,28*2,int);

    CUDA_SAFE_CALL( cudaMemset(d_group_idx,0,28*2 * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(d_a,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_b,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_c,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_m,0,memSize) );
    CUDA_SAFE_CALL( cudaMemset(d_diff,0,memSize) );

    CUDA_SAFE_CALL( cudaMemset(d_spline_out,0,data_len * sizeof(T)) );

    
    
    // preparing_parameters_gpu<T>(d_a,d_b,d_c,d_m,d_data_x,d_diff,d_data_y,d_check_point_idx,check_point_len,
    //     systemSize,numOfBlocks);
    _prepare_systems<T><<<28*2,512>>>(d_a,d_b,d_c,d_m,d_check_point_idx,d_data_y,d_diff,check_point_len);
    cudaThreadSynchronize();
#ifdef DEBUG
    CUDA_SAFE_CALL( cudaMemcpy(m,d_m, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(a,d_a, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(b,d_b, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(c,d_c, memSize,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(h,d_diff, memSize,cudaMemcpyDeviceToHost) );
    

    printArray(m,check_point_len);
    printArray(a,check_point_len);
    printArray(b,check_point_len);
    printArray(c,check_point_len);
    printArray(h,check_point_len);
#endif

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS){
        println("CUSPARSE Library initializing failed.");
        res = -1;
        goto finalize;
    }

    
    status = cusparseSgtsv(
        handle,
        check_point_len,1,
        d_a,
        d_b,
        d_c,
        d_m,
        check_point_len
    );
    cudaThreadSynchronize();

#ifdef DEBUG
    CUDA_SAFE_CALL( cudaMemcpy(m,d_m, memSize,cudaMemcpyDeviceToHost) );
    printArray(m,check_point_len);
#endif
    
    if( status != CUSPARSE_STATUS_SUCCESS){
        println("solve tridiagonal system failed.%d",status);
        res = -1;
        goto finalize;
    }
    
    _get_group_idx<T><<< 28*2,systemSize >>>(d_check_point_idx,check_point_len,d_group_idx);
    cudaThreadSynchronize();
    _cubic_spline3<T><<< 28*2,systemSize,systemSize * 4 * sizeof(T) >>>(d_check_point_idx,d_data_y,d_spline_out,d_m,d_diff,check_point_len,d_group_idx);

    
    // _cubic_spline2<T><<< 28*2,systemSize >>>(d_check_point_idx,d_data_y,d_spline_out,d_m,d_diff,check_point_len);


    cudaThreadSynchronize();


/*
 * Finalize
 */
 finalize:
    cusparseDestroy(handle);
    
    autofreeD(d_diff);
#ifdef DEBUG
    free(m);
    free(a);
    free(b);
    free(c);
#endif
    autofreeD(d_group_idx);
    autofreeD(d_a);
    autofreeD(d_b);
    autofreeD(d_c);
    autofreeD(d_m);

    return res;
}


#endif