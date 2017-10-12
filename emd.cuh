#ifndef _EMD_
#define _EMD_
#include "common.cuh"
#include "cubic_spline.cuh"
#include "cudpp.h"

#define DEBUG 

template <typename T>
void sifting(T * d_data,T * d_h,const int len);


template <typename T>
void emd(T * idata,T** imfs,const unsigned int len,unsigned int max_num_of_imf)
{
	T * d_h = NULL;
	T * d_last_h = NULL;
	T * d_sd = NULL;
	T * temp = NULL;
	T res = T();
#ifdef DEBUG
	int rounds = 0;
	T * imf = NULL;
	T * sd = NULL;

	automalloc(imf,len,T);
	automalloc(sd,len,T);
#endif

	automallocD(d_h,len,T);
	automallocD(d_last_h,len,T);
	automallocD(d_sd,len,T);

	CUDA_SAFE_CALL( cudaMemcpy(d_last_h,idata,len * sizeof(T),cudaMemcpyDeviceToHost) );

	do{
		sifting<T>(d_last_h, d_h, len);
		_sd<T><<< 28 * 2, 512>>>(d_h, d_last_h, d_sd, len);
		cudaThreadSynchronize();
#ifdef DEBUG
	CUDA_SAFE_CALL( cudaMemcpy(sd,d_sd,len * sizeof(T),cudaMemcpyDeviceToHost) );
	printArray(sd,len);
#endif
		res = sum<T, 28, 512>(d_sd, len );
// swap d_last_h with d_h
		temp = d_last_h;
		d_last_h = d_h;
		d_h = temp;
#ifdef DEBUG
		println("h_%d, sd=%f",rounds++,res);
#endif
	} while (res > 1e-2);

#ifdef DEBUG
	println("got one imf");
	CUDA_SAFE_CALL( cudaMemcpy(imf,d_last_h,len * sizeof(T),cudaMemcpyDeviceToHost) );
	printArray(imf,len);
#endif

	CUDA_SAFE_CALL( cudaMemcpy(imfs[0],d_last_h,len * sizeof(T),cudaMemcpyDeviceToHost) );

finally:
#ifdef DEBUG
	free(imf);
	free(sd);
#endif
	autofreeD(d_h);
	autofreeD(d_last_h);
	autofreeD(d_sd);

}

template <typename T>
void sifting(T * d_data,T * d_h,const int len)
{
	int * d_maxima = NULL;
	int * d_minima = NULL;
	T* d_maxima_y = NULL;
	T* d_minima_y = NULL;
	T* d_diff = NULL;
	int halflen = (len >> 1) + 2;
	int * d_counter = NULL;
	int * counter = NULL;
	int temp = len -1;
	
	T* d_top = NULL;
	T* d_bottom = NULL;
	T el = T();
	T er = T();

	CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEYS_ONLY;
    CUDPPResult result = CUDPP_SUCCESS;  
    CUDPPHandle theCudpp;
    CUDPPHandle plan;   
    



#ifdef DEBUG
	int * maxima = NULL;
	T * minima = NULL;
	T * diff = NULL;
	T* top = NULL;
	T* bottom = NULL;
	
	automalloc(diff,len,T);
	automalloc(top,len,T);
	automalloc(bottom,len,T);
	automalloc(maxima,halflen,int);
	automalloc(minima,halflen,T);
#endif
	
	automallocD(d_maxima,halflen,int);
	automallocD(d_minima,halflen,int);
	automallocD(d_maxima_y,halflen,T);
	automallocD(d_minima_y,halflen,T);
	automallocD(d_diff,len,T); 
	automalloc(counter,2,int);
	automallocD(d_counter,2,int);
	automallocD(d_top,len,T);
	automallocD(d_bottom,len,T);

	CUDA_SAFE_CALL( cudaMemset(d_maxima,0,(halflen) * sizeof(int)));
	CUDA_SAFE_CALL( cudaMemset(d_minima,0,(halflen) * sizeof(int)));
	CUDA_SAFE_CALL( cudaMemset(d_maxima_y,0,(halflen) * sizeof(T)));
	CUDA_SAFE_CALL( cudaMemset(d_minima_y,0,(halflen) * sizeof(T)));
	CUDA_SAFE_CALL( cudaMemset(d_diff,0,(len) * sizeof(T)));
	// CUDA_SAFE_CALL( cudaMemset(d_out,0,(len) * sizeof(T)));

	result = cudppCreate(&theCudpp);
	result = cudppPlan(theCudpp, &plan, config, halflen, 1, 0);    

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        goto finally;
    }

	_diff<T><<< 28*2,512>>>(d_data,d_diff,len);
	cudaThreadSynchronize();

#ifdef DEBUG_3
	CUDA_SAFE_CALL( cudaMemcpy(diff,d_diff,(len) * sizeof(T),cudaMemcpyDeviceToHost) );	
	printArray(diff,len);
#endif
	CUDA_SAFE_CALL( cudaMemset(d_counter,0,2 * sizeof(int)) );
	_extrema<T><<< 28*2 , 512 >>>(d_data,d_diff,len,&d_maxima[1],&d_maxima_y[1],&d_minima[1],&d_minima_y[1],d_counter);
	cudaThreadSynchronize();

	CUDA_SAFE_CALL( cudaMemcpy(counter,d_counter,(2) * sizeof(int),cudaMemcpyDeviceToHost) );
	counter[0] += 2;
	counter[1] += 2;

#ifdef DEBUG
	printArrayFmt(counter,2,%d);
	println("len:%d",len);
#endif
	temp = 0;
	// CUDA_SAFE_CALL( cudaMemset(d_maxima,0,(1) * sizeof(T)));
	CUDA_SAFE_CALL( cudaMemcpy(d_maxima,&temp,(1) * sizeof(int),cudaMemcpyHostToDevice) );	
	CUDA_SAFE_CALL( cudaMemcpy(d_minima,&temp,(1) * sizeof(int),cudaMemcpyHostToDevice) );	
	// CUDA_SAFE_CALL( cudaMemset(d_maxima + counter[0] + 1,len-1,(1) * sizeof(int)));
	temp = len -1;
	CUDA_SAFE_CALL( cudaMemcpy(d_maxima + counter[0] -1,&temp,(1) * sizeof(int),cudaMemcpyHostToDevice) );	
	CUDA_SAFE_CALL( cudaMemcpy(d_minima + counter[1] -1,&temp,(1) * sizeof(int),cudaMemcpyHostToDevice) );	

	result =  cudppRadixSort(plan, d_maxima, NULL, counter[0]);
	if( result != CUDPP_SUCCESS){
		println("error while sort d_maxima");
	}
	result = cudppRadixSort(plan, d_minima, NULL, counter[1]);	
	if( result != CUDPP_SUCCESS){
		println("error while sort d_minima ");
	}




	_linear_extrapolation<T><<< 1,2>>>(d_data,len,d_maxima,d_maxima_y,counter[0],d_minima,d_minima_y,counter[1]);
	cudaThreadSynchronize();

#ifdef DEBUG
	CUDA_SAFE_CALL( cudaMemcpy(maxima,d_maxima,counter[0] * sizeof(int),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(minima,d_maxima_y,counter[0] * sizeof(T),cudaMemcpyDeviceToHost) );
	printArrayFmt(maxima,counter[0],%d);
	printArrayFmtAlias(minima,maxima_y,counter[0],%f);

	CUDA_SAFE_CALL( cudaMemcpy(maxima,d_minima,counter[1] * sizeof(int),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(minima,d_minima_y,counter[1] * sizeof(T),cudaMemcpyDeviceToHost) );
	printArrayFmtAlias(maxima,minima,counter[1],%d);
	printArrayFmtAlias(minima,minima_y,counter[1],%f);
#endif





	cubic_spline_gpu<T,512>(d_maxima,d_maxima_y,counter[0],d_top,len);
	cudaThreadSynchronize();
	cubic_spline_gpu<T,512>(d_minima,d_minima_y,counter[1],d_bottom,len);
	cudaThreadSynchronize();

#ifdef DEBUG
	CUDA_SAFE_CALL( cudaMemcpy(top,d_top,len * sizeof(T),cudaMemcpyDeviceToHost) );		
	CUDA_SAFE_CALL( cudaMemcpy(bottom,d_bottom,len * sizeof(T),cudaMemcpyDeviceToHost) );		

	printArray(top,len);
	printArray(bottom,len);
#endif

	_mean<T><<< 28*2,512>>>(d_top,d_bottom,len);
	cudaThreadSynchronize();
	_residual<T><<< 28*2,512 >>>(d_data,d_top,d_h,len);
	cudaThreadSynchronize();

#ifdef DEBUG
	CUDA_SAFE_CALL( cudaMemcpy(top,d_h,len * sizeof(T),cudaMemcpyDeviceToHost) );		
	// println("_residual");
	printArrayFmtAlias(top,residual,len,%f);
#endif

finally:
	cudppDestroyPlan(plan);
	cudppDestroy(theCudpp);
	autofreeD(d_maxima);
	autofreeD(d_maxima_y);
	autofreeD(d_minima);
	autofreeD(d_minima_y);
	autofreeD(d_counter);
	autofreeD(d_diff);
	autofreeD(d_bottom);
	autofreeD(d_top);
	free(counter);
	
#ifdef DEBUG
	free(top);
	free(maxima);
	free(minima);
	free(diff);
	free(bottom);
#endif
	

}

#endif