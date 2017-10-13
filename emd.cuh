#ifndef _EMD_
#define _EMD_
#include "common.cuh"
#include "cubic_spline.cuh"

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

// #define DEBUG 

/**
Global
*/

template <typename T>
void sifting(T * d_data,T * d_h,const int len,int * d_maxima,T* d_maxima_y,int * d_minima,T * d_minima_y,T * d_diff,T* d_top,T* d_bottom,int * d_counter,int * counter,int halflen,T* d_a,T* d_b,T* d_c,T* d_m,T* d_diff2);


template <typename T>
void emd(T * idata,T** imfs,const int len,int max_num_of_imf)
{
	T * d_h = NULL;
	T * d_last_h = NULL;
	T * d_idata = NULL;
	T * d_sd = NULL;
	T * temp = NULL;
/*
for sifting function
*/
	int * d_maxima = NULL;
	int * d_minima = NULL;
	T* d_maxima_y = NULL;
	T* d_minima_y = NULL;
	T* d_diff = NULL;
	int halflen = (len >> 1) + 2;
	int * d_counter = NULL;
	int * counter = NULL;
	T* d_top = NULL;
	T* d_bottom = NULL;

/*
for cubic_spline interpolation inside sifting 
*/
	T * d_a = NULL;
    T * d_b = NULL;
    T * d_c = NULL;
    T * d_m = NULL; 
    T * d_diff2 = NULL;

 /*
 for sum
 */
 	T * d_acc;
 	T * acc;

	T res = T();
	int imf_idx = 0;

	int gridSize = 28;
	int rounds = 0;


#ifdef DEBUG
	
	T * imf = NULL;
	T * sd = NULL;

	automalloc(imf,len,T);
	automalloc(sd,len,T);
#endif

	automallocD(d_h,len,T);
	automallocD(d_last_h,len,T);
	automallocD(d_sd,len,T);
	automallocD(d_idata,len,T);

	automallocD(d_maxima,halflen,int);
	automallocD(d_minima,halflen,int);
	automallocD(d_maxima_y,halflen,T);
	automallocD(d_minima_y,halflen,T);
	automallocD(d_diff,len,T); 
	automalloc(counter,2,int);
	automallocD(d_counter,2,int);
	automallocD(d_top,len,T);
	automallocD(d_bottom,len,T);

	automallocD(d_a,halflen,T);
	automallocD(d_b,halflen,T);
	automallocD(d_c,halflen,T);
	automallocD(d_m,halflen,T);
	automallocD(d_diff2,halflen,T);

	automallocD(d_acc,gridSize,T);
	automalloc(acc,gridSize,T);

	init_cusparse();

	CUDA_SAFE_CALL( cudaMemcpy(d_idata,idata,len * sizeof(T),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(d_last_h,idata,len * sizeof(T),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemset(d_h,0,len * sizeof(T)) );



	while (imf_idx < max_num_of_imf) {

		_residual<T> <<< gridSize * 2, 512 >>>(d_last_h, d_h, d_last_h, len);
		cudaThreadSynchronize();
		rounds = 0;

		do {
			
			sifting<T>( d_last_h, d_h, len,d_maxima,d_maxima_y,d_minima,d_minima_y,d_diff,d_top,d_bottom,d_counter,counter,halflen,
				d_a,d_b,d_c,d_m,d_diff2
				);

			_sd<T> <<< gridSize * 2, 512>>>(d_h, d_last_h, d_sd, len);
			cudaThreadSynchronize();
#ifdef DEBUG
			CUDA_SAFE_CALL( cudaMemcpy(sd, d_sd, len * sizeof(T), cudaMemcpyDeviceToHost) );
			printArray(sd, len);
#endif
			res = sum<T,512>(d_sd, len ,d_acc,acc,gridSize);
// swap d_last_h with d_h
			temp = d_last_h;
			d_last_h = d_h;
			d_h = temp;
			rounds ++ ;
#ifdef DEBUG
			println("h_%d, sd=%f", rounds, res);
#endif
		} while (res > 1e-5);
		// } while( rounds < 100);
		println("get imf[%d], use %d rounds, res %f",imf_idx,rounds,res);
#ifdef DEBUG
		println("got one imf");
		CUDA_SAFE_CALL( cudaMemcpy(imf, d_last_h, len * sizeof(T), cudaMemcpyDeviceToHost) );
		printArray(imf, len);
#endif

		CUDA_SAFE_CALL( cudaMemcpy(imfs[imf_idx], d_last_h, len * sizeof(T), cudaMemcpyDeviceToHost) );

		imf_idx ++;
	}
finally:
#ifdef DEBUG
	autofree(imf);
	autofree(sd);
#endif
	autofreeD(d_h);
	autofreeD(d_last_h);
	autofreeD(d_sd);
	autofreeD(d_idata);
	autofreeD(d_maxima);
	autofreeD(d_maxima_y);
	autofreeD(d_minima);
	autofreeD(d_minima_y);
	autofreeD(d_counter);
	autofreeD(d_diff);
	autofreeD(d_bottom);
	autofreeD(d_top);

	autofreeD(d_a);
	autofreeD(d_b);
	autofreeD(d_c);
	autofreeD(d_m);
	autofreeD(d_diff2);

	autofreeD(d_acc);
	autofree(acc);
	autofree(counter);

	free_cusparse();

	

}

template <typename T>
void sifting(T * d_data,T * d_h,const int len,int * d_maxima,T* d_maxima_y,int * d_minima,T * d_minima_y,T * d_diff,T* d_top,T* d_bottom,int * d_counter,int * counter,int halflen,
	T* d_a,T* d_b,T* d_c,T* d_m,T* d_diff2
	)
{
	// int * d_maxima = NULL;
	// int * d_minima = NULL;
	// T* d_maxima_y = NULL;
	// T* d_minima_y = NULL;
	// T* d_diff = NULL;
	// int halflen = (len >> 1) + 2;
	// int * d_counter = NULL;
	// int * counter = NULL;
	// T* d_top = NULL;
	// T* d_bottom = NULL;
	int temp = len -1;
	
	
	T el = T();
	T er = T();


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
	automalloc(minima,halflen,int);
#endif
	
	// automallocD(d_maxima,halflen,int);
	// automallocD(d_minima,halflen,int);
	// automallocD(d_maxima_y,halflen,T);
	// automallocD(d_minima_y,halflen,T);
	// automallocD(d_diff,len,T); 
	// automalloc(counter,2,int);
	// automallocD(d_counter,2,int);
	// automallocD(d_top,len,T);
	// automallocD(d_bottom,len,T);


	CUDA_SAFE_CALL( cudaMemset(d_maxima,0,(halflen) * sizeof(int)));
	CUDA_SAFE_CALL( cudaMemset(d_minima,0,(halflen) * sizeof(int)));
	CUDA_SAFE_CALL( cudaMemset(d_maxima_y,0,(halflen) * sizeof(T)));
	CUDA_SAFE_CALL( cudaMemset(d_minima_y,0,(halflen) * sizeof(T)));
	CUDA_SAFE_CALL( cudaMemset(d_diff,0,(len) * sizeof(T)));
	

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

	thrust::sort_by_key(thrust::device,thrust::device_ptr<int>(d_maxima),thrust::device_ptr<int>(d_maxima + counter[0]),thrust::device_ptr<T>(d_maxima_y));
	thrust::sort_by_key(thrust::device,thrust::device_ptr<int>(d_minima),thrust::device_ptr<int>(d_minima + counter[1]),thrust::device_ptr<T>(d_minima_y));



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





	cubic_spline_gpu<T,512>((const int *)d_maxima,d_maxima_y,counter[0],d_top,len,d_a,d_b,d_c,d_m,d_diff2);
	cudaThreadSynchronize();
	cubic_spline_gpu<T,512>((const int *)d_minima,d_minima_y,counter[1],d_bottom,len,d_a,d_b,d_c,d_m,d_diff2);
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
	// autofreeD(d_maxima);
	// autofreeD(d_maxima_y);
	// autofreeD(d_minima);
	// autofreeD(d_minima_y);
	// autofreeD(d_counter);
	// autofreeD(d_diff);
	// autofreeD(d_bottom);
	// autofreeD(d_top);
	// autofree(counter);
	
#ifdef DEBUG
	autofree(top);
	autofree(maxima);
	autofree(minima);
	autofree(diff);
	autofree(bottom);
#endif
	

}

#endif