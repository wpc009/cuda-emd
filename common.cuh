#ifndef _COMMON_
#define _COMMON_

// #define DEBUG

#define  autofreeD(buffer) do{ \
if(buffer != NULL){ CUDA_SAFE_CALL( cudaFree(buffer) ); } \
}while(0)

#define autofree(buffer) do{ \
if(buffer != NULL){ free(buffer);} \
}while(0)

#define printArray(array,len) printArrayFmt(array,len,%f)

#define automalloc(buffer,size,type) buffer = (type*) malloc(size* sizeof(type))

#define automallocD(buffer,size,type) CUDA_SAFE_CALL( cudaMalloc( (void**)&buffer,size * sizeof(type)) )

#define printArrayFmt(array,len,fmt) printArrayFmtAlias(array,array,len,fmt)

#define printArrayFmtAlias(array,alias,len,fmt) printf(#alias"=["); \
    for(int i =0;i<len;i++){ \
        if(i % 10 == 0){ \
            printf("\n");   \
        }   \
        printf(#fmt", ",array[i]); \
    } \
    printf("]\n")
    
#define println(fmt,...) printf("[%d]"fmt,__LINE__,##__VA_ARGS__);printf("\n")

#define blockalign(x,block_size) (x + (block_size) -1 )/ (block_size)

#define PI 3.1415926



template <typename T>
__global__ void _linear_extrapolation(T* d_data,int len,int * maxima,T* maxima_y,int maxima_len,int * minima,T* minima_y,int minima_len)
{
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;
	T temp = T();
	int idx_l = 0;
	int idx_r = 0;
	int idx_t = 0;
	if ( ti == 0) {
		maxima_y[0] = d_data[0];
		maxima_y[maxima_len - 1] = d_data[len - 1];
		minima_y[0] = d_data[0];
		minima_y[minima_len - 1] = d_data[len - 1];

	}
	__syncthreads();
	

	if(maxima_len > 4){	

		if(ti == 0){
			idx_l = 1;
			idx_r = 2;
			idx_t = 0;

		}else if( ti == 1){
			idx_l = maxima_len - 3;
			idx_r = maxima_len - 2;
			idx_t = maxima_len - 1;
			
			// T x1 = maxima[maxima_len -3];
			// T x2 = maxima[maxima_len -2];
			// T y1 = maxima_y[maxima_len -3];
			// T y2 = maxima_y[maxima_len -2];
			// temp = y1 + (y2 - y1) * (len -1 - x1) / (x2 - x1);
			// maxima_y[maxima_len-1] = max(temp,d_data[len - 1]);
		}
		T x1 = (T)maxima[idx_l];
		T x2 = (T)maxima[idx_r];
		T y1 = maxima_y[idx_l];
		T y2 = maxima_y[idx_r];
		temp = y1 + (y2 - y1) * ( maxima[idx_t]- x1) / (x2 - x1);
		maxima_y[idx_t] = max(temp,maxima_y[idx_t]);
	}

	if(minima_len > 4)
	{
		if( ti == 0)
		{
			idx_l = 1;
			idx_r = 2;
			idx_t = 0;
			// T x1 = minima[1];
			// T x2 = minima[2];
			// T y1 = d_data[x1];
			// T y2 = d_data[x2];
			// temp = y1 + (y2 - y1) * (0 - x1) / (x2 - x1);
			// minima_y[0] = min(temp,d_data[0]);

		}else if( ti == 1)
		{
			idx_l = minima_len - 3;
			idx_r = minima_len - 2;
			idx_t = minima_len - 1;
			// T x1 = minima[minima_len -3];
			// T x2 = minima[minima_len -2];
			// T y1 = d_data[x1];
			// T y2 = d_data[x2];
			// temp = y1 + (y2 - y1) * (len -1 - x1) / (x2 - x1);
			// minima_y[minima_len - 1] = min(temp,d_data[len - 1]);

		}
		T x1 = (T) minima[idx_l];
		T x2 = (T) minima[idx_r];
		T y1 = minima_y[idx_l];
		T y2 = minima_y[idx_r];
		temp = y1 + (y2 - y1) * (minima[idx_t] - x1) / (x2 - x1);
		minima_y[idx_t] = min(temp, minima_y[idx_t]);
	}
}

template <typename T> 
__global__ void _extrema(const T* data,const T* diff,int len,int* maxima,T* maxima_y,int* minima,T* minima_y,int* counter)
{
	__shared__ int  l_n[2];
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;
	int pos = 0;
	int isMaxima =0;
	T temp = 0;

	for(int i= ti + bw * bi; i < len; i += bw*gridDim.x)
	{
		
		if ( ti == 0){
			l_n[0] = 0;
			l_n[1] = 0;
		}

		__syncthreads();
		pos = 0;
		isMaxima = -1;

		// if (i+1 <len){
		// 	temp = diff[i+1] * diff[i];
		// 	if (temp < 0){
		// 		if (diff[i] > 0){
		// 			//maxima
		// 			pos = atomicAdd(&l_n[0],1);
		// 			isMaxima = 1;
		// 		} else if( diff[i] < 0){
		// 			//minima
		// 			pos = atomicAdd(&l_n[1],1);
		// 			isMaxima = 2;
		// 		}
		// 	}
		// }
		if ( i > 0) {
			temp = diff[i] * diff[i - 1];
			if (temp < 0)
			{
				if ( diff[i] < 0)
				{
					//maxima
					pos = atomicAdd(&l_n[0], 1);
					isMaxima = 1;
				} else
				{
					//minima
					pos = atomicAdd(&l_n[1], 1);
					isMaxima = 2;

				}

			}
		}
		__syncthreads();
		if (ti == 0){
			l_n[0] = atomicAdd(&counter[0],l_n[0]);
			l_n[1] = atomicAdd(&counter[1],l_n[1]);
		}
		__syncthreads();


		if (isMaxima == 1) {
			maxima[l_n[0] + pos] = (i - 1);
			maxima_y[l_n[0] + pos] = data[i -1];
		} else if (isMaxima == 2) {
			minima[l_n[1] + pos] = (i - 1);
			minima_y[l_n[1] + pos] = data[i -1];
		}

		__syncthreads();
	}
}



template <typename T>
__global__ void _diff(const T* a,T* diff,int len){
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;

	for ( int i= ti + bw * bi; i< len; i+= bw * gridDim.x)
	{
		if( i == 0 ){
			diff[i] = 0;
		}else if (i > 0){
			diff[i] = a[i] - a[i-1];
		}
	}
}

template <typename T>
__global__ void _mean(T* a,T* b,int len){
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;

	for( int i =ti + bw * bi; i < len;i += bw * gridDim.x)
	{
		a[i] = (a[i] + b[i]) * 0.5;
	}
}

template <typename T>
__global__ void _residual(const T* a,const T* b,T* res,int len){
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;

	for( int i = ti+ bw*bi;i < len; i+= bw*gridDim.x)
	{
		res[i] = a[i] - b[i];
	}
}

template <typename T,unsigned int blockSize>
__global__ void _reduce_sum(T* g_idata,T* g_odata,unsigned int n)
{
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid; 
	if( i >= n)
		return;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for(unsigned int s = 1;s < blockDim.x; s*=2)
	{
		if( tid % (2*s) == 0 && tid + s < n)
		{
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if( tid ==0) g_odata[blockIdx.x] = sdata[0];

}

template <typename T,unsigned int blockSize>
__global__ void _reduce6_sum(T *g_idata,T *g_odata, unsigned int n) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid; 
	unsigned int gridSize = blockSize*2*gridDim.x; sdata[tid] = 0;

	while (i < n) { 
		sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
		i += gridSize; 
	}
	 __syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); } 
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); } 
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) { 
		if (blockSize >= 64) sdata[tid] += sdata[tid+32];
		if (blockSize >= 32) sdata[tid] += sdata[tid+16];
		if (blockSize >= 16) sdata[tid] += sdata[tid+8];
		if (blockSize >= 8) sdata[tid] += sdata[tid+4];
		if (blockSize >= 4) sdata[tid] += sdata[tid+2];
		if (blockSize >= 2) sdata[tid] += sdata[tid+1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

template <typename T,unsigned int gridSize,unsigned int blockSize>
T sum(T* d_data,unsigned int len)
{
	T* d_temp = NULL;
	T* temp =NULL;
	T res = T();
	int pageSize = gridSize * blockSize;
	int remains = 0;
	int numOfPages = blockalign(len,pageSize);
#ifdef DEBUG	
	println("num of segments %d",numOfPages);
#endif
	automalloc(temp,gridSize,T);
	automallocD(d_temp,gridSize,T);
	for (int j = 0; j < numOfPages; j++)
	{
		CUDA_SAFE_CALL( cudaMemset(d_temp, 0, gridSize * sizeof(T)) );
		_reduce_sum<T, blockSize> <<< gridSize, blockSize, blockSize * sizeof(T) >>>(&d_data[j * pageSize], d_temp, len);
		cudaThreadSynchronize();

		CUDA_SAFE_CALL( cudaMemcpy(temp, d_temp, gridSize * sizeof(T), cudaMemcpyDeviceToHost) );
		
#ifdef DEBUG
		printArray(temp,gridSize);
		println("res=%f,remains=%d",res,remains);
#endif
		for (int i = 0; i < gridSize; i++) {
			res += temp[i];
		}
	}

	autofreeD(d_temp);
	autofree(temp);
	return res;
}

template <typename T>
__global__ void _sd(T* h,T* last_h,T* sd,unsigned int len)
{
	int bi = blockIdx.x;
	int bw = blockDim.x;
	int ti = threadIdx.x;
	T temp;

	for( int i = ti+ bw*bi;i < len; i+= bw*gridDim.x)
	{
		sd[i] = (h[i] - last_h[i]) * (h[i] - last_h[i]) / (last_h[i]*last_h[i] + 1e-10f);
	}
}

#endif