#ifndef _EMD_
#define _EMD_


template <typename T> 
__global__ void _extrema(const T* data,int len,int* maxima,int* minima,int* counter)
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
		if (i+1 <len){
			temp = data[i+1] * data[i];
			if (temp < 0){
				if (data[i] > 0){
					//maxima
					pos = atomicAdd(&l_n[0],1);
					isMaxima = 1;
				} else if( data[i] < 0){
					//minima
					pos = atomicAdd(&l_n[1],1);
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

		if ( i + 1 < len && isMaxima > 0){
			if (isMaxima == 1){
				maxima[l_n[0] + pos] = i + 1;
			}else if(isMaxima == 2){
				minima[l_n[1] + pos] = i + 1;
			}
		}
		__syncthreads();
	}
}

template <typename T>
void sifting(const T * data,const int len)
{

}

#endif