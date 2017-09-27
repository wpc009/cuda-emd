#ifndef _COMMON_
#define _COMMON_

#define  autofree(buffer) do{ \
if(buffer != NULL){ CUDA_SAFE_CALL( cudaFree(buffer) ); } \
}while(0)

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
#endif