#pragma once
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <iostream>
#include <math.h>
#include <vector>

namespace Myelin
{
	namespace CUDA
	{
		#define CHK_CUERROR(ans) { Myelin::CUDA::AssertCuda((ans), __FILE__, __LINE__); }

		#define CSIZE 512

		#ifdef __CUDACC__
		#define DWORK(work, stream) <<<ceilf( float( work ) / CSIZE ), min( CSIZE, work ), 0, stream >>>
		#else
		#define DWORK(a, b)
		#endif

		#ifdef _DEBUG
		#define CHECK_CUDA_ERRORS
		#endif

		inline static void AssertCuda( cudaError_t code, const char *file, int line )
		{
			#ifdef CHECK_CUDA_ERRORS
			{
				if ( code != cudaSuccess )
				{
					char buffer[1024];
					sprintf( buffer, "GPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
					fprintf( stderr, buffer );
					throw buffer;
				}
			}
			#endif
		}

		template<typename lambda>
		__global__ static void LambdaKernel( lambda fn )
		{
			fn();
		}

		template<typename lambda>
		inline static void RunLambda( lambda l )
		{
			LambdaKernel << <1, 1 >> >(l);
		}

		inline static void SetDevice( int device )
		{
			CHK_CUERROR( cudaSetDevice( device ) );
			CHK_CUERROR( cudaDeviceReset() );
		}

		inline static void Initialize( int device )
		{
			SetDevice( device );
			cudaFree( 0 );
			cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
		}


		template<typename T>
		struct shared_array_t
		{
			T * cpu;
			T * gpu;
			size_t num_items;

			inline __device__ __host__ size_t get_num_bytes()
			{
				return num_items * sizeof( T );
			}

			shared_array_t()
			{
				cpu = nullptr;
				gpu = nullptr;
				num_items = 0;
			}

			shared_array_t( size_t count )
			{
				num_items = count;
				CHK_CUERROR( cudaHostAlloc( &cpu, get_num_bytes(), 0 ) );
				CHK_CUERROR( cudaMalloc( &gpu, get_num_bytes() ) );
			}

			shared_array_t( const shared_array_t<T>& other )
			{
				if ( !other.cpu )
				{
					cpu = nullptr;
					gpu = nullptr;
					num_items = 0;
				}
				else
				{
					num_items = other.num_items;

					CHK_CUERROR( cudaHostAlloc( &cpu, get_num_bytes(), 0 ) );
					CHK_CUERROR( cudaMalloc( &gpu, get_num_bytes() ) );

					CHK_CUERROR( cudaMemcpy( gpu, other.gpu, get_num_bytes(), cudaMemcpyDeviceToDevice ) );
					memcpy( cpu, other.cpu, get_num_bytes() );
				}
			}

			void copy_from( shared_array_t<T>& other )
			{
				assert( this->num_items == other.num_items );

				CHK_CUERROR( cudaMemcpy( gpu, other.gpu, get_num_bytes(), cudaMemcpyDeviceToDevice ) );
				memcpy( cpu, other.cpu, get_num_bytes() );
			}

			void copy_from( const std::vector<T>& other )
			{
				assert( this->num_items == other.size() );
				memcpy( cpu, other.data(), get_num_bytes() );
				this->sync_gpu();
			}

			void sync_gpu()
			{
				CHK_CUERROR( cudaMemcpy( gpu, cpu, get_num_bytes(), cudaMemcpyHostToDevice ) );
			}

			void sync_cpu()
			{
				CHK_CUERROR( cudaMemcpy( cpu, gpu, get_num_bytes(), cudaMemcpyDeviceToHost ) );
			}


			~shared_array_t()
			{
				if ( cpu )
					CHK_CUERROR( cudaFreeHost( cpu ) );
				if ( gpu )
					CHK_CUERROR( cudaFree( gpu ) );
			}
		};

		using shared_memory_t = shared_array_t<unsigned char>;
	};
};