#pragma once
#include "cuda_simplifier.cuh"

namespace Myelin
{
	namespace ErrorFunction
	{
		enum class ErrorFunctionID
		{
			CrossEntrophy,
			MSE,
			Weighted_MSE,
			Identity
		};

		struct ErrorFunction
		{
			ErrorFunctionID id;
			size_t arg1 = 0;
			size_t arg2 = 0;
			ErrorFunction( ErrorFunctionID i ) :id( i )
			{

			}
			ErrorFunction( ErrorFunctionID i, size_t a ) :id( i ), arg1( a )
			{

			}
			ErrorFunction( ErrorFunctionID i, size_t a, size_t b ) :id( i ), arg1( a ), arg2( b )
			{

			}
		};

		ErrorFunction CreateWeightedMSE( float * weights_gpu )
		{
			return ErrorFunction( ErrorFunctionID::Weighted_MSE, (size_t) weights_gpu );
		}

		struct cost_counter_t
		{
			int count;
			float sum_cost;
			float last_cost;

			float sum_perc_error;
			float last_perc_error;

			float sum_count_wrong;
			int last_count_wrong;

			__host__ __device__ void add_cost( float x, float perc, int wrong )
			{
				count++;
				sum_cost += x;
				last_cost = x;

				sum_count_wrong += wrong;
				last_count_wrong = wrong;

				sum_perc_error += perc;
				last_perc_error = perc;
			}
			__host__ __device__ float get_instantaneous_cost()
			{
				return last_cost;
			}
			__host__ __device__ float get_average_cost()
			{
				return sum_cost / count;
			}
			__host__ __device__ float get_instantaneous_perc_error()
			{
				return last_perc_error;
			}
			__host__ __device__ float get_average_perc_error()
			{
				return sum_perc_error / count;
			}
			__host__ __device__ float get_instantaneous_misses()
			{
				return last_count_wrong;
			}
			__host__ __device__ float get_average_misses()
			{
				return sum_count_wrong / count;
			}
			__host__ __device__ void reset_counter()
			{
				sum_cost = get_average_cost();
				sum_perc_error = get_average_perc_error();
				sum_count_wrong = get_average_misses();

				count = 1;
			}
		};

		void CalculateCost( ErrorFunction error_function, float const * out, float const * expected, cost_counter_t * out_cost, int count )
		{
			if ( error_function.id == ErrorFunctionID::CrossEntrophy )
			{
				Myelin::CUDA::RunLambda( [out, expected, out_cost, count] __device__ ()
				{
					float out_new = 0;
					float dev = 0;
					int wrng = 0;
					for ( int i = 0; i < count; i++ )
					{
						float o_normalized = out[i];
						float e_normalized = expected[i];
						
						dev += abs( out[i] - expected[i] );

						if ( abs( round( out[i] ) - round( expected[i] ) ) > 0 )
							wrng++;

						if ( o_normalized > 0.9999999f )
							o_normalized = 0.9999999f;
						if ( o_normalized < 0.0000001f )
							o_normalized = 0.0000001f;

						if ( e_normalized > 0.9999999f )
							e_normalized = 0.9999999f;
						if ( e_normalized < 0.0000001f )
							e_normalized = 0.0000001f;

						out_new += e_normalized * log( o_normalized ) + (1 - e_normalized) * log( 1 - o_normalized );
					}
					out_new *= -1.0f / count;
					out_cost->add_cost( out_new, dev / count, wrng );
				} );
				return;
			}
			if ( error_function.id == ErrorFunctionID::MSE )
			{
				Myelin::CUDA::RunLambda( [out, expected, out_cost, count] __device__()
				{
					float out_new = 0;
					float dev = 0;
					int wrng = 0;
					for ( int i = 0; i < count; i++ )
					{
						dev += abs( out[i] - expected[i] );

						if ( abs( round( out[i] ) - round( expected[i] ) ) > 0 )
							wrng++;

						out_new += ( expected[i] - out[i] ) * ( expected[i] - out[i] );
					}
					out_new *= 1.0f / 2.0f;
					out_cost->add_cost( out_new, dev / count, wrng );
				} );
				return;
			}
			if ( error_function.id == ErrorFunctionID::Weighted_MSE )
			{
				float * weights = (float *)(error_function.arg1);
				Myelin::CUDA::RunLambda( [weights, out, expected, out_cost, count] __device__()
				{
					float out_new = 0;
					float dev = 0;
					int wrng = 0;
					for ( int i = 0; i < count; i++ )
					{
						dev += abs( out[i] - expected[i] );

						if ( abs( round( out[i] ) - round( expected[i] ) ) > 0 )
							wrng++;

						out_new += weights[i] * ( expected[i] - out[i] ) * ( expected[i] - out[i] );
					}
					out_new *= 1.0f / 2.0f;
					out_cost->add_cost( out_new, dev / count, wrng );
				} );
				return;
			}
			if ( error_function.id == ErrorFunctionID::Identity )
			{
				Myelin::CUDA::RunLambda( [out, expected, out_cost, count] __device__()
				{
					float out_new = 0;
					for ( int i = 0; i < count; i++ ) 
						out_new += abs(expected[i]);
					out_cost->add_cost( out_new, 0, 0 );
				} );
				return;
			}
		}

		__global__ void CalculateGradients( ErrorFunction error_function, float const * out, float const * expected, float * out_grads, int count )
		{
			const int id = threadIdx.x + blockIdx.x * blockDim.x;
			if ( id >= count )
				return;

			float outv = out[id];
			float expv = expected[id];

			float temp = 0.0f;

			switch ( error_function.id )
			{
				case ErrorFunctionID::CrossEntrophy:
					temp = (expv - outv) / (outv * (1 - outv) + 1e-10);

					if ( temp > 10e3 )
						temp = 10e3;
					else if ( temp < -10e3 )
						temp = -10e3;

					out_grads[id] = -1.0f / count * temp;
					return;
				case ErrorFunctionID::Identity:
					out_grads[id] = expv;
					return;
				case ErrorFunctionID::Weighted_MSE:
					out_grads[id] = ( (float*) ( error_function.arg1 ) )[id] * ( outv - expv );
					return;
				default:
				case ErrorFunctionID::MSE:
					out_grads[id] = ( outv - expv );
					return;
			};
		}
	};
};
