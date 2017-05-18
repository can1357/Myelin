#pragma once
#include "..\Network.cuh"
#include <curand.h>
#include <curand_kernel.h>

namespace Myelin
{
	namespace Layers
	{
		using namespace Myelin;
		using namespace Myelin::CUDA;
		using namespace Myelin::Activators;

		namespace Kernels
		{
			namespace Activation
			{
				// work = dim
				__global__ void Activate(
					const vsize_t dimensions,
					float const * input,
					float * output,
					int * hitmap,
					int seed, 
					float drop_percentage
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = dimensions.linearize();
					if ( id >= lout_size )
						return;

					curandState state;
					curand_init( seed + id, 0, 0, &state );

					int32_t skip = curand_uniform( &state ) <= drop_percentage;

					if ( skip )
					{
						hitmap[id] = 0;
						output[id] = 0.0f;
					}
					else
					{
						hitmap[id] = 1;
						output[id] = input[id];
					}
				}

				// work = dim
				__global__ void CalculatePrevLayerGradients(
					const vsize_t dimensions,
					float const * input,
					float const * grad_next_layer,
					float * grads_prev,
					int * hitmap
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = dimensions.linearize();
					if ( id >= lout_size )
						return;

					grads_prev[id] = hitmap[id] ? grad_next_layer[id] : 0.0f;
				}
			};
		};

		class Dropout : public ILayer
		{
		public:
			const vsize_t dim_size;
			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;

			shared_array_t<int> hitmap;
			float drop_percentage;
			bool enabled = true;

			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Dropout( dim_size, drop_percentage );
				if ( copy_state )
				{
					new_layer->enabled = this->enabled;
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
					new_layer->hitmap.copy_from( this->hitmap );
				}

				return new_layer;
			}

			Dropout(
				const vsize_t& dim_size,
				float drop_percentage
			) :
				drop_percentage( drop_percentage ),
				dim_size( dim_size ),
				grads_in( dim_size.linearize() ),
				input( dim_size.linearize() ),
				output( dim_size.linearize() ),
				hitmap( dim_size.linearize() )
			{
			
			}


			virtual void calc_grads( float const* grads_next )
			{
				using namespace Myelin::Layers::Kernels::Activation;

				CalculatePrevLayerGradients DWORK( dim_size.linearize(), cudaStreamDefault ) (
					dim_size,
					input.gpu,
					grads_next,
					grads_in.gpu,
					hitmap.gpu);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::Activation;
				
				Activate DWORK( dim_size.linearize(), cudaStreamDefault ) (
					dim_size,
					input.gpu,
					output.gpu,
					hitmap.gpu,
					rand(),
					enabled ? drop_percentage : 0.0f
				);
			}

			virtual void update_weights( float lr_mutlp, cudaStream_t str )
			{
				
			}

			virtual shared_array_t<float>& get_output()
			{
				return output;
			}

			virtual shared_array_t<float>& get_input()
			{
				return input;
			}

			virtual shared_array_t<float>& get_grads()
			{
				return grads_in;
			}

			virtual vsize_t get_input_size()
			{
				return dim_size;
			}

			virtual vsize_t get_output_size()
			{
				return dim_size;
			}

			virtual std::string get_name()
			{
				return "Dropout p=" + std::to_string( this->drop_percentage );
			}

			virtual std::string get_short_name()
			{
				return "dropout";
			}
			

			SimpleHash hash()
			{
				SimpleHash hash;
				hash.begin_hash();
				hash.add_item( dim_size );
				return hash;
			}

			virtual SaveHeader* save()
			{
				SaveBlob b;
				b.push( this->grads_in );
				b.push( this->input );
				b.push( this->output );
				b.push( this->hitmap );
				return b.to_savehdr( this->hash() );
			}

			virtual bool load( SaveHeader* data )
			{
				if ( !data )
					return false;

				if ( data->hash.hash == this->hash().hash )
				{
					SaveBlob b = SaveBlob::from_savehdr( data );

					b.pop( this->grads_in );
					b.pop( this->input );
					b.pop( this->output );
					b.pop( this->hitmap );
					return true;
				}
				return false;
			}
		};
	};
};