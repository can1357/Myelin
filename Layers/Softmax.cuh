#pragma once
#include "..\Network.cuh"

namespace Myelin
{
	namespace Layers
	{
		using namespace Myelin;
		using namespace Myelin::CUDA;
		using namespace Myelin::Activators;

		namespace Kernels
		{
			namespace Softmax
			{
				__device__ float f_max( float const * input, int count )
				{
					float flt = -FLT_MAX;
					for ( int i = 0; i < count; i++ )
					{
						float val = input[i];
						if ( val > flt )
							flt = val;
					}
					return flt;
				}

				// work = dim
				__global__ void Activate(
					const vsize_t dimensions,
					float const * input,
					float * output
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = dimensions.linearize();
					if ( id >= lout_size )
						return;

					float mx = f_max( input, lout_size );

					float nom = exp( input[id] - mx );
					float denom = 0.0f;

					for ( int i = 0; i < lout_size; i++ )
						denom += exp( input[i] - mx );

					output[id] = nom / denom;
				}

				// work = dim
				__global__ void CalculatePrevLayerGradients(
					const vsize_t dimensions,
					float const * input,
					float const * grad_next_layer,
					float * grads_prev,
					float const * output
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lin_size = dimensions.linearize();
					if ( id >= lin_size )
						return;

					float gradient = 0.0f;

					float out_cur = output[id];
					for ( int i = 0; i < lin_size; i++ )
					{
						if ( i == id )
							gradient += out_cur*(1 - out_cur) * grad_next_layer[i];
						else
						{
							float out = output[i];
							gradient += -out_cur*out * grad_next_layer[i];
						}
					}

					grads_prev[id] = gradient;
				}
			};
		};

		class Softmax : public ILayer
		{
		public:
			const vsize_t dim_size;
			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;

			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Softmax( dim_size );

				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}

			Softmax(
				const vsize_t& dim_size
			) :
				dim_size( dim_size ),
				grads_in( dim_size.linearize() ),
				input( dim_size.linearize() ),
				output( dim_size.linearize() )
			{
			
			}


			virtual void calc_grads( float const* grads_next )
			{
				using namespace Myelin::Layers::Kernels::Softmax;

				CalculatePrevLayerGradients DWORK( dim_size.linearize(), cudaStreamDefault ) (
					dim_size,
					input.gpu,
					grads_next,
					grads_in.gpu,
					output.gpu);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::Softmax;
				
				Activate DWORK( dim_size.linearize(), cudaStreamDefault ) (
					dim_size,
					input.gpu,
					output.gpu
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
				return "Softmax";
			}

			virtual std::string get_short_name()
			{
				return "softmax";
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
					return true;
				}
				return false;
			}
		};
	};
};