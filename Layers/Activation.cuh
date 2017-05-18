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
			namespace Activation
			{
				// work = dim
				__global__ void Activate(
					const vsize_t dimensions,
					float const * input,
					float * output,
					ActivatorIdentifier act_id
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = dimensions.linearize();
					if ( id >= lout_size )
						return;

					output[id] = Activate( act_id, input[id] );
				}

				// work = dim
				__global__ void CalculatePrevLayerGradients(
					const vsize_t dimensions,
					float const * input,
					float const * grad_next_layer,
					float * grads_prev,
					ActivatorIdentifier act_id
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = dimensions.linearize();
					if ( id >= lout_size )
						return;

					grads_prev[id] = Derive( act_id, input[id] ) * grad_next_layer[id];
				}
			};
		};

		class Activation : public ILayer
		{
		public:
			const vsize_t dim_size;
			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;

			ActivatorIdentifier activator_id;

			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Activation( dim_size, activator_id );
				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}

			Activation(
				const vsize_t& dim_size,
				ActivatorIdentifier activator_id
			) :
				activator_id( activator_id ),
				dim_size( dim_size ),
				grads_in( dim_size.linearize() ),
				input( dim_size.linearize() ),
				output( dim_size.linearize() )
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
					activator_id);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::Activation;
				
				Activate DWORK( dim_size.linearize(), cudaStreamDefault ) (
					dim_size,
					input.gpu,
					output.gpu,
					activator_id
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
				return "Activator " + GetActivatorName( activator_id );
			}

			virtual std::string get_short_name()
			{
				return "activator";
			}

			SimpleHash hash()
			{
				SimpleHash hash;
				hash.begin_hash();
				hash.add_item( activator_id );
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