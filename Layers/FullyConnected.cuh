#pragma once
#include "..\Network.cuh"

namespace Myelin
{
	namespace Layers
	{
		using namespace Myelin;
		using namespace Myelin::CUDA;
		using namespace Myelin::Optimizers;

		namespace Kernels
		{
			namespace FullyConnected
			{
				// work = out
				__global__ void Activate(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float * output,
					float const * weights,
					float multp
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = out_size.linearize();
					if ( id >= lout_size )
						return;

					const int lin_size = in_size.linearize();
					float const * weights_base = weights + lin_size * id;

					// calculate sum [Sum[i=0 -> i=n]  W(a->i)x(i)]
					float sum = 0.0f;
					for ( int i = 0; i < lin_size; i++ )
						sum += weights_base[i] * input[i];

					// out = result * multp
					output[id] = sum * multp;
				}

				// work = n(weights)
				__global__ void UpdateWeights(
					int lmax,
					float * weights,
					GradHeader* grads,
					OptimizerID o_id,
					OptimizerArgs args,
					float lr_multp,
					size_t grad_size
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					if ( id >= lmax )
						return;
					GradHeader* grad = (GradHeader*)((size_t)grads + id*grad_size);
					UpdateParameter( grad, weights[id], o_id, args, lr_multp );
				}

				// work = in * out
				__global__ void CalculateWeightGradients(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float const * grad_next_layer,
					GradHeader* grads,
					OptimizerID o_id,
					size_t grad_size,
					float multp
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = out_size.linearize();
					const int lin_size = in_size.linearize();
					const int lmax = lout_size  * lin_size;

					if ( id >= lmax )
						return;

					GradHeader* grad = (GradHeader*)((size_t)grads + id*grad_size);

					// id = in_id + (out_id) * in_size
					int in_id = id % lin_size;
					int out_id = (id - in_id) / lin_size;

					float inputv = input[in_id];
					float outputgv = grad_next_layer[out_id];
					
					grad->gradient += inputv * outputgv * multp;
				}

				// work = in
				__global__ void CalculatePrevLayerGradients(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * weights,
					float const * grad_next_layer,
					float * grads_prev,
					float multp
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lin_size = in_size.linearize();
					if ( id >= lin_size )
						return;

					const int lout_size = out_size.linearize();

					float sum = 0.0f;
					for ( int i = 0; i < lout_size; i++ )
						sum += grad_next_layer[i] * weights[lin_size * i + id];

					grads_prev[id] = sum * multp;
				}
			};
		};

		class FullyConnected : public ILayer
		{
		public:
			const vsize_t in_size;
			const vsize_t out_size;

			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;

			
			shared_array_t<float> weights;

			shared_memory_t gradients;

			function<OptimizerArgs()> get_args;
			OptimizerID optimizer_id;
			size_t grad_size;

			float multiplier = 1.0f;


			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new FullyConnected( in_size, out_size, optimizer_id, get_args, []( int x )
				{
					return 0.0f;
				}, false );

				new_layer->multiplier = this->multiplier;
				new_layer->gradients.copy_from( this->gradients );
				new_layer->weights.copy_from( this->weights );

				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}

			FullyConnected(
				const vsize_t& in_size,
				const vsize_t& out_size,
				OptimizerID optimizer_id,
				function<OptimizerArgs()> get_args,
				std::function<float( int )> weight_initializer,
				bool average_input
			) :
				optimizer_id( optimizer_id ),
				grad_size( GetGradientSize( optimizer_id ) ),
				grads_in( in_size.linearize() ),
				input( in_size.linearize() ),
				output( out_size.linearize() ),
				weights( in_size.linearize()*out_size.linearize() ),
				gradients( in_size.linearize() * out_size.linearize() * GetGradientSize( optimizer_id ) ),
				in_size( in_size ),
				out_size( out_size ),
				get_args( get_args ),
				multiplier( average_input ? 1.0f / in_size.linearize() : 1.0f )
			{
				for ( int i = 0; i < in_size.linearize()*out_size.linearize(); i++ )
					weights.cpu[i] = weight_initializer( i );
				weights.sync_gpu();
			}


			virtual void calc_grads( float const* grads_next )
			{
				using namespace Myelin::Layers::Kernels::FullyConnected;

				CalculateWeightGradients DWORK(in_size.linearize()*out_size.linearize(), cudaStreamDefault ) ( 
					in_size,
					out_size, 
					input.gpu, 
					grads_next, 
					(GradHeader*)gradients.gpu, 
					optimizer_id, 
					grad_size,
					multiplier);


				CalculatePrevLayerGradients DWORK( in_size.linearize(), cudaStreamDefault ) (
					in_size,
					out_size,
					weights.gpu,
					grads_next,
					grads_in.gpu,
					multiplier);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::FullyConnected;
				
				Activate DWORK( out_size.linearize(), cudaStreamDefault ) (
					in_size,
					out_size,
					input.gpu,
					output.gpu,
					weights.gpu,
					multiplier
				);
			}

			virtual void update_weights( float lr_mutlp, cudaStream_t str )
			{
				using namespace Myelin::Layers::Kernels::FullyConnected;

				UpdateWeights DWORK( in_size.linearize()*out_size.linearize(), str )  (
					in_size.linearize()*out_size.linearize(),
					weights.gpu,
					(GradHeader*)gradients.gpu,
					optimizer_id,
					get_args(),
					lr_mutlp,
					grad_size
				);

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
				return in_size;
			}

			virtual vsize_t get_output_size()
			{
				return out_size;
			}

			virtual std::string get_name()
			{
				return "Fully Connected " + in_size.to_string() + " -> " + out_size.to_string();
			}

			virtual std::string get_short_name()
			{
				return "fc";
			}
			
			SimpleHash hash()
			{
				SimpleHash hash;
				hash.begin_hash();
				hash.add_item( out_size );
				hash.add_item( in_size );
				hash.add_item( optimizer_id );
				return hash;
			}

			virtual SaveHeader* save()
			{
				SaveBlob b;
				b.push( this->grads_in );
				b.push( this->input );
				b.push( this->output );
				b.push( this->weights );
				b.push( this->gradients );

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
					b.pop( this->weights );
					b.pop( this->gradients );

					return true;
				}
				return false;
			}
		};
	};
};