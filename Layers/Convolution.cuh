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
			namespace Convolution
			{
				// work = out
				__global__ void Activate(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float * output,
					float const * filter_data,
					int extend,
					int stride,
					float multp
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = out_size.linearize();
					if ( id >= lout_size )
						return;

					point_t out_id = point_t( id, out_size );
					float const * filter_base = filter_data + out_id.z * (extend*extend*in_size.z);
					point_t input_start = point_t( out_id.x * stride, out_id.y * stride, 0 );
					int xy = in_size.x * in_size.y;

					float sum = 0.0f;

					int lfilter_id = 0;
					int linput_idz = input_start.linearize( in_size );

					for ( int z = 0; z < in_size.z; z++ )
					{
						int linput_idy = linput_idz;

						for ( int y = 0; y < extend; y++ )
						{
							for ( int x = 0; x < extend; x++ )
							{
								int linput_idx = linput_idy + x;
								sum += filter_base[lfilter_id] * input[linput_idx];
								lfilter_id++;
							}
							linput_idy += in_size.x;
						}

						linput_idz += xy;
					}

					output[id] = sum * multp;
				}

				__device__ int MinNorm( float f )
				{
					if ( f <= 0 )
						return 0;
					return ceilf( f );
				}
				__device__ int MaxNorm( float f, int mx )
				{
					if ( f < mx )
						return f;
					return mx - 1;
				}

				// work = n(weights)
				__global__ void UpdateWeights(
					int lmax,
					float * filter_data,
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
					UpdateParameter( grad, filter_data[id], o_id, args, lr_multp );
				}

				// work = n(weights)
				__global__ void CalculateWeightGradients(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float const * grad_next_layer,
					GradHeader* grads,
					OptimizerID o_id,
					size_t grad_size,
					int extend,
					int stride,
					float multp
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					int ef2 = extend * extend;
					int filter_size = ef2 * in_size.z;
					int max_work = filter_size * out_size.z;
					if ( id >= max_work )
						return;

					int weight_id = id % filter_size;
					int filter_id = (id - weight_id) / filter_size;

					int w_xy = weight_id % ef2;
					int w_z = (weight_id - w_xy) / ef2;

					int w_x = w_xy % extend;
					int w_y = (w_xy - w_x) / extend;

					GradHeader* grad = (GradHeader*)((size_t)grads + id*grad_size);

					const float * in_base = input + in_size.x * in_size.y * w_z;
					const float * out_base = grad_next_layer + out_size.x * out_size.y * filter_id;

					float sum = 0.0f;

					for ( int y = 0; y < out_size.y; y++ )
					{
						for ( int x = 0; x < out_size.x; x++ )
						{
							int inx = x * stride + w_x;
							int iny = y * stride + w_y;

							sum += in_base[inx + iny*in_size.x] *
								out_base[x + y*out_size.x];
						}
					}

					grad->gradient += sum * multp;
				}

				// work = in 
				__global__ void CalculatePrevLayerGradients(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * filter_data,
					float const * grad_next_layer,
					float * grads_in,
					int extend,
					int stride,
					float multp
				)
				{
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lin_size = in_size.linearize();

					if ( id >= lin_size )
						return;

					point_t pin = point_t( id, in_size );

					int minx = MinNorm( float( pin.x - extend + 1 ) / stride );
					int miny = MinNorm( float( pin.y - extend + 1 ) / stride );
					int maxx = MaxNorm( float( pin.x / stride ), out_size.x );
					int maxy = MaxNorm( float( pin.y / stride ), out_size.y );


					const int lfilter_size = extend * extend * in_size.z;
					const int z_add = (extend * extend) * pin.z;
					const float * base_filterz = filter_data + z_add;

					int o_xy = out_size.x * out_size.y;
					int o_x = out_size.x;

					float sum_error = 0;
					for ( int f = 0; f < out_size.z; f++ )
					{
						const int k_add = lfilter_size*f;
						const float * base_filterk = base_filterz + k_add;
						for ( int j = miny; j <= maxy; j++ )
						{
							int iminy = j * stride;
							const int y_add = (pin.y - iminy)*extend;
							const float * base_filtery = base_filterk + y_add;
							for ( int i = minx; i <= maxx; i++ )
							{
								int iminx = i * stride;

								const int x_add = (pin.x - iminx);

								float w_applied = base_filtery[x_add];
								float g_next = grad_next_layer[i + j*o_x + f*o_xy];
								sum_error += w_applied * g_next;
							}
						}
					}
					grads_in[id] = sum_error * multp;
				}
			};
		};

		class Convolution : public ILayer
		{
		public:
			const vsize_t in_size;
			const vsize_t out_size;

			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;
			shared_array_t<float> filters;

			shared_memory_t gradients;

			function<OptimizerArgs()> get_args;
			OptimizerID optimizer_id;
			size_t grad_size;

			float multiplier = 1.0f;

			int extend, stride, num_filters;


			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Convolution( in_size, extend, stride, num_filters, optimizer_id, get_args, []( int x )
				{ 
					return 0.0f;
				}, false);

				new_layer->multiplier = this->multiplier;
				new_layer->gradients.copy_from( this->gradients );
				new_layer->filters.copy_from( this->filters );

				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}

			Convolution(
				const vsize_t& in_size,
				int extend,
				int stride,
				int num_filters,
				OptimizerID optimizer_id,
				function<OptimizerArgs()> get_args,
				std::function<float( int )> weight_initializer,
				bool average_input
			) :
				optimizer_id( optimizer_id ),
				grad_size( GetGradientSize( optimizer_id ) ),
				grads_in( in_size.linearize() ),
				input( in_size.linearize() ),
				filters( extend * extend * in_size.z * num_filters ),
				gradients( extend * extend * in_size.z * num_filters * GetGradientSize( optimizer_id ) ),
				in_size( in_size ),
				out_size(
					(in_size.x - extend) / stride + 1,
					(in_size.y - extend) / stride + 1,
					num_filters
				),
				output(
					((in_size.x - extend) / stride + 1)*
					((in_size.y - extend) / stride + 1)*
					(num_filters)
				),
				get_args( get_args ),
				multiplier( average_input ? 1.0f / (extend*extend*in_size.z) : 1.0f ),
				extend( extend ),
				stride( stride ),
				num_filters( num_filters )
			{
				assert( (float( in_size.x - extend ) / stride + 1)
						==
						((in_size.x - extend) / stride + 1) );

				assert( (float( in_size.y - extend ) / stride + 1)
						==
						((in_size.y - extend) / stride + 1) );

				for ( int i = 0; i < extend * extend * in_size.z * num_filters; i++ )
					filters.cpu[i] = weight_initializer( i );
				filters.sync_gpu();
			}


			virtual void calc_grads( float const* grads_next )
			{
				using namespace Myelin::Layers::Kernels::Convolution;

				int n_weights = extend * extend * in_size.z * num_filters;
				CalculateWeightGradients DWORK(n_weights, cudaStreamDefault) (
					in_size,
					out_size,
					input.gpu,
					grads_next,
					(GradHeader*)gradients.gpu,
					optimizer_id,
					grad_size,
					extend,
					stride,
					multiplier
				);

				CalculatePrevLayerGradients DWORK(in_size.linearize(), cudaStreamDefault) (
					in_size,
					out_size,
					filters.gpu,
					grads_next,
					grads_in.gpu,
					extend,
					stride,
					multiplier
				);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::Convolution;
				
				Activate DWORK( out_size.linearize(), cudaStreamDefault ) (
					in_size,
					out_size,
					input.gpu,
					output.gpu,
					filters.gpu,
					extend,
					stride,
					multiplier
				);
			}

			virtual void update_weights( float lr_mutlp, cudaStream_t str )
			{
				using namespace Myelin::Layers::Kernels::Convolution;

				int n_weights = extend * extend * in_size.z * num_filters;
				UpdateWeights DWORK( n_weights, str ) (
					n_weights,
					filters.gpu,
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
				return "Convolution " + in_size.to_string() + " -> " + out_size.to_string() +
					" (" + std::to_string( extend ) + "x" + std::to_string( extend ) + ")";
			}

			virtual std::string get_short_name()
			{
				return "conv";
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
				b.push( this->filters );
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
					b.pop( this->filters );
					b.pop( this->gradients );

					return true;
				}
				return false;
			}
		};
	};
};