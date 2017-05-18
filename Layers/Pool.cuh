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
			namespace Pool
			{
				// work = out
				__global__ void Activate(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float * output,
					int extend,
					int stride
				){
					const int id = threadIdx.x + blockIdx.x * blockDim.x;
					const int lout_size = out_size.linearize();
					if ( id >= lout_size )
						return;

					point_t work = point_t( id, out_size );

					point_t input_start = point_t( work.x * stride, work.y * stride, work.z );

					int xy = in_size.x * in_size.y;

					float val = -FLT_MAX;
					int fid = 0;

					int iid_y = input_start.linearize( in_size );
					for ( int y = 0; y < extend; y++ )
					{
						for ( int x = 0; x < extend; x++ )
						{
							int iid_x = iid_y + x;

							float in = input[iid_x];
							val = val > in ? val : in;
							fid++;
						}
						iid_y += in_size.x;
					}

					output[id] = val;
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

				// work = in 
				__global__ void CalculatePrevLayerGradients(
					const vsize_t in_size,
					const vsize_t out_size,
					float const * input,
					float const * output,
					float const * grad_next_layer,
					float * grads_in,
					int extend,
					int stride
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

					int o_xy = out_size.x * out_size.y;
					int o_x = out_size.x;

					float sum_error = 0;

					for ( int j = miny; j <= maxy; j++ )
					{
						for ( int i = minx; i <= maxx; i++ )
						{
							float g_next = grad_next_layer[i + j*o_x + pin.z*o_xy];
							float in = input[id];
							float out = output[i + j*o_x + pin.z*o_xy];
							int is_max = in == out ? 1 : 0;
							sum_error += is_max * g_next;
						}
					}

					grads_in[id] = sum_error;
				}
			};
		};

		class Pool : public ILayer
		{
		public:
			const vsize_t in_size;
			const vsize_t out_size;

			shared_array_t<float> grads_in;

			shared_array_t<float> input;
			shared_array_t<float> output;

			int extend, stride;

			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Pool( in_size, extend, stride );

				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}

			Pool(
				const vsize_t& in_size,
				int extend,
				int stride
			) :
				grads_in( in_size.linearize() ),
				input( in_size.linearize() ),
				in_size( in_size ),
				out_size(
					(in_size.x - extend) / stride + 1,
					(in_size.y - extend) / stride + 1,
					in_size.z
				),
				output(
					((in_size.x - extend) / stride + 1)*
					((in_size.y - extend) / stride + 1)*
					(in_size.z)
				),
				extend( extend ),
				stride( stride )
			{
				assert( (float( in_size.x - extend ) / stride + 1)
						==
						((in_size.x - extend) / stride + 1) );

				assert( (float( in_size.y - extend ) / stride + 1)
						==
						((in_size.y - extend) / stride + 1) );
			}


			virtual void calc_grads( float const* grads_next )
			{
				using namespace Myelin::Layers::Kernels::Pool;
				
				CalculatePrevLayerGradients DWORK(in_size.linearize(), cudaStreamDefault) (
					in_size,
					out_size,
					input.gpu,
					output.gpu,
					grads_next,
					grads_in.gpu,
					extend,
					stride
				);
			}

			virtual void activate()
			{
				using namespace Myelin::Layers::Kernels::Pool;
				
				Activate DWORK( out_size.linearize(), cudaStreamDefault ) (
					in_size,
					out_size,
					input.gpu,
					output.gpu,
					extend,
					stride
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
				return in_size;
			}

			virtual vsize_t get_output_size()
			{
				return out_size;
			}

			virtual std::string get_name()
			{
				return "Pool " + in_size.to_string() + " -> " + out_size.to_string() +
					" (" + std::to_string( extend ) + "x" + std::to_string( extend ) + ")";
			}

			virtual std::string get_short_name()
			{
				return "pool";
			}
			


			SimpleHash hash()
			{
				SimpleHash hash;
				hash.begin_hash();
				hash.add_item( in_size );
				hash.add_item( out_size );
				hash.add_item( extend );
				hash.add_item( stride );
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