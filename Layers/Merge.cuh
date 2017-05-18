#pragma once
#include "..\Network.cuh"

namespace Myelin
{
	namespace Layers
	{
		using namespace Myelin;
		using namespace Myelin::CUDA;
		using namespace Myelin::Activators;

		class Merge : public ILayer
		{
		public:
			const vsize_t in_size, extra_info;
			const vsize_t out_size;

			shared_array_t<float> grads_in;
			shared_array_t<float> input;
			shared_array_t<float> output;


			virtual ILayer* copy( bool copy_state )
			{
				auto new_layer = new Merge( in_size, extra_info );

				if ( copy_state )
				{
					new_layer->input.copy_from( this->input );
					new_layer->output.copy_from( this->output );
					new_layer->grads_in.copy_from( this->grads_in );
				}

				return new_layer;
			}


			Merge(
				const vsize_t& in_size,
				const vsize_t& extra_info
			) :
				in_size( in_size ),
				extra_info( extra_info ),
				grads_in( in_size.linearize() + extra_info.linearize() ),
				input( in_size.linearize() ),
				output( in_size.linearize() + extra_info.linearize() ),
				out_size( in_size.linearize() + extra_info.linearize(), 1, 1 )
			{
			
			}


			virtual void calc_grads( float const* grads_next )
			{
				CHK_CUERROR( cudaMemcpy( grads_in.gpu, grads_next, grads_in.get_num_bytes(), cudaMemcpyDeviceToDevice ) );
			}

			virtual void activate()
			{
				CHK_CUERROR( cudaMemcpy( output.gpu, input.gpu, input.get_num_bytes(), cudaMemcpyDeviceToDevice ) );
			}

			void add_extra( float * extra )
			{
				CHK_CUERROR( cudaMemcpy( output.gpu + in_size.linearize(), extra, extra_info.linearize()*sizeof(float), cudaMemcpyDeviceToDevice ) );
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
				return "Merge " + in_size.to_string() + " + "+ extra_info.to_string() +" -> " + out_size.to_string();
			}

			virtual std::string get_short_name()
			{
				return "merge";
			}
			


			SimpleHash hash()
			{
				SimpleHash hash;
				hash.begin_hash();
				hash.add_item( in_size );
				hash.add_item( out_size );
				hash.add_item( extra_info );
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