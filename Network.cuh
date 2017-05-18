#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <inttypes.h>
#include <functional>
#include <assert.h>
#include "cuda_simplifier.cuh"
#include "point_t.cuh"
#include "activators.cuh"
#include "optimizers.cuh"
#include "error_functions.cuh"
#include "device_launch_parameters.h"
#include "core.cuh"
#include <algorithm>b

namespace Myelin
{
	static void* read_file( const std::string& path )
	{
		std::ifstream file( path, std::ios::binary | std::ios::ate );
		std::streamsize size = file.tellg();
		file.seekg( 0, std::ios::beg );

		if ( size == -1 )
			return nullptr;

		BYTE* buffer = new BYTE[size];
		file.read( (char*) buffer, size );
		return buffer;
	}

	template<typename T>
	static void load_folder( vector<T>& to, const std::string& path, string search_sf )
	{
		vector<T> ar;

		WIN32_FIND_DATA find_file_data;
		std::string search = path + search_sf;
		HANDLE list_handle = FindFirstFile( search.c_str(), &find_file_data );
		while ( list_handle != INVALID_HANDLE_VALUE )
		{
			void * file = read_file( path + find_file_data.cFileName );
			T* str = (T*) file;
			to.push_back( *str );
			delete[] file;

			if ( !FindNextFileA( list_handle, &find_file_data ) ) break;
		}
		FindClose( list_handle );
	}

	template<typename T>
	class memory_batch
	{
		void reset_batch()
		{
			internal_counter = 0;
			mem_history = 0;
			std::shuffle( data->begin(), data->end(), Myelin::Core::Random::generator );
		}

		int mem_history = 0;
		int idx_start = 0;
		bool cache()
		{
			int data_to_cache = data->size() - mem_history;
			data_to_cache = min( data_to_cache, int( mem_batch_size ) );
			if ( data_to_cache <= 0 )
				return false;

			T* start = &data->at( mem_history );
			size_t size = sizeof( T ) * data_to_cache;

			CHK_CUERROR( cudaMemcpy( gpu_raw, start, size, cudaMemcpyHostToDevice ) );
			idx_start = mem_history;
			mem_history += data_to_cache;
			return true;
		}

		public:
		int internal_counter = 0;
		T * gpu_raw;
		vector<T>* data;
		size_t mem_batch_size;
		memory_batch( vector<T>* data, size_t mem_batch_size ) :
			data( data )
		{
			mem_batch_size = min( mem_batch_size, data->size() );
			this->mem_batch_size = mem_batch_size;

			CHK_CUERROR( cudaMalloc( &gpu_raw, mem_batch_size * sizeof( T ) ) );
			reset_batch();
		}

		~memory_batch()
		{
			CHK_CUERROR( cudaFree( gpu_raw ) );
		}

		T* pick()
		{
			if ( mem_history == internal_counter )
			{
				//printf( "END BATCH\n" );
				while ( !cache() )
				{
					//printf( "RESET BATCH\n\n" );
					reset_batch();
				}
			}
			int gpu_id = internal_counter - idx_start;
			internal_counter++;
			return gpu_raw + gpu_id;
		}
	};

	namespace Layers
	{
		struct SimpleHash
		{
			uint32_t hash;

			void begin_hash()
			{
				hash = 0xEDB88320;
			}

			template<typename T>
			void add_item( T& x )
			{
				size_t total = sizeof( T );

				uint32_t * ptr = (uint32_t*)&x;

				while ( total > sizeof( uint32_t ) )
				{
					hash ^= *ptr;
					total -= sizeof( uint32_t );
					ptr++;
				}

				uint32_t left_over = 0;
				memcpy( &left_over, ptr, total );
				hash += left_over;
			}

			uint32_t end_hash()
			{
				return hash;
			}
		};

		struct SaveHeader
		{
			size_t num_bytes;
			SimpleHash hash;
		};

		struct SaveBlob
		{
			vector<BYTE> s;
			BYTE* allocate( int x )
			{
				int i = s.size();
				s.resize( s.size() + x );
				return s.data() + i;
			}

			template<typename T>
			void push( Myelin::CUDA::shared_array_t<T>& r )
			{
				void * x = this->allocate( r.get_num_bytes() );
				r.sync_cpu();
				memcpy( x, r.cpu, r.get_num_bytes() );
			}

			template<typename T>
			void pop( Myelin::CUDA::shared_array_t<T>& r )
			{
				vector<BYTE> newx( s.size() - r.get_num_bytes() );
				memcpy( newx.data(), s.data() + r.get_num_bytes(), s.size() - r.get_num_bytes() );
				memcpy( r.cpu, s.data(), r.get_num_bytes() );
				r.sync_gpu();
				s = newx;
			}

			SaveHeader* to_savehdr( SimpleHash hsh )
			{
				int num_bytes = sizeof( SaveHeader ) + s.size();
				char * out = new char[num_bytes];

				SaveHeader * header = (SaveHeader*) out;
				header->hash = hsh;
				header->num_bytes = num_bytes;

				memcpy( out + sizeof( SaveHeader ), s.data(), s.size() );
				return header;
			}

			static SaveBlob from_savehdr( SaveHeader* hdr )
			{
				SaveBlob b;
				int blob_size = hdr->num_bytes - sizeof( SaveHeader );
				b.s.resize( blob_size );
				memcpy( b.s.data(), ( (char*) hdr ) + sizeof( SaveHeader ), blob_size );
				return b;
			}
		};

		class ILayer
		{
		public:
			virtual void calc_grads( float const* grads_next ) = 0;
			virtual void activate() = 0;
			virtual void update_weights( float lr_mutlp, cudaStream_t str ) = 0;
			virtual Myelin::CUDA::shared_array_t<float>& get_output() = 0;
			virtual Myelin::CUDA::shared_array_t<float>& get_input() = 0;
			virtual Myelin::CUDA::shared_array_t<float>& get_grads() = 0;
			virtual vsize_t get_input_size() = 0;
			virtual vsize_t get_output_size() = 0;
			virtual std::string get_name() = 0;
			virtual std::string get_short_name() = 0;
			virtual SaveHeader* save() = 0;
			virtual bool load( SaveHeader* data ) = 0;
			virtual ILayer* copy( bool copy_state ) = 0;
			void activate( float * gpu_in )
			{
				CHK_CUERROR( cudaMemcpy( this->get_input().gpu, gpu_in, this->get_input_size().linearize() * sizeof( float ), cudaMemcpyDeviceToDevice ) );
				this->activate();
			}
		};
	};

	class Network
	{
	protected:
		bool finished = false;
	public:
		vsize_t in_size, out_size;
		std::vector<float> error_history;
		std::vector<cudaStream_t> streams;
		std::vector<Layers::ILayer*> layers;
		Myelin::ErrorFunction::ErrorFunction err_function = Myelin::ErrorFunction::ErrorFunction(Myelin::ErrorFunction::ErrorFunctionID::MSE);
		Myelin::CUDA::shared_array_t<float> * grads = nullptr;
		Myelin::CUDA::shared_array_t<Myelin::ErrorFunction::cost_counter_t> * cost_counter = nullptr;
		int num_g_update = 0;
		int num_w_update = 0;

		uint32_t hash_network()
		{
			string hsh = "";
			for ( auto layer : layers )
				hsh += layer->get_name();
			return std::hash<string>{}( hsh );
		}

		void next()
		{
			if ( num_g_update % 100 == 0 )
			{
				auto cost = get_cost();
				error_history.push_back( cost.get_average_cost() );
			}

			if ( num_g_update % 10000 == 0 )
			{
				auto cost = get_cost();
				cost.reset_counter();
				*cost_counter->cpu = cost;
				cost_counter->sync_gpu();
			}
		}

		void copy_from( Network& n2 )
		{
			this->finished = n2.finished;
			this->in_size = n2.in_size;
			this->out_size = n2.out_size;
			this->num_g_update = n2.num_g_update;
			this->num_w_update = n2.num_w_update;

			for ( Layers::ILayer* layer : n2.layers )
				this->layers.push_back( layer->copy( true ) );


			for ( int i = 0; i < this->layers.size(); i++ )
			{
				cudaStream_t str;
				CHK_CUERROR( cudaStreamCreate( &str ) );
				this->streams.push_back( str );
			}

			this->err_function = n2.err_function;


			this->grads = new Myelin::CUDA::shared_array_t<float>( out_size.linearize() );
			this->grads->copy_from( *n2.grads );

			this->cost_counter = new Myelin::CUDA::shared_array_t<Myelin::ErrorFunction::cost_counter_t>( 1 );
			this->cost_counter->copy_from( *n2.cost_counter );
		}

		struct runtime_info
		{
			int num_g_update;
			int num_w_update;
			int num_layers;
			int num_errors;
			float error[1];
		};

		static void write_file( const std::string& path, void* in, size_t size )
		{
			std::ofstream file;
			file.open( path, std::ios_base::binary );
			file.write( (char*)in, size );
			file.close();
		}

		void load_layers( const std::string& directory, bool strict = true )
		{
			CHK_CUERROR( cudaDeviceSynchronize() );

			error_history.clear();

			void * net_info_bytes = read_file( directory + "newtork.info" );
			runtime_info * net_info = (runtime_info*)(net_info_bytes);

			if ( !net_info_bytes )
			{
				printf( "Error! Network runtime information could not be found!\n" );
				delete[] net_info_bytes;
				return;
			}

			if ( net_info->num_layers != layers.size() )
			{
				printf( "Error! Networks do not match!\n" );
				if( strict )
				{
					delete[] net_info_bytes;
					return;
				}
			}

			num_g_update = net_info->num_g_update;
			num_w_update = net_info->num_w_update;

			for ( int i = 0; i < net_info->num_errors; i++ )
				error_history.push_back( net_info->error[i] );

			delete[] net_info_bytes;

			int layer_id = 0;
			for ( Layers::ILayer* layer : layers )
			{
				std::string full_path = directory + layer->get_short_name() + "_" + std::to_string( layer_id ) + ".layer";
				if ( !layer->load( nullptr ) )
				{
					void * save = read_file( full_path );
					if ( save )
					{
						if ( !layer->load( (Layers::SaveHeader*)save ) )
						{
							printf( "Error Layer %d [%s] cannot be loaded!\n", layer_id, layer->get_name().c_str() );
							return;
						}

						delete[] save;
					}
					else
					{
						printf( "Error %s not found!\n", full_path.c_str() );
						return;
					}
				}
				layer_id++;
			}
		}

		Myelin::ErrorFunction::cost_counter_t get_cost()
		{
			assert( cost_counter );
			cost_counter->sync_cpu();
			return *cost_counter->cpu;
		}

		void save_layers( const std::string& directory )
		{
			CHK_CUERROR( cudaDeviceSynchronize() );

			CreateDirectoryA( directory.c_str(), NULL );
			int layer_id = 0;
			for ( Layers::ILayer* layer : layers )
			{
				std::string full_path = directory + layer->get_short_name() + "_" + std::to_string( layer_id ) + ".layer";
				if ( Layers::SaveHeader * save = layer->save() )
				{
					write_file( full_path, save, save->num_bytes );
					delete[] save;
				}
				layer_id++;
			}



			size_t num_bytes = sizeof( runtime_error ) + (error_history.size() - 1) * sizeof( float );
			BYTE * output = new BYTE[num_bytes];


			runtime_info * data = new (output) runtime_info;

			data->num_g_update = num_g_update;
			data->num_w_update = num_w_update;
			data->num_layers = layers.size();
			data->num_errors = error_history.size();

			for ( int i = 0; i < data->num_errors; i++ )
				data->error[i] = error_history[i];
			
			write_file( directory + "newtork.info", output, num_bytes );

			delete[] output;
		}

		Network( vsize_t in_size ):
			in_size(in_size),
			out_size(in_size)
		{
		}

		Network() :
			in_size( {0,0,0} ),
			out_size( { 0,0,0 } )
		{
		}

		void add_layer( Layers::ILayer* layer )
		{
			assert( !finished );

			layers.push_back( layer );
			out_size = layer->get_output_size();
		}

		bool is_finalized()
		{
			return finished;
		}

		void finalize( Myelin::ErrorFunction::ErrorFunction error_function )
		{
			assert( layers.size() > 0 );

			cost_counter = new Myelin::CUDA::shared_array_t<Myelin::ErrorFunction::cost_counter_t>( 1 );
			grads = new Myelin::CUDA::shared_array_t<float>( out_size.linearize() );
			err_function = error_function;
			finished = true;
			in_size = layers[0]->get_input_size();

			for ( int i = 0; i < layers.size(); i++ )
			{
				cudaStream_t str;
				CHK_CUERROR( cudaStreamCreate( &str ) );
				streams.push_back( str );
			}
		}

		void finalize( Myelin::ErrorFunction::ErrorFunctionID error_function )
		{
			this->finalize( Myelin::ErrorFunction::ErrorFunction(error_function) );
		}

		void activate( float * in_gpu )
		{
			assert( is_finalized() );

			layers[0]->activate( in_gpu );
			for ( int i = 1; i < layers.size(); i++ )
				layers[i]->activate( layers[i - 1]->get_output().gpu );
		}

		Myelin::CUDA::shared_array_t<float>& get_output()
		{
			return layers.back()->get_output();
		}

		void calc_grads( float * expected_gpu )
		{
			assert( is_finalized() );

			CHK_CUERROR( cudaStreamSynchronize( cudaStreamDefault ) );

			num_g_update++;

			Myelin::CUDA::shared_array_t<float>& output = layers.back()->get_output();

			int work = out_size.linearize();
			Myelin::ErrorFunction::CalculateGradients DWORK( work, cudaStreamDefault ) (
				err_function,
				output.gpu,
				expected_gpu,
				grads->gpu,
				work
				);

			Myelin::ErrorFunction::CalculateCost( err_function, output.gpu, expected_gpu, cost_counter->gpu, work );

			layers.back()->calc_grads( grads->gpu );
			for ( int i = layers.size() - 2; i >= 0; i-- )
				layers[i]->calc_grads( layers[i + 1]->get_grads().gpu );
		}

		void update_weights( float lr_multp )
		{
			assert( is_finalized() );

			num_w_update++;

			for ( int i = 0; i < layers.size(); i++ )
				layers[i]->update_weights( lr_multp, streams[i] );
			/*for ( int i = 0; i < layers.size(); i++ )
				CHK_CUERROR( cudaStreamSynchronize( streams[i] ) );*/
		}

		~Network()
		{
			if ( grads )
				delete grads;
			if ( cost_counter )
				delete cost_counter;

			for ( cudaStream_t& str : streams )
				CHK_CUERROR( cudaStreamDestroy( str ) );
		}
	};
};