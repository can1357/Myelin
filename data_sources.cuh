#pragma once
#include "cuda_simplifier.cuh"
#include "image_io.h"
#include <vector>

namespace Myelin
{
	namespace DataSource
	{
		static std::vector<std::string> list_files( const std::string& folder, const std::string& extension )
		{
			std::vector<std::string> files;
			WIN32_FIND_DATA find_file_data;
			std::string search = folder + "*." + extension;
			HANDLE h_find = FindFirstFile( search.c_str(), &find_file_data );
			while ( h_find != INVALID_HANDLE_VALUE )
			{
				if ( !FindNextFileA( h_find, &find_file_data ) ) break;
				files.push_back( string( find_file_data.cFileName ) );
			}
			FindClose( h_find );
			return files;
		}

		static std::string replace_all( std::string str, const std::string& from, const std::string& to )
		{
			size_t start_pos = 0;
			while ( (start_pos = str.find( from, start_pos )) != std::string::npos )
			{
				str.replace( start_pos, from.length(), to );
				start_pos += to.length();
			}
			return str;
		}

		class LabeledImages
		{
		public:
			struct class_info_t
			{
				std::vector<std::string> names;
				std::vector<int> abundance;
				std::vector<std::vector<std::string>> files;

				int get_count()
				{
					return names.size();
				}

				void add( const std::string& to_add, const std::string& file_name )
				{
					for ( int i = 0; i < names.size(); i++ )
					{
						if ( names[i] == to_add )
						{
							abundance[i]++;
							files[i].push_back( file_name );
							return;
						}
					}

					files.push_back( { file_name } );
					names.push_back( to_add );
					abundance.push_back( 1 );
				}
			};

			struct class_t
			{
				Myelin::CUDA::shared_array_t<float> in;
				Myelin::CUDA::shared_array_t<float> out;
				int data_filled;
				int data_max;
				int class_id;
				size_t in_size;
				size_t out_size;

				class_t( int class_id, int n_data, size_t in_size, size_t out_size )
					:
					data_filled( 0 ),
					data_max( n_data ),
					class_id( class_id ),
					in_size( in_size ),
					out_size( out_size ),
					out( out_size ),
					in(in_size * n_data)
				{
					out.cpu[class_id] = 1.0f;
					out.sync_gpu();
				}

				void add_data( float * src )
				{
					assert( data_filled < data_max );
					float * trgt_ptr = (float*)((char*)in.gpu + (in_size * data_filled));
					CHK_CUERROR( cudaMemcpy( trgt_ptr, src, in_size, cudaMemcpyHostToDevice ) );
				}
			};

			static class_info_t get_classes( const std::string& folder, const std::vector<std::string>& ignore_list )
			{
				auto files = list_files( folder, "ppm" );

				class_info_t classes;

				for ( std::string str : files )
				{
					std::string file_name = str;
					for ( const std::string& ignore : ignore_list )
					{
						str = replace_all( str, ignore, "" );
						str = replace_all( str, "_", "" );
						for( int i = 0; i <= 9; i++ )
							str = replace_all( str, std::to_string(i), "" );
						str = replace_all( str, ".ppm", "" );
					}
					classes.add( str, folder + "\\" + file_name );
				}

				return classes;
			}

			vector<class_t> classes;
			class_info_t class_info;

			LabeledImages( const std::string& folder )
				: class_info( get_classes( folder, { "aug" } ) )
			{
				for ( int c = 0; c < class_info.get_count(); c++ )
				{
					printf( "Class %s [%d]\n", class_info.names[c].c_str(), class_info.abundance[c] );

					vector<iio::image_t*> images;
					for ( std::string& file : class_info.files[c] )
						images.push_back(iio::ppm::load( file ));

					class_t cl( c, class_info.abundance[c], images[0]->width * images[0]->height * 3 * sizeof( float ), class_info.get_count() * sizeof( float ) );

					for ( auto img : images )
					{
						void * d = iio::raw_out::get_raw_data( img, iio::raw_out::format::fl32_rrggbb );
						cl.add_data( (float*)d );
						delete[] d;
						img->destroy();
					}

					images.clear();
					classes.push_back( cl );
				}
			}
		};
	};
};