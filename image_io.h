#pragma once
#include <Windows.h>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <assert.h>
#include <functional>
#include <random>
using namespace std;

namespace iio
{
	#pragma pack(push, 1)
	struct RGB
	{
		BYTE r, g, b;
	};
	#pragma pack(pop)

	struct point_t
	{
		int x, y;
		point_t( int x, int y ) : x( x ), y( y )
		{

		}

		inline point_t operator+( const point_t& o ) const
		{
			return{
				x + o.x,
				y + o.y
			};
		}
		inline point_t operator-( const point_t& o ) const
		{
			return{
				x - o.x,
				y - o.y
			};
		}
		inline point_t operator*( const point_t& o ) const
		{
			return{
				x * o.x,
				y * o.y
			};
		}

		inline int linearize() const
		{
			return x * y;
		}

		inline int linearize( const point_t size ) const
		{
			return y * (size.x) + x;
		}
	};
	struct image_t
	{
		int width;
		int height;
		BYTE * raw_data;

		image_t() = delete;
		~image_t() = delete;

		static image_t* create_image( int w, int h )
		{
			image_t * t = (image_t*)(malloc( sizeof( image_t ) ));
			t->width = w;
			t->height = h;
			t->raw_data = new BYTE[w*h * 3];
			return t;
		}

		RGB* get_rgb()
		{
			return (RGB*)raw_data;
		}

		RGB& get_pixel( int w, int h )
		{
			assert( w < this->width && h < this->height );
			return get_rgb()[w + h * this->width];
		}

		static void destroy_image( image_t* img )
		{
			delete[] img->raw_data;
			free( img );
		}
		static image_t* copy_image( image_t* img )
		{
			image_t* new_img = create_image( img->width, img->height );
			memcpy(
				new_img->get_rgb(),
				img->get_rgb(),
				sizeof( RGB ) * img->width * img->height
			);
			return new_img;
		}

		image_t* copy()
		{
			return copy_image( this );
		}

		void destroy()
		{
			destroy_image( this );
		}
	};

	namespace utility
	{
		void loop_through( point_t from, point_t to, const function<void( point_t )>& fn )
		{
			for ( int j = from.y; j < to.y; j++ )
			{
				for ( int i = from.x; i < to.x; i++ )
				{
					fn( { i, j } );
				}
			}
		}
	};

	namespace raw_out
	{

		enum structure
		{
			rrggbb = 1,
			rgbrgb = 2,
			yyuuvv = 4,
			grayscale = 8
		};
		enum type
		{
			fl32 = 0x00AA0000,
		};
		enum format
		{
			fl32_yyuuvv = fl32 | yyuuvv,
			fl32_rrggbb = fl32 | rrggbb,
			fl32_rgbrgb = fl32 | rgbrgb,
			fl32_grayscale = fl32 | grayscale
		};

		void* get_raw_data( image_t*img, format frm )
		{
			if ( frm & fl32 )
			{
				if ( frm & rrggbb )
				{
					float * out = new float[img->width*img->height * 3];
					int dim2 = img->width * img->height;
					utility::loop_through( { 0, 0 }, { img->width, img->height }, [&]( point_t p )
					{
						RGB color = img->get_pixel( p.x, p.y );
						out[0 * dim2 + (p.x + p.y*img->width)] = color.r / 255.0f;
						out[1 * dim2 + (p.x + p.y*img->width)] = color.g / 255.0f;
						out[2 * dim2 + (p.x + p.y*img->width)] = color.b / 255.0f;
					} );
					return out;
				}
				else if ( frm & yyuuvv )
				{
					float * out = new float[img->width*img->height * 3];
					int dim2 = img->width * img->height;
					utility::loop_through( { 0, 0 }, { img->width, img->height }, [&]( point_t p )
					{
						RGB color = img->get_pixel( p.x, p.y );
						float r = color.r / 255.0f;
						float g = color.g / 255.0f;
						float b = color.b / 255.0f;

						float y = 0.299 * r + 0.587 * g + 0.114 * b;
						float u = 0.492 * (b - y);
						float v = 0.877 * (r - y);

						y = max( min( 1.0, y ), 0.0 );
						u = max( min( 1.0, u ), 0.0 );
						v = max( min( 1.0, v ), 0.0 );

						out[0 * dim2 + (p.x + p.y*img->width)] = y;
						out[1 * dim2 + (p.x + p.y*img->width)] = u;
						out[2 * dim2 + (p.x + p.y*img->width)] = v;
					} );
					return out;
				}
				else if ( frm & rgbrgb )
				{
					float * out = new float[img->width*img->height * 3];
					utility::loop_through( { 0, 0 }, { img->width, img->height }, [&]( point_t p )
					{
						RGB color = img->get_pixel( p.x, p.y );
						out[3 * (p.x + p.y*img->width) + 0] = color.r / 255.0f;
						out[3 * (p.x + p.y*img->width) + 1] = color.g / 255.0f;
						out[3 * (p.x + p.y*img->width) + 2] = color.b / 255.0f;
					} );
					return out;
				}
				else if ( frm & grayscale )
				{
					float * out = new float[img->width*img->height];
					utility::loop_through( { 0, 0 }, { img->width, img->height }, [&]( point_t p )
					{
						RGB color = img->get_pixel( p.x, p.y );
						out[p.x + p.y*img->width] = (color.r + color.g + color.b) / (255.0f * 3);
					} );
					return out;
				}
			}
			return nullptr;
		}

	};

	namespace simple_io
	{
		BYTE* read_file( const string& path )
		{
			ifstream file( path, ios::binary | ios::ate );
			streamsize size = file.tellg();
			file.seekg( 0, ios::beg );

			if ( size == -1 )
				return nullptr;

			BYTE* buffer = new BYTE[size];
			file.read( (char*)buffer, size );
			return buffer;
		}

		void write_file( const string& path, BYTE* in, size_t size )
		{
			ofstream file;
			file.open( path, ios_base::binary );
			file.write( (char*)in, size );
			file.close();
		}
	};

	namespace ppm
	{
		int create_ppm_header( BYTE * out, int w, int h )
		{
			string s;
			s += "P6";
			s += '\x0A';
			s += to_string( w );
			s += '\x0A';
			s += to_string( h );
			s += '\x0A';
			s += "255";
			s += '\x0A';

			if ( out )
			{
				memcpy( out, s.data(), s.length() );
				return 0;
			}
			else
			{
				return s.length();
			}
		}
		image_t* load( const string& path )
		{
			const auto next_alnum = []( char* file ) -> char*
			{
				while ( !isalnum( *file ) ) file++;
				return file;
			};

			char* file = (char*)simple_io::read_file( path );
			if ( !file )
				return nullptr;


			char * res = next_alnum( file + 2 );

			int state = 0;
			int w = 0, h = 0, max_col = 0;

			char * color = 0;

			for ( int i = 0; ; i++ )
			{
				if ( !isalnum( res[i] ) )
				{
					if ( state == 2 )
					{
						color = res + i + 1;
						break;
					}
					state++;
					continue;
				}
				if ( state == 0 )
					w = w * 10 + (res[i] - '0');
				if ( state == 1 )
					h = h * 10 + (res[i] - '0');
				if ( state == 2 )
					max_col = max_col * 10 + (res[i] - '0');
			}

			assert( w != 0 && h != 0 && max_col > 0 );

			image_t* img = image_t::create_image( w, h );

			RGB * rgb = (RGB*)color;

			memcpy(
				img->get_rgb(),
				rgb,
				sizeof( RGB ) * w * h
			);

			delete[] file;

			return img;


		}
		void save( image_t* img, const string& path )
		{
			int header_size = create_ppm_header( NULL, img->width, img->height );
			int total_size = header_size + (img->width * img->height) * sizeof( RGB );
			BYTE * bytes = new BYTE[total_size];
			create_ppm_header( bytes, img->width, img->height );
			memcpy(
				bytes + header_size,
				img->get_rgb(),
				(img->width * img->height) * sizeof( RGB )
			);
			simple_io::write_file( path, bytes, total_size );
			delete[] bytes;
		}
	}

	namespace windows
	{
		template<typename T>
		void copy_rgb( HBITMAP hBitmap, T * data )
		{
			BITMAP bmp = { 0 };
			BITMAPINFO header = { 0 };

			HDC     hScreen = GetDC( NULL );
			HDC     DC = CreateCompatibleDC( hScreen );

			GetObject( hBitmap, sizeof( bmp ), &bmp );

			int w = bmp.bmWidth;
			int h = bmp.bmHeight;

			header.bmiHeader.biSize = sizeof( BITMAPINFOHEADER );
			header.bmiHeader.biWidth = w;
			header.bmiHeader.biHeight = h;
			header.bmiHeader.biPlanes = 1;
			header.bmiHeader.biBitCount = bmp.bmBitsPixel;
			header.bmiHeader.biCompression = BI_RGB;
			header.bmiHeader.biSizeImage = ((w * bmp.bmBitsPixel + 31) / 32) * 4 * h;

			BYTE * bits = new BYTE[header.bmiHeader.biSizeImage];

			GetDIBits( DC, hBitmap, 0, h, bits, &header, DIB_RGB_COLORS );

			for ( int y = 0; y < h; y++ )
			{
				for ( int x = 0; x < w; x++ )
				{
					int col = x;
					int row = h - y - 1;
					int index = (row * w + col) * 4;

					BYTE r = bits[index + 2];
					BYTE g = bits[index + 1];
					BYTE b = bits[index + 0];

					data[3 * (x + y*w) + 0] = r;
					data[3 * (x + y*w) + 1] = g;
					data[3 * (x + y*w) + 2] = b;
				}
			}

			delete[] bits;
			DeleteDC( DC );
		}
		template<typename T>
		void screenshot( POINT a, POINT b, T * data )
		{
			HDC     hScreen = GetDC( NULL );
			HDC     hDC = CreateCompatibleDC( hScreen );
			HBITMAP hBitmap = CreateCompatibleBitmap( hScreen, abs( b.x - a.x ), abs( b.y - a.y ) );
			HGDIOBJ old_obj = SelectObject( hDC, hBitmap );
			BOOL    bRet = BitBlt( hDC, 0, 0, abs( b.x - a.x ), abs( b.y - a.y ), hScreen, a.x, a.y, SRCCOPY );

			copy_rgb<T>( hBitmap, data );

			SelectObject( hDC, old_obj );
			DeleteDC( hDC );
			ReleaseDC( NULL, hScreen );
			DeleteObject( hBitmap );
		}
		image_t* screenshot( point_t a, int w, int h )
		{
			image_t * img = image_t::create_image( w, h );
			screenshot( POINT{ a.x, a.y }, POINT{ a.x + w, a.y + h }, img->raw_data );
			return img;
		}
	};

	namespace processing
	{
		BYTE get_max_color( image_t* img, int col, float percentile )
		{
			int histo[256];
			ZeroMemory( histo, 256 * sizeof( int ) );

			for ( int i = 0; i < img->width; i++ )
			{
				for ( int j = 0; j < img->height; j++ )
				{
					histo[img->raw_data[3 * (i + j * img->width) + col]]++;
				}
			}
			int target = img->width*img->height*percentile;
			int reached = 0;
			int at = 255;

			while ( reached < target )
				reached += histo[at--];
			return at;
		}
		BYTE get_min_color( image_t* img, int col, float percentile )
		{
			int histo[256];
			ZeroMemory( histo, 256 * sizeof( int ) );

			for ( int i = 0; i < img->width; i++ )
			{
				for ( int j = 0; j < img->height; j++ )
				{
					histo[img->raw_data[3 * (i + j * img->width) + col]]++;
				}
			}

			int target = img->width*img->height*percentile;
			int reached = 0;
			int at = 0;

			while ( reached < target )
				reached += histo[at++];
			return at;
		}
		void equalize( image_t* img, float percentile, bool treat_all_colors_same = false )
		{
			assert( percentile < 1 && 0 < percentile );
			percentile /= 2;

			BYTE mins[] = {
				get_min_color( img, 0, percentile ),
				get_min_color( img, 1, percentile ),
				get_min_color( img, 2, percentile )
			};

			BYTE maxes[] = {
				get_max_color( img, 0, percentile ),
				get_max_color( img, 1, percentile ),
				get_max_color( img, 2, percentile )
			};

			if ( treat_all_colors_same )
			{
				BYTE mi = min( min( mins[0], mins[1] ), mins[2] );
				BYTE ma = max( max( maxes[0], maxes[1] ), maxes[2] );

				mins[0] = mi;
				mins[1] = mi;
				mins[2] = mi;

				maxes[0] = ma;
				maxes[1] = ma;
				maxes[2] = ma;
			}

			for ( int i = 0; i < 3; i++ )
			{
				// fix channels
				BYTE a = mins[i];
				BYTE b = maxes[i];

				mins[i] = min( a, b );
				maxes[i] = max( a, b );
			}

			const auto fix_color = [&]( BYTE b, int c ) -> BYTE
			{
				BYTE mn = mins[c];
				BYTE mx = maxes[c];

				BYTE rn = mx - mn;

				float fx = (((float)(b - mn) / (float)rn) * 255.0f);
				float fxe = max( min( fx, 255.0f ), 0.0f );

				return (BYTE)fxe;
			};

			for ( int i = 0; i < img->width; i++ )
			{
				for ( int j = 0; j < img->height; j++ )
				{
					for ( int c = 0; c < 3; c++ )
					{
						BYTE& pixel = img->raw_data[3 * (i + j * img->width) + c];
						pixel = fix_color( pixel, c );
					}
				}
			}
		}
		void flip_vertical( image_t* img )
		{
			for ( int j = 0; j < img->height; j++ )
			{
				for ( int i = 0; i < floor( img->width / 2.0f ) - 1.0f; i++ )
				{
					int i_c = img->width - i - 1;
					swap( img->get_pixel( i, j ), img->get_pixel( i_c, j ) );
				}
			}
		}
		void flip_horizontal( image_t* img )
		{
			for ( int i = 0; i < img->width; i++ )
			{
				for ( int j = 0; j < floor( img->height / 2.0f ) - 1.0f; j++ )
				{
					int j_c = img->height - j - 1;
					swap( img->get_pixel( i, j ), img->get_pixel( i, j_c ) );
				}
			}
		}
		void copy_to( image_t* i_dst, image_t* i_src, point_t dst_point, point_t src_point, point_t src_size )
		{
			utility::loop_through( { 0, 0 }, src_size - point_t{ 1, 1 }, [&]( point_t p )
			{
				point_t src = src_point + p;
				point_t dst = dst_point + p;
				i_dst->get_pixel( dst.x, dst.y ) = i_src->get_pixel( src.x, src.y );
			} );
		}
		void fill_color( image_t* img, RGB r )
		{
			for ( int i = 0; i < img->width * img->height; i++ )
				img->get_rgb()[i] = r;
		}
		void add_noise( image_t* img, float percentage )
		{
			for ( int i = 0; i < img->width * img->height; i++ )
			{
				float unir = rand() / float( RAND_MAX );
				float unig = rand() / float( RAND_MAX );
				float unib = rand() / float( RAND_MAX );

				float distort_rater = 1.0f + unir * (2 * percentage) - percentage;
				float distort_rateg = 1.0f + unig * (2 * percentage) - percentage;
				float distort_rateb = 1.0f + unib * (2 * percentage) - percentage;

				RGB& rgb = img->get_rgb()[i];

				float new_r = (rgb.r * distort_rater);
				float new_g = (rgb.g * distort_rateg);
				float new_b = (rgb.b * distort_rateb);

				new_r = max( min( new_r, 255.0f ), 0.0f );
				new_g = max( min( new_g, 255.0f ), 0.0f );
				new_b = max( min( new_b, 255.0f ), 0.0f );

				rgb.r = BYTE( new_r );
				rgb.g = BYTE( new_g );
				rgb.b = BYTE( new_b );
			}
		}

		void augment( image_t* img, int rnd, float eq = -1 )
		{
			std::default_random_engine generator;
			std::uniform_real_distribution<float> float_d( 0.0f, 1.0f );
			std::uniform_int_distribution<int> bool_d( 0, 1 );
			std::uniform_int_distribution<int> int_d( 0, RAND_MAX );
			generator.seed( rnd );

			if ( bool_d( generator ) )
				flip_horizontal( img );
			if ( bool_d( generator ) )
				flip_vertical( img );

			if ( eq != -1 )
				equalize( img, eq, true );
			else
				equalize( img, float_d( generator ) * 0.08, true );

			add_noise( img, float_d( generator ) * 0.05 );
		}
	};
}