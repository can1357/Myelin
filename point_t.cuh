#pragma once
#include "cuda_simplifier.cuh"
#include <string>
struct point_t
{
	int x, y, z;

	inline __device__ __host__ point_t( int x, int y, int z )
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	inline __device__ __host__ point_t( )
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	inline __device__ __host__ point_t( int i, const point_t size )
	{
		x = i % size.x;
		i -= x;

		y = (i % (size.x * size.y)) / size.x;
		i -= y;

		z = i / (size.x * size.y);
	}

	inline __device__ __host__ point_t operator+( const point_t& o ) const
	{
		return{
			x + o.x,
			y + o.y,
			z + o.z
		};
	}
	inline __device__ __host__ point_t operator-( const point_t& o ) const
	{
		return{
			x - o.x,
			y - o.y,
			z - o.z
		};
	}
	inline __device__ __host__ point_t operator*( const point_t& o ) const
	{
		return{
			x * o.x,
			y * o.y,
			z * o.z
		};
	}

	inline __device__ __host__ int linearize() const
	{
		return x * y * z;
	}

	inline __device__ __host__ int linearize( const point_t size ) const
	{
		return z * (size.x * size.y) +
			y * (size.x) +
			x;
	}

	inline std::string to_string() const
	{
		std::string str_x = std::to_string( x );
		std::string str_y = std::to_string( y );
		std::string str_z = std::to_string( z );
		return "[" + str_x + ", " + str_y + ", " + str_z + "]";
	}
};
using vsize_t = point_t;