#pragma once
#include <random>
#include <ctime>
#include <functional>

namespace Myelin
{
	namespace Core
	{
		namespace Random
		{
			static std::default_random_engine generator( time( NULL ) );

			using Factory = std::function<float( int )>;
			class NormalDistribution
			{
			public:
				std::normal_distribution<float> distribution;

				NormalDistribution( float mean, float std ) :
					distribution( mean, std )
				{
				}

				NormalDistribution() :
					distribution( 0, 1 )
				{
				}

				operator Factory()
				{
					auto x = [&]( int x )
					{
						return this->generate();
					};
					return x;
				}

				float generate()
				{
					return distribution( generator );
				}
			};
			class UniformDistribution
			{
			public:
				std::uniform_real_distribution<float> distribution;

				UniformDistribution( float mn, float mx ) :
					distribution(mn, mx)
				{
				}

				UniformDistribution() :
					distribution( -1,- 1 )
				{
				}

				operator Factory()
				{
					auto x = [&]( int x )
					{
						return this->generate();
					};
					return x;
				}

				float generate()
				{
					return distribution( generator );
				}
			};
		};
	}
}