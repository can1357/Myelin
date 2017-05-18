#pragma once
#include "cuda_simplifier.cuh"

namespace Myelin
{
	namespace Activators
	{
		enum class ActivatorID
		{
			sigmoid,
			arctan,
			tanh,
			relu,
			leaky_relu,
			divisor,
			tanh_stricter,
			none,
			binary,
		};

		struct ActivatorIdentifier
		{
			ActivatorID id;
			float param;
		};

		std::string GetActivatorName( ActivatorIdentifier act )
		{
			switch ( act.id )
			{
				case	ActivatorID::sigmoid:
					return "Sigmoid";
				case 	ActivatorID::binary:
					return "Binary";
				case 	ActivatorID::tanh:
					return "Tanh";
				case 	ActivatorID::tanh_stricter:
					return "Tanh Stricter";
				case 	ActivatorID::arctan:
					return "Arctan * 1/" + std::to_string( act.param );
				case 	ActivatorID::relu:
					return "ReLU";
				case 	ActivatorID::leaky_relu:
					return "Leaky ReLU a=" + std::to_string(act.param);
				case 	ActivatorID::divisor:
					return "Divisor x=" + std::to_string( act.param );
				case 	ActivatorID::none:
				default:
					return "None";
			}
		}

		__device__ inline static float Activate( ActivatorIdentifier act, float x )
		{
			switch ( act.id )
			{
				case	ActivatorID::sigmoid:
					return (1.0f / (1.0f + expf( -x )));
				case 	ActivatorID::tanh:
					return tanhf( x );
				case 	ActivatorID::binary:
					return x >= 0.5 ? 1.0f : 0.0f;
				case 	ActivatorID::tanh_stricter:
					return tanhf( x * act.param );
				case 	ActivatorID::arctan:
					return atanf( x ) / act.param;
				case 	ActivatorID::relu:
					if ( x > 0 )
						return x;
					return 0;
				case 	ActivatorID::leaky_relu:
					if ( x > 0 )
						return x;
					return act.param * x;
				case 	ActivatorID::divisor:
					return x / act.param;
				case 	ActivatorID::none:
				default:
					return x;
			}
		}

		__device__ inline static float Derive( ActivatorIdentifier act, float x )
		{
			float temp = 0.0f;
			switch ( act.id )
			{
				case	ActivatorID::sigmoid:
					temp = (1.0f / (1.0f + expf( -x )));
					return temp * (1 - temp);
				case 	ActivatorID::tanh:
					temp = tanhf( x );
					return 1 - temp*temp;
				case 	ActivatorID::tanh_stricter:
					temp = tanhf( x * act.param );
					return (1 - temp*temp) * act.param;
				case 	ActivatorID::arctan:
					return 1.0f / (act.param * (x*x + 1));
				case 	ActivatorID::relu:
					if ( x > 0 )
						return 1;
					return 0;
				case 	ActivatorID::leaky_relu:
					if ( x > 0 )
						return 1;
					return act.param;
				case 	ActivatorID::divisor:
					return 1 / act.param;
				case 	ActivatorID::binary:
					return 1.0f;
				case 	ActivatorID::none:
				default:
					return 1;
			}
		}
	};
};