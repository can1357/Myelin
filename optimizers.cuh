#pragma once
#include "cuda_simplifier.cuh"

namespace Myelin
{
	namespace Optimizers
	{
		struct OptimizerArgs
		{
			enum class RegularizationType
			{
				L1, // E(w) = E(w) + L|wi|
				L2,  // E(w) = E(w) + L/2(wi)^2
				Sinusodial,  // E(w) = E(w) + (|sin(ax)|-1)^2
			} regularization_type;
			float regularization_coeff; // lambda L

			float momentum;
			float learning_rate;
		};

		class OptimizerArgFactory
		{
		public:
			OptimizerArgs * args;

			OptimizerArgs& get_default()
			{
				return *args;
			}

			OptimizerArgFactory()
			{
				args = new OptimizerArgs;
			}
			~OptimizerArgFactory()
			{
				delete args;
			}
			function<OptimizerArgs()> base()
			{
				OptimizerArgs * source = args;
				return [source]()
				{
					OptimizerArgs x = *source;
					return x;
				};
			}

			function<OptimizerArgs()> multiply_learning_rate( float f )
			{
				OptimizerArgs * source = args;
				return [source, f]()
				{
					OptimizerArgs x = *source;
					x.learning_rate *= f;
					return x;
				};
			}
		};

		enum class OptimizerID
		{
			SGD,
			SGDMomentum
		};

		struct GradHeader
		{
			float gradient;
		};

		struct SGDGrad
		{
			GradHeader header;

			__device__ void update( float& value, OptimizerID optimizer, OptimizerArgs args, float lr_mutlp )
			{
				value -= args.learning_rate * lr_mutlp * this->header.gradient;
				this->header.gradient = 0.0f;
			}
		};

		struct SGDMomentumGrad
		{
			GradHeader header;
			float pervious_gradient;

			__device__ void update( float& value, OptimizerID optimizer, OptimizerArgs args, float lr_mutlp )
			{
				float new_grad = this->header.gradient + args.momentum * this->pervious_gradient;
				this->pervious_gradient = new_grad;
				value -= args.learning_rate * lr_mutlp * new_grad;
				this->header.gradient = 0.0f;
			}
		};

		__device__ __host__ inline static size_t GetGradientSize( OptimizerID opt )
		{
			switch ( opt )
			{
				case	OptimizerID::SGDMomentum:
					return sizeof( SGDMomentumGrad );
				default:
				case	OptimizerID::SGD:
					return sizeof( SGDGrad );
			}
			return 0;
		}

		__device__ inline static void UpdateParameter( GradHeader* _grad, float& value, OptimizerID optimizer, OptimizerArgs args, float lr_multp )
		{
			switch ( optimizer )
			{
				case 	OptimizerID::SGDMomentum:
					((SGDMomentumGrad*)_grad)->update( value, optimizer, args, lr_multp );
					break;
				default:
				case	OptimizerID::SGD:
					((SGDGrad*)_grad)->update( value, optimizer, args, lr_multp );
					break;
			}

			switch ( args.regularization_type )
			{
				case OptimizerArgs::RegularizationType::L1:
					value -= args.learning_rate * args.regularization_coeff * (value > 0 ? 1 : -1);
					break;
				default:
				case OptimizerArgs::RegularizationType::L2:
					value -= args.learning_rate * args.regularization_coeff * value;
					break;
			}
		}
	};
};