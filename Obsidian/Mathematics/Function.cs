namespace Obsidian.Mathematics
{
	[Serializable]
	public readonly struct Function
	{
		private readonly FunctionDelegate func;

		private readonly FunctionDerivativeDelegate? manualDerivative;

		/// <summary>
		/// The amount of inputs this function is expected to use.
		/// </summary>
		public readonly int InputCount;

		/// <summary>
		/// The offset by which inputs are nudged when performing derivatives.
		/// </summary>
		/// <remarks>
		/// The cube root of machine epsilon is optimal for central-difference methods. In this case, for 32-bit floats, the machine epsilon can be calculated as 2^-23.
		/// </remarks>
		public const float HalfDerivativeOffset = 0.00034526f;

		/// <summary>
		/// The inverse of <see cref="HalfDerivativeOffset"/>, divided by two.
		/// </summary>
		public const float InverseDerivativeOffset = 1448.1546f;

		public delegate float FunctionDelegate(params float[] inputs);

		public delegate float FunctionDerivativeDelegate(int term, params float[] inputs);

		public Function(Func<float, float> fx, Func<int, float, float>? fxPrime = null)
		{
			func = inputs => fx(inputs[0]);
			if (fxPrime is not null)
				manualDerivative = (term, inputs) => fxPrime(term, inputs[0]);

			InputCount = 1;
		}

		public Function(Func<float, float, float> fx, Func<int, float, float, float>? fxPrime = null)
		{
			func = inputs => fx(inputs[0], inputs[1]);
			if (fxPrime is not null)
				manualDerivative = (term, inputs) => fxPrime(term, inputs[0], inputs[1]);

			InputCount = 2;
		}

		public Function(Func<float, float, float, float> fx, Func<int, float, float, float, float>? fxPrime = null)
		{
			func = inputs => fx(inputs[0], inputs[1], inputs[2]);
			if (fxPrime is not null)
				manualDerivative = (term, inputs) => fxPrime(term, inputs[0], inputs[1], inputs[2]);

			InputCount = 3;
		}

		public Function(FunctionDelegate fx, int inputCount)
		{
			func = fx;
			InputCount = inputCount;
		}

		/// <summary>
		/// Evaluates the function at the given input(s).
		/// </summary>
		/// <param name="inputs">The input values.</param>
		public float Evaluate(params float[] inputs) => func(inputs);

		/// <summary>
		/// Evaluates the function at all elements in the given tensor. This does not work if the <see cref="InputCount"/> is not 1.
		/// </summary>
		/// <param name="tensor">The input tensor..</param>
		public Tensor Evaluate(Tensor tensor)
		{
			Tensor result = new(tensor.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = func(tensor.data[i]);

			return result;
		}

		/// <summary>
		/// Evaluates the derivative of the function at a given term. All variables other than that at the index of the term are treated as constants.<br></br>
		/// This uses numerical differentiation via the Central Difference Method, with the following limit form:<br></br>
		/// lim h -> 0 (f(x + h) - f(x - h)) / 2h<br></br><br></br>
		/// An exception to this is if a manual derivative function was supplied for this function, in which case that's used instead.
		/// </summary>
		/// <param name="term">The index of differentiation.</param>
		/// <param name="inputs">The input values.</param>
		public float Derivative(int term, params float[] inputs)
		{
			// TODO -- Verify inputs match InputCount.

			// Use the manual derivative, if one was supplied.
			if (manualDerivative is not null)
				return manualDerivative(term, inputs);

			float[] inputsRight = new float[InputCount];
			float[] inputsLeft = new float[InputCount];
			for (int i = 0; i < InputCount; i++)
			{
				inputsRight[i] = inputs[i];
				inputsLeft[i] = inputs[i];

				// If at the term of differentiation, apply the derivative offsets.
				if (i == term)
				{
					inputsRight[i] += HalfDerivativeOffset;
					inputsLeft[i] -= HalfDerivativeOffset;
				}
			}

			// Evaluate the limit at a tiny value for h.
			return (Evaluate(inputsRight) - Evaluate(inputsLeft)) * InverseDerivativeOffset;
		}

		/// <summary>
		/// Evaluates the derivative of the function at a given term. All variables other than that at the index of the term are treated as constants.<br></br>
		/// This performs derivative elementwise, comparing the index of each tensor relative to each other to get the final result.
		/// </summary>
		/// <param name="tensor">The input tensor..</param>
		public Tensor Derivative(int term, params Tensor[] inputTensors)
		{
			// TODO -- Verify same size across matrices.

			Tensor result = new(inputTensors[0].DimensionSizes);
			for (int i = 0; i < result.Length; i++)
			{
				float[] derivativeInputs = new float[InputCount];
				for (int j = 0; j < InputCount; j++)
					derivativeInputs[j] = inputTensors[j].data[i];

				result[i] = Derivative(term, derivativeInputs);
			}

			return result;
		}
	}
}
