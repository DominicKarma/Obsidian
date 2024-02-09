using Obsidian.Auxiliary;
using Obsidian.Mathematics;
using Obsidian.Utilities;

namespace Obsidian.NetworkArchitecture
{
	[Serializable]
	public class DenseLayer : Layer
	{
		public Function ActivationFunction
		{
			get;
			internal set;
		}

		public Tensor Biases
		{
			get;
			internal set;
		}

		public const string InputName = "Input";

		public const string WeightedOutputName = "WeightedOutput";

		public readonly int InputCount;

		public DenseLayer(int inputCount, int neuronCount, Function activationFunction)
		{
			// Store the input and neuron count.
			InputCount = inputCount;
			NeuronCount = neuronCount;

			// Store the activation function.
			ActivationFunction = activationFunction;

			// Generate the weights and biases.
			Weights = new(NeuronCount, InputCount);
			Biases = new(1, NeuronCount);

			// Randomize the weights and biases.
			for (int i = 0; i < NeuronCount; i++)
				Biases[0, i] = RandomUtils.RNG.NextFloatDirection(0.01f);

			for (int i = 0; i < NeuronCount; i++)
			{
				for (int j = 0; j < InputCount; j++)
					Weights[i, j] = RandomUtils.RNG.NextFloatDirection(0.1f);
			}
		}

		public override LayerOutput CalculateOutput(Tensor newInput)
		{
			// Calculate the output as f(W * X + B), saving the function input separately for the
			// backwards step.
			Tensor weightedOutput = Weights * newInput + Biases;
			Tensor outputTensor = ActivationFunction.Evaluate(weightedOutput);
			LayerOutput output = new(outputTensor);

			// Save the input and weighted output.
			output.Add(InputName, newInput);
			output.Add(WeightedOutputName, weightedOutput);

			return output;
		}

		#region Derivations

		// As a sanity check, the chain rule dictates the following for a final dense layer:
		// ∂C/∂w_ij = ∂C/∂a_j * ∂a_j/∂w_ij = 
		// ∂C/∂a_j * ∂a_j/∂z_j * ∂z_j/∂w_ij

		// In this context, "a" refers to the results from the activation function, and "z" refers to the inputs to said input function in the following form:
		// z = w * x + b
		// a = activation(z)
		// Naturally, x refers to the input tensor, w refers to the weights, and b refers to the bias.

		public override Tensor CalculateGradient(Layer aheadLayer, LayerOutput output, Tensor aheadGradient)
		{
			// Use the above definition to calculate ∂a_j/∂z_j = A'(z).
			Tensor weightedOutput = output.Find(WeightedOutputName);
			Tensor activationDerivative = ActivationFunction.Derivative(0, weightedOutput);

			// Use the ahead layer's weights and gradients to calculate the intermediate cost derivatives.
			// TODO -- This is probably not going to work with non-dense layer architectures.
			Tensor gradientDerivative = activationDerivative * Tensor.Transpose(aheadLayer.Weights);
			Tensor gradient = gradientDerivative * aheadGradient;

			return gradient;
		}

		public override Tensor CalculateGradient(Tensor expectedOutput, LayerOutput output, Function costFunction)
		{
			// Use the above definition to calculate ∂C/∂y = C'(y, t).
			Tensor costDerivative = costFunction.Derivative(1, expectedOutput, output);

			// Use the above definition to calculate ∂a_j/∂z_j = A'(z).
			Tensor weightedOutput = output.Find(WeightedOutputName);
			Tensor activationDerivative = ActivationFunction.Derivative(0, weightedOutput);

			Tensor gradient = costDerivative ^ activationDerivative;
			return gradient;
		}

		public override LayerUpdateTensors CalculateUpdateSteps(Tensor gradient, LayerOutput output)
		{
			LayerUpdateTensors results = new();

			// Recall that ∂C/a_j * ∂a_j/∂z_j * ∂z_j/∂w_ij.
			// Since ∂C/∂a_j * ∂a_j/∂z_j is already stored in the auxiliary gradient parameter, all that is necessary is calculating
			// ∂z_j/∂w_ij, which simply comes out to be the inputs.
			Tensor input = output.Find(InputName);
			results.Add("WeightsUpdate", gradient * Tensor.Transpose(input));

			// In the case of biases, the equation becomes the following:
			// ∂C/∂a_j * ∂a_j/∂z_j * ∂z_j/∂b_j
			// ∂z_j/∂b_j turns out to be just one, effectively meaning there's nothing to multiply by.
			results.Add("BiasesUpdate", gradient);

			// Store the old weights and biases.
			results.Add("Weights", Weights);
			results.Add("Biases", Biases);

			return results;
		}

		public override void ApplyUpdateSteps(float learningRate, params LayerUpdateTensors[] updateSteps)
		{
			int totalSamples = updateSteps.Length;
			float updateStep = learningRate / totalSamples;

			// Accumulate the weight and bias step updates together in single tensors.
			// They will be averaged together before the learning rate and other optimizers are used.
			Tensor weightStep = new(Weights.DimensionSizes);
			Tensor biasStep = new(Biases.DimensionSizes);

			for (int i = 0; i < totalSamples; i++)
			{
				weightStep += updateSteps[i].Find("WeightsUpdate");
				biasStep += updateSteps[i].Find("BiasesUpdate");
			}

			// Use the gradient descent rule on the averaged results.
			Weights -= weightStep * updateStep;
			Biases -= biasStep * updateStep;
		}

		#endregion Derivations

		#region Optimizer Utilities

		public override void SupplyHistoryGradientsToOptimizer(Dictionary<string, Tensor> gradients)
		{
			gradients[$"Weights_{LayerIndex}"] = new(Weights.DimensionSizes);
			gradients[$"Biases_{LayerIndex}"] = new(Biases.DimensionSizes);
		}

		#endregion Optimizer Utilities
	}
}
