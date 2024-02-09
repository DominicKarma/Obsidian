using Obsidian.Auxiliary;
using Obsidian.Mathematics;

namespace Obsidian.NetworkArchitecture
{
	[Serializable]
	public abstract class Layer
	{
		/// <summary>
		/// The amount of inputs expected for the layer.
		/// </summary>
		public int LayerIndex
		{
			get;
			protected internal set;
		}

		/// <summary>
		/// The amount of neurons contained in the layer.
		/// </summary>
		public int NeuronCount
		{
			get;
			protected set;
		}

		public Tensor Weights
		{
			get;
			internal set;
		}

		/// <summary>
		/// Calculates the resulting output from a given input relative to the layer's weights and other attributes.
		/// </summary>
		/// <param name="newInput">The input to provide to the layer.</param>
		public abstract LayerOutput CalculateOutput(Tensor newInput);

		/// <summary>
		/// Used to compute the ∂C/∂a_j * ∂a_j/∂z_j gradient for the case where the given layer is <i>not</i> the last one.
		/// </summary>
		/// <param name="aheadLayer">The ahead layer in the network.</param>
		/// <param name="output">The output associated with the update step.</param>
		/// <param name="aheadGradient">The gradient computed from the ahead layer.</param>
		public abstract Tensor CalculateGradient(Layer aheadLayer, LayerOutput output, Tensor aheadGradient);

		/// <summary>
		/// Used to compute the ∂C/∂a_j * ∂a_j/∂z_j gradient for the case where the given layer is the final layer in a network.
		/// </summary>
		/// <param name="expectedOutput">The expected output for the given input sample.</param>
		/// <param name="output">The output that the layer provided at the time of sampling/</param>
		/// <param name="costFunction">The cost function to evaluate the results with. Should expect two inputs: The output and expected value, in that order.</param>
		public abstract Tensor CalculateGradient(Tensor expectedOutput, LayerOutput output, Function costFunction);

		public abstract LayerUpdateTensors CalculateUpdateSteps(Tensor gradient, LayerOutput output);

		public abstract void ApplyUpdateSteps(float learningRate, params LayerUpdateTensors[] updateSteps);

		public abstract void SupplyHistoryGradientsToOptimizer(Dictionary<string, Tensor> gradients);
	}
}
