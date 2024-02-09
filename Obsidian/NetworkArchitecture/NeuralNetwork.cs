using System.Runtime.Serialization.Formatters.Binary;
using Obsidian.Auxiliary;
using Obsidian.Mathematics;

namespace Obsidian.NetworkArchitecture
{
	/// <summary>
	/// An abstraction for a learning unit composed of various <see cref="Layer"/>s.
	/// </summary>
	[Serializable]
	public class NeuralNetwork
	{
		/// <summary>
		/// The sequential collection of all layers in the network.
		/// </summary>
		protected List<Layer> layers = new();

		/// <summary>
		/// The last layer in the network.
		/// </summary>
		public Layer LastLayer => layers.Last();

		/// <summary>
		/// The amount of layers the network has.
		/// </summary>
		public int LayerCount => layers.Count;

		/// <summary>
		/// The base amount of inputs that a new layer should have. Equivalent to <see cref="LastLayer"/>.NeuronCount.<br></br>
		/// Useful for defining the layout of new layers that are to be added to the network.
		/// </summary>
		public int OutputCount => LastLayer.NeuronCount;

		/// <summary>
		/// Adds a new layer at the end of the network.
		/// </summary>
		/// <param name="layer">The new layer to add.</param>
		public void Add(Layer layer)
		{
			layer.LayerIndex = layers.Count;
			layers.Add(layer);
		}

		/// <summary>
		/// Returns the layer at a given index in the network.
		/// </summary>
		/// <param name="index">The desired layer index.</param>
		public Layer this[int index]
		{
			get => layers[index];
		}

		/// <summary>
		/// Calculates the resulting output from a given input relative to the network's weights and other attributes, passing it forward through each layer.
		/// </summary>
		/// <param name="newInput">The input to provide to the network.</param>
		public Tensor CalculateOutput(Tensor newInput)
		{
			// Propagate the output through the layers, with each successive layer taking the previous layer's outputs as inputs.
			Tensor output = layers[0].CalculateOutput(newInput);
			for (int i = 1; i < layers.Count; i++)
				output = layers[i].CalculateOutput(output);

			return output;
		}

		/// <summary>
		/// Calculates the resulting outputs from a given input relative to the network's weights and other attributes, passing it forward through each layer.<br></br>
		/// Unlike <see cref="CalculateOutput(Tensor)"/>, this provides the output of each layer in a sequential list.<br></br>
		/// This method is primarily used during the training process. For input sampling it is more sensible to use <see cref="CalculateOutput(Tensor)"/>.
		/// </summary>
		public List<LayerOutput> CalculateIndividualLayerOutputs(Tensor newInput)
		{
			List<LayerOutput> outputs = new();

			// Propagate the output through the layers, with each successive layer taking the previous layer's outputs as inputs.
			LayerOutput output = layers[0].CalculateOutput(newInput);
			outputs.Add(output);

			for (int i = 1; i < layers.Count; i++)
			{
				output = layers[i].CalculateOutput(outputs.Last());
				outputs.Add(output);
			}

			return outputs;
		}

		/// <summary>
		/// Saves the contents of a neural network to a file via serialization.
		/// </summary>
		/// <param name="path">The file path.</param>
		public void SaveToFile(string path)
		{
#pragma warning disable SYSLIB0011 // Type or member is obsolete
			using FileStream stream = new(path, FileMode.Create);

			BinaryFormatter formatter = new();
			formatter.Serialize(stream, this);
		}

		public static NeuralNetwork FromFile(string path)
		{
			using FileStream stream = new(path, FileMode.Open);

			BinaryFormatter formatter = new();
			return (NeuralNetwork)formatter.Deserialize(stream);
#pragma warning restore SYSLIB0011 // Type or member is obsolete
		}
	}
}
