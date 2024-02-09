using Obsidian.Mathematics;

namespace Obsidian.Auxiliary
{
	public class LayerOutput
	{
		private readonly Dictionary<string, Tensor> extraData = new();

		public Tensor FinalOutput
		{
			get;
			internal set;
		}

		public LayerOutput(Tensor finalOutput) => FinalOutput = finalOutput;

		internal void Add(string name, Tensor value) => extraData[name] = value;

		/// <summary>
		/// Attempts to find and access output data from a given name, returning false if nothing could be found.
		/// </summary>
		/// <param name="name">The data name.</param>
		/// <param name="value">The output value of the data. <see langword="null"/> if nothing could be found.</param>
		internal bool TryFind(string name, out Tensor? value)
		{
			if (extraData.TryGetValue(name, out value))
				return true;

			return false;
		}

		/// <summary>
		/// Attempts to find and access output data from a given name, throwing an error if nothing could be found.
		/// </summary>
		/// <param name="name">The data name.</param>
		/// <param name="value">The output value of the data. <see langword="null"/> if nothing could be found.</param>
		internal Tensor Find(string name)
		{
			// The "value is not null" is definitely irrelevant but I'd like my IDE to stop nagging me about it.
			if (extraData.TryGetValue(name, out Tensor? value) && value is not null)
				return value;

			throw new InvalidOperationException($"Could not found the data from the key '{name}'.");
		}

		/// <summary>
		/// An easy implicit cast that allows for interpreting this output as simply the final output tensor.
		/// </summary>
		/// <param name="output">The layer output.</param>
		public static implicit operator Tensor(LayerOutput output) => output.FinalOutput;
	}
}
