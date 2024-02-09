using Obsidian.Auxiliary;
using Obsidian.Data;
using Obsidian.Mathematics;
using Obsidian.NetworkArchitecture;

namespace Obsidian.Training
{
    /// <summary>
    /// Represents an abstraction for training paradigms for networks in accordance with data sets.
    /// </summary>
    public abstract class NeuralNetworkOptimizer
    {
        public Dataset Data
        {
            get;
            protected set;
        }

        public Function CostFunction
        {
            get;
            protected set;
        }

        public Dictionary<string, Tensor> PreviousGradients
        {
            get;
            private set;
        } = new();

        /// <summary>
        /// Trains a given <see cref="NeuralNetwork"/> with the desired quantity of training cycles.
        /// </summary>
        /// <param name="network">The neural network to train.</param>
        /// <param name="epochs">The amount of training cycles to perform before finishing.</param>
        public abstract void Train(NeuralNetwork network, int epochs);
    }
}
