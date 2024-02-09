using Obsidian.Auxiliary;
using Obsidian.Data;
using Obsidian.Mathematics;
using Obsidian.NetworkArchitecture;

namespace Obsidian.Training
{
    /// <summary>
    /// The most basic optimizer, using regular, unaltered gradient descent to train a neural network.
    /// </summary>
    public class GradientDescentOptimizer : NeuralNetworkOptimizer
    {
        public readonly float LearningRate;

        public GradientDescentOptimizer(float learningRate, Function costFunction, Dataset data)
        {
            LearningRate = learningRate;
            CostFunction = costFunction;
            Data = data;
        }

        /// <summary>
        /// Trains a given <see cref="NeuralNetwork"/> across <i>all</i> points in the data set across each training cycle, ensuring reliable gradient update steps.<br></br>
        /// Generally, this technique is considered too inefficient for practical training requirements.
        /// </summary>
        /// <param name="network">The neural network to train.</param>
        /// <param name="epochs">The amount of training cycles to perform before finishing.</param>
        public override void Train(NeuralNetwork network, int epochs)
        {
            // Store the data set size and layer count in separate local variables.
            int dataSize = Data.Size;
            int layerCount = network.LayerCount;

            for (int i = 0; i < epochs; i++)
            {
                Tensor[] outputs = new Tensor[dataSize * layerCount];
                Tensor[] gradients = new Tensor[dataSize * layerCount];
                LayerUpdateTensors[] updateSteps = new LayerUpdateTensors[dataSize * layerCount];

                // Go across the dataset.
                for (int j = 0; j < dataSize; j++)
                {
                    // Calculate the network's output for a given input sample in each layer.
                    List<LayerOutput> localOutputs = network.CalculateIndividualLayerOutputs(Data[j % dataSize].Input);
                    for (int k = layerCount - 1; k >= 0; k--)
                    {
                        // Store the outputs.
                        int layerSpecificIndex = j * layerCount + k;
                        outputs[layerSpecificIndex] = localOutputs[k];

                        // Calculate gradients and update steps for each layer.
                        Tensor expected = Data.Datapoints[j].Expected;
                        Tensor gradient;
                        if (k == layerCount - 1)
                            gradient = network[k].CalculateGradient(expected, localOutputs[k], CostFunction);
                        else
                            gradient = network[k].CalculateGradient(network[k + 1], localOutputs[k], gradients[layerSpecificIndex + 1]);

                        gradients[layerSpecificIndex] = gradient;
                        updateSteps[layerSpecificIndex] = network[k].CalculateUpdateSteps(gradient, localOutputs[k]);
                    }
                }

                // Update the network.
                for (int j = 0; j < layerCount; j++)
                    network[j].ApplyUpdateSteps(LearningRate, updateSteps.Where((_, index) => index % layerCount == j).ToArray());
            }
        }
    }
}
