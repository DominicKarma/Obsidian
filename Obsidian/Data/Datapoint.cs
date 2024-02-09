using Obsidian.Mathematics;

namespace Obsidian.Data
{
    /// <summary>
    /// Represents a data point for a learning task, containing an input to supply to a network, and an expected value after processing.
    /// </summary>
    public class Datapoint
    {
        /// <summary>
        /// The input that should be provided to a network.
        /// </summary>
        public Tensor Input;

        /// <summary>
        /// The expected value for the given input. Is checked against the network's outputs as a measure of "incorrect-ness".
        /// </summary>
        public Tensor Expected;

        public Datapoint(float input, float expected)
        {
            Input = new(input);
            Expected = new(expected);
        }

        public Datapoint(Tensor input, Tensor expected)
        {
            Input = input;
            Expected = expected;
        }
    }
}
