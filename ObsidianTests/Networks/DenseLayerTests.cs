using Microsoft.VisualStudio.TestTools.UnitTesting;
using Obsidian.Auxiliary;
using Obsidian.Mathematics;
using Obsidian.NetworkArchitecture;
using Obsidian.Utilities;

namespace ObsidianTests.Networks
{
    [TestClass]
    public class DenseLayerTests
    {
        [TestMethod]
        public void TestWeightedOutputs()
        {
            // Ensure that the underlying logic in DenseLayer is sound. This uses predefined weights and biases and a linear activation to make things as streamlined as possible.
            // The expected output was simply calculated via Symbolab.
            Tensor input = new(new float[2]
            {
                0.24119f,
                0.91683f
            });
            Tensor weights = new(new float[2, 2]
            {
                { 0.31119f, -0.29239f },
                { 0.13604f,  0.04988f },
            });
            Tensor biases = new(new float[2]
            {
                0.158f,
                0.472f
            });
            Tensor expected = new(new float[2]
            {
                -0.035016f,
                0.5505429f
            });
            DenseLayer layer = new(2, 2, new(x => x))
            {
                Weights = weights,
                Biases = biases
            };

            AssertUtilities.AreEqual(layer.CalculateOutput(input), expected, 0.0001f);
        }

        [TestMethod]
        public void TestActivatedOutputs()
        {
            // Same as above, except with a hyperbolic tangent activation function in place.
            // Symbolab was again used for computing the expected values, with the results of course then being altered with the aforementioned activation function.
            Tensor input = new(new float[3]
            {
                -4.38127f,
                1.64118f,
                0.08477f
            });
            Tensor weights = new(new float[3, 3]
            {
                { 0.27178f,  0.49583f, -0.33747f },
                { 0.39587f, -0.41915f, 0.47566f },
                { 0.00494f, -0.11801f, 0.17472f },
            });
            Tensor biases = new(new float[3]
            {
                0f,
                0f,
                0.15744f,
            });
            Tensor expected = new(new float[3]
            {
                -0.384732f,
                -0.983081f,
                -0.043068f,
            });
            DenseLayer layer = new(3, 3, new(MathF.Tanh))
            {
                Weights = weights,
                Biases = biases
            };

            AssertUtilities.AreEqual(layer.CalculateOutput(input), expected, 0.0001f);
        }

        [TestMethod]
        public void TestLayerOutputReferences()
        {
            Function cost = new((output, expected) => MathF.Pow(output - expected, 2f) * 0.5f);
            Function activation = new(MathF.Tanh);
            DenseLayer d = new(1, 1, activation);

            int samples = 8;
            Tensor[] inputs = new Tensor[samples];
            Tensor[] expected = new Tensor[samples];
            LayerOutput[] outputs = new LayerOutput[samples];
            LayerUpdateTensors[] updateSteps = new LayerUpdateTensors[samples];

            Tensor firstOutput = new(1);

            for (int i = 0; i < samples; i++)
            {
                inputs[i] = new(RandomUtils.RNG.NextFloatDirection(6f));
                expected[i] = new(MathF.Tanh(inputs[i][0]));
                outputs[i] = d.CalculateOutput(inputs[i]);
                updateSteps[i] = d.CalculateUpdateSteps(d.CalculateGradient(expected[i], outputs[i], cost), outputs[i]);

                if (i == 0)
                    firstOutput[0] = outputs[0].FinalOutput[0];
            }

            AssertUtilities.AreEqual(firstOutput, outputs[0].FinalOutput, 0.0001f);
        }
    }
}
