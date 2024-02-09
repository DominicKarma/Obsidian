using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Obsidian.Mathematics
{
    [DebuggerDisplay("{this.ToString()}")]
    [Serializable]
    public partial class Tensor
    {
        internal readonly float[] data;

        internal readonly uint lengthBytes;

        internal readonly int[] inputOffsets;

        public readonly int[] DimensionSizes;

        /// <summary>
        /// The overall length of the tensor, computed as the product of all the dimension sizes.<br></br>
        /// Internally, this represents the length of the underlying flattened array.
        /// </summary>
        public readonly int Length;

        /// <summary>
        /// The amount of dimensions the tensor has. Also commonly known as the rank.
        /// </summary>
        public readonly int TotalDimensions;

        public Tensor(params int[] sizes)
        {
            // Calculate the amount of dimensions this tensor has based on the amount of dimension values supplied in the sizes value.
            TotalDimensions = sizes.Length;
            DimensionSizes = sizes;

            // Initialize the input offsets array.
            inputOffsets = new int[TotalDimensions];

            // Get the overall length of the tensor but taking the product of each of the dimension sizes.
            Length = 1;
            for (int i = 0; i < TotalDimensions; i++)
            {
                inputOffsets[i] = Length;
                Length *= sizes[i];
            }

            // Initialize the underlying data array.
            data = new float[Length];

            // Cache the length of the tensor in bytes. This is used for efficiency with equality checks, which make use of low level byte-indexed memory operations.
            lengthBytes = (byte)Length * 4u;
        }

        public Tensor(float scalar) : this(1, 1)
        {
            data = new float[1]
            {
                scalar
            };
        }

        public Tensor(float[] data) : this(1, data.Length)
        {
            this.data = data;
        }

        public Tensor(float[,] data) : this(data.GetLength(0), data.GetLength(1))
        {
            int length = data.GetLength(0);
            int width = data.GetLength(1);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < width; j++)
                    this[i, j] = data[i, j];
            }
        }

        /// <summary>
        /// Gets a given flattened dimension index from an arbitrary array of dimension indices.
        /// </summary>
        /// <param name="indices">The dimension indices.</param>
        protected int DimensionIndex(params int[] indices)
        {
            // Assume a tensor of size [2, 2, 2]. Its index configuration is as follows:
            // 01, 02    05, 06
            // 03, 04    07, 08

            // For each dimension, the index offset is offset based on the overall multiplier.
            // Said multipliers are cached in the inputOffsets array.
            int index = 0;
            for (int i = 0; i < indices.Length; i++)
                index += indices[i] * inputOffsets[i];

            return index;
        }

        /// <summary>
        /// Gets the dimension size (such as the width or the height) of the given dimension index.
        /// </summary>
        /// <param name="dimension"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int DimensionSize(int dimension) => DimensionSizes[dimension];

        /// <summary>
        /// Returns a simplified <see cref="string"/> representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            StringBuilder sb = new();
            for (int i = 0; i < Length; i++)
            {
                bool startOfDimension = DimensionSizes.Any(s => i % s == 0);
                bool endOfDimension = DimensionSizes.Any(s => i % s == s - 1);
                bool newLine = DimensionSizes.Skip(2).Any(s => i % s == s - 1);

                if (startOfDimension)
                    sb.Append('[');
                sb.Append($" {data[i]} ");
                if (endOfDimension)
                    sb.Append("]\n");
                if (newLine)
                    sb.Append('\n');
            }

            return sb.ToString();
        }
    }
}
