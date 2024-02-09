using System.Diagnostics;
using Obsidian.Interop;

namespace Obsidian.Mathematics
{
	public partial class Tensor
	{
		/// <summary>
		/// Accesses or sets the tensor at a given input, with arbitrarily dimensional inputs.
		/// </summary>
		/// <param name="indices"></param>
		public float this[params int[] indices]
		{
			[DebuggerStepThrough]
			get => data[DimensionIndex(indices)];

			[DebuggerStepThrough]
			set => data[DimensionIndex(indices)] = value;
		}

		/// <summary>
		/// Efficiently multiplies two matrices via BLAS calls. This only works if the tensors supplied are of rank 2, and as such are assumed to be matrices,
		/// </summary>
		/// <param name="a">The first matrix, represented as a tensor.</param>
		/// <param name="b">The second matrix, represented as a tensor.</param>
		public static unsafe Tensor MatrixMultiply(Tensor a, Tensor b)
		{
			int resultWidth = a.DimensionSize(0);
			int aHeight = a.TotalDimensions <= 1 ? 1 : a.DimensionSize(1);
			int resultHeight = b.TotalDimensions <= 1 ? 1 : b.DimensionSize(1);
			Tensor result = new(resultWidth, resultHeight);

			sbyte nta = 110; // Equivalent to the n char value.
			BlasCalls.GeneralMatrixMultiplySafe(&nta, &nta, ref resultWidth, ref resultHeight, ref aHeight, a.data, b.data, result.data);

			return result;
		}

		/// <summary>
		/// Transposes a matrix. This only works if the tensor supplied is of rank 2, and as such are assumed to be matrices,
		/// </summary>
		/// <param name="a">The first matrix, represented as a tensor.</param>
		/// <param name="b">The second matrix, represented as a tensor.</param>
		public static unsafe Tensor Transpose(Tensor a)
		{
			int height = a.DimensionSize(0);
			int width = a.TotalDimensions >= 2 ? 1 : a.DimensionSize(1);

			if (width == 1 && height == 1)
				return a;

			Tensor result = new(width, height);
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
					result[i, j] = a[j, i];
			}

			return result;
		}

		/// <summary>
		/// Performs scalar addition to the given tensor.
		/// </summary>
		/// <param name="a">The tensor to add to.</param>
		/// <param name="b">The scalar.</param>
		[DebuggerStepThrough]
		public static Tensor operator +(Tensor a, float b)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = a[i] + b;

			return result;
		}

		/// <summary>
		/// Performs element-wise addition to two given tensors.
		/// </summary>
		/// <param name="a">The first tensor.</param>
		/// <param name="b">The second tensor.</param>
		[DebuggerStepThrough]
		public static Tensor operator +(Tensor a, Tensor b)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = a[i] + b[i];

			return result;
		}

		/// <summary>
		/// Negates a tensor's elements.
		/// </summary>
		/// <param name="a">The tensor to negate.</param>
		[DebuggerStepThrough]
		public static Tensor operator -(Tensor a)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = -a[i];

			return result;
		}

		/// <summary>
		/// Performs scalar subtraction to a given tensor.
		/// </summary>
		/// <param name="a">The tensor to subtract from.</param>
		/// <param name="b">The scalar.</param>
		[DebuggerStepThrough]
		public static Tensor operator -(Tensor a, float b) => a + -b;

		/// <summary>
		/// Performs element-wise subtraction to two given tensors.
		/// </summary>
		/// <param name="a">The first tensor.</param>
		/// <param name="b">The second tensor.</param>
		[DebuggerStepThrough]
		public static Tensor operator -(Tensor a, Tensor b)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = a[i] - b[i];

			return result;
		}

		/// <summary>
		/// Performs a generalized multiplication to two tensors.<br></br>
		/// This is <b>NOT</b> the Hadamard product, and this operation is not commutative (meaning that the order in which the multiplication occurs matters).
		/// </summary>
		/// <param name="a">The first tensor.</param>
		/// <param name="b">The second tensor.</param>
		public static Tensor operator *(Tensor a, Tensor b)
		{
			if (a.Length == 1 && b.Length == 1)
				return new(a.data[0] * b.data[0]);

			if (a.TotalDimensions <= 2 && b.TotalDimensions <= 2)
				return MatrixMultiply(a, b);

			return new(new int[] { 1 });
		}

		/// <summary>
		/// Performs a Hadamard Product (or element-wise multiplication to two tensors.
		/// </summary>
		/// <param name="a">The first tensor.</param>
		/// <param name="b">The second tensor.</param>
		[DebuggerStepThrough]
		public static Tensor operator ^(Tensor a, Tensor b)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = a[i] * b[i];

			return result;
		}

		/// <summary>
		/// Performs scalar multiplication to a given tensor, scaling its elements.
		/// </summary>
		/// <param name="a">The tensor to multiply to.</param>
		/// <param name="b">The scalar.</param>
		[DebuggerStepThrough]
		public static Tensor operator *(Tensor a, float b)
		{
			Tensor result = new(a.DimensionSizes);
			for (int i = 0; i < result.Length; i++)
				result[i] = a[i] * b;

			return result;
		}

		/// <summary>
		/// Efficiently determines whether two tensors are equivalent based on their elements.
		/// </summary>
		/// <param name="a">The first tensor to check against.</param>
		/// <param name="b">The second tensor to check against.</param>
		public static bool operator ==(Tensor a, Tensor b)
		{
			return a.Equals(b);
		}

		/// <summary>
		/// Efficiently determines whether two tensors are NOT equivalent based on their elements.
		/// </summary>
		/// <param name="a">The first tensor to check against.</param>
		/// <param name="b">The second tensor to check against.</param>
		public static bool operator !=(Tensor a, Tensor b)
		{
			return !a.Equals(b);
		}

		public override int GetHashCode() => data.GetHashCode();

		public override unsafe bool Equals(object? obj)
		{
			if (obj is not Tensor other || other.Length != Length)
				return false;

			fixed (float* p1 = data, p2 = other.data)
				return GenericLowLevelCalls.MemoryCompare((IntPtr)p1, (IntPtr)p2, lengthBytes) == 0;
		}
	}
}
