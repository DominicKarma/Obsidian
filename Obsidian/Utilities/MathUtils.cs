namespace Obsidian.Utilities
{
    public static class MathUtils
    {
        /// <summary>
        /// Calculates the rectified linear unit of a given input, or max(a * x, x). This serves as a <i>technically</i> non-linear function (The negative slope is different from the positive one) for neural networks.<br></br>
        /// </summary>
        /// <param name="x">The input number.</param>
        /// <param name="negativeFactor">The slope that should be used for negative inputs. Typically either 0 or slightly above 0.</param>
        public static float ReLU(float x, float negativeFactor)
        {
            return MathF.Max(negativeFactor * x, x);
        }
    }
}
