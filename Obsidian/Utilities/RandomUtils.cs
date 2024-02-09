namespace Obsidian.Utilities
{
    public static class RandomUtils
    {
        /// <summary>
        /// A general purpose random number generator.
        /// </summary>
        public static readonly Random RNG = new();

        /// <summary>
        /// Generates a random 0-1 floating point value from a given random number generator.
        /// </summary>
        /// <param name="rng">The random number generator</param>
        public static float NextFloat(this Random rng) => (float)rng.NextDouble();

        /// <summary>
        /// Generates a random floating point value across a desired range from a given random number generator.
        /// </summary>
        /// <param name="rng">The random number generator</param>
        /// <param name="min">The minimum possible value to return.</param>
        /// <param name="max">The maximum possible value to return.</param>
        public static float NextFloat(this Random rng, float min, float max)
        {
            return rng.NextFloat() * (max - min) + min;
        }

        /// <summary>
        /// Generates a random floating point value up to a desired maximum (or equivalent absolute minimum in the negative values) from a given random number generator.
        /// </summary>
        /// <param name="rng">The random number generator</param>
        /// <param name="max">The maximum possible absolute value to return.</param>
        public static float NextFloatDirection(this Random rng, float max)
        {
            return rng.NextFloat(-max, max);
        }
    }
}
