using System.Collections.Immutable;

namespace Obsidian.Data
{
    /// <summary>
    /// Represents a collection of many <see cref="Datapoint"/>s for the purposes of being used as a basis for training.
    /// </summary>
    public class Dataset
    {
        /// <summary>
        /// The collection of data points.
        /// </summary>
        public List<Datapoint> Datapoints = new();

        /// <summary>
        /// The amount of data points contained in the data set.
        /// </summary>
        public int Size => Datapoints.Count;

        /// <summary>
        /// Accesses a <see cref="Datapoint"/> at a desired index.
        /// </summary>
        /// <param name="index">The index accessor.</param>
        public Datapoint this[int index]
        {
            get => Datapoints[index];
        }

        /// <summary>
        /// Adds a new <see cref="Datapoint"/> to the data set.
        /// </summary>
        /// <param name="point">The datapoint to add.</param>
        public void Add(Datapoint point) => Datapoints.Add(point);
    }
}
