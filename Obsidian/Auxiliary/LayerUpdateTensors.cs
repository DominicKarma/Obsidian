using Obsidian.Mathematics;

namespace Obsidian.Auxiliary
{
    public class LayerUpdateTensors
    {
        private readonly Dictionary<string, Tensor> data = new();

        internal void Add(string name, Tensor value) => data[name] = value;

        internal Tensor Find(string name) => data[name];

        internal bool TryFind(string name, out Tensor? value)
        {
            if (data.TryGetValue(name, out value))
                return true;

            return false;
        }
    }
}
