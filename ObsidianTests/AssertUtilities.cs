using Microsoft.VisualStudio.TestTools.UnitTesting;
using Obsidian.Mathematics;

namespace ObsidianTests
{
    public static class AssertUtilities
    {
        public static void AreEqual(Tensor a, Tensor b, float delta)
        {
            bool tensorsAreClose = a.Length == b.Length;

            if (tensorsAreClose)
            {
                for (int i = 0; i < a.data.Length; i++)
                {
                    if (MathF.Abs(a.data[i] - b.data[i]) >= delta)
                    {
                        tensorsAreClose = false;
                        break;
                    }
                }
            }

            Assert.IsTrue(tensorsAreClose);
        }
    }
}
