using Microsoft.VisualStudio.TestTools.UnitTesting;
using Obsidian.Mathematics;

namespace ObsidianTests.Mathematics
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void TestIndexerGetter()
        {
            Tensor tensor = new(new float[3, 3]
            {
                { 0f, 1f, 2f },
                { 3f, 4f, 5f },
                { 6f, 7f, 8f }
            });
            Assert.AreEqual(tensor[1, 1], 4f);
        }

        [TestMethod]
        public void TestIndexerSetter()
        {
            Tensor tensor = new(new float[2, 2]
            {
                { 0f, 1f },
                { 2f, 3f }
            });
            tensor[0, 1] = 4f;
            Assert.AreEqual(tensor[0, 1], 4f);
        }

        [TestMethod]
        public void TestMultiply()
        {
            Tensor a = new(new float[3, 3]
            {
                { 1f, 2f, 5f },
                { 6f, 1f, 3f },
                { 4f, 9f, 2f },
            });
            Tensor b = new(new float[3, 3]
            {
                { 2f, 0f, 8f },
                { 6f, 1f, 0f },
                { 7f, 4f, 2f },
            });
            Tensor expected = new(new float[3, 3]
            {
                { 49f, 22f, 18f },
                { 39f, 13f, 54f },
                { 76f, 17f, 36f },
            });
            Assert.AreEqual(a * b, expected);
        }

        [TestMethod]
        public void TestMultiplyDifferentSizes()
        {
            Tensor a = new(new float[3, 1]
            {
                { 1f },
                { 3f },
                { 2f },
            });
            Tensor b = new(new float[1, 3]
            {
                { 1f, 1f, 5f }
            });
            Tensor expected = new(new float[3, 3]
            {
                { 1f, 1f, 5f },
                { 3f, 3f, 15f },
                { 2f, 2f, 10f },
            });
            Assert.AreEqual(a * b, expected);
        }

        [TestMethod]
        public void TestEquality()
        {
            Tensor a = new(new float[4, 4]
            {
                { 01f, 02f, 03f, 04f },
                { 05f, 06f, 07f, 08f },
                { 09f, 10f, 11f, 12f },
                { 13f, 14f, 15f, 16f }
            });
            Tensor b = new(new float[4, 4]
            {
                { 01f, 02f, 03f, 04f },
                { 05f, 06f, 07f, 08f },
                { 09f, 10f, 11f, 12f },
                { 13f, 14f, 15f, 16f }
            });
            Assert.IsTrue(a == b);
        }
    }
}
