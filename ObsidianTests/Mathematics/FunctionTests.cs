using Microsoft.VisualStudio.TestTools.UnitTesting;
using Obsidian.Mathematics;

namespace ObsidianTests.Mathematics
{
    [TestClass]
    public class FunctionTests
    {
        [TestMethod]
        public void TestDerivative()
        {
            Function func = new(MathF.Sin);
            Assert.AreEqual(func.Derivative(0, 1.94f), MathF.Cos(1.94f), 0.0001f);
        }
    }
}
