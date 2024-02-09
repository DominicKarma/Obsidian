using System.Runtime.CompilerServices;
using Obsidian.Interop;

[assembly: InternalsVisibleTo("ObsidianTests")]
namespace Obsidian
{
	public static class Program
	{
		public static void Main()
		{
			BlasCalls.SetNumberOfThreadsSafe(Environment.ProcessorCount);
		}
	}
}
