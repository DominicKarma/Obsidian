using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Obsidian.Interop
{
    internal static partial class GenericLowLevelCalls
    {
        [LibraryImport("msvcrt.dll", EntryPoint = "memcmp")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(CallConvCdecl) })]
        internal static partial int MemoryCompare(IntPtr a, IntPtr b, uint count);
    }
}
