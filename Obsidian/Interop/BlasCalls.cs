using System.Runtime.InteropServices;
using System.Security;

namespace Obsidian.Interop
{
    internal static unsafe class BlasCalls
    {
        public static int ThreadCount
        {
            get;
            private set;
        } = 1;

        [SuppressUnmanagedCodeSecurity]
        [DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl, EntryPoint = "sgemm_")]
        internal static extern void GeneralMatrixMultiply(sbyte* transa, sbyte* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);

        [SuppressUnmanagedCodeSecurity]
        [DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl, EntryPoint = "openblas_set_num_threads")]
        internal static extern ulong SetNumberOfThreads(int threadCount);

        public static void GeneralMatrixMultiplySafe(sbyte* transa, sbyte* transb, ref int m, ref int n, ref int k, float[] a, float[] b, float[] c)
        {
            int lda = m;
            int ldb = k;
            int ldc = m;
            float alpha = 1f;
            float beta = 0f;

            fixed (int* mPinned = &m)
            fixed (int* nPinned = &n)
            fixed (int* kPinned = &k)
            fixed (float* aPinned = a)
            fixed (float* bPinned = b)
            fixed (float* cPinned = c)
            {
                GeneralMatrixMultiply(transa, transb, mPinned, nPinned, kPinned, &alpha, aPinned, &lda, bPinned, &ldb, &beta, cPinned, &ldc);
            }
        }

        public static void SetNumberOfThreadsSafe(int threadCount)
        {
            ThreadCount = threadCount;
            SetNumberOfThreads(threadCount);
        }
    }
}
