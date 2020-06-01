using System;
using System.Threading;

namespace leetcode.thread
{
    public class ZeroEvenOdd
    {
        private int n;
        AutoResetEvent zero = new AutoResetEvent(true);
        AutoResetEvent even = new AutoResetEvent(false);
        AutoResetEvent odd = new AutoResetEvent(false);

        public ZeroEvenOdd(int n)
        {
            this.n = n;
        }

        // printNumber(x) outputs "x", where x is an integer.
        public void Zero(Action<int> printNumber)
        {
            for (int i = 0; i < n; i++)
            {
                zero.WaitOne();
                printNumber(0);
                if ((i & 1) == 0) //i是偶数，下一步打印奇数
                {
                    odd.Set();
                }
                else
                {
                    even.Set();
                }
            }
        }

        public void Even(Action<int> printNumber)
        {
            for (int i = 2; i <= n; i+=2)
            {
                even.WaitOne();
                printNumber(i);
                zero.Set();
            }            
        }

        public void Odd(Action<int> printNumber)
        {            
            for (int i = 1; i <= n; i += 2)
            {
                odd.WaitOne();
                printNumber(i);
                zero.Set();
            }
        }
    }

}