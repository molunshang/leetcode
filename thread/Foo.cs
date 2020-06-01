using System;
using System.Threading;

namespace leetcode.thread
{
    public class Foo
    {
        public Foo()
        {
        }

        private int status = 0;

        public void First(Action printFirst)
        {
            // printFirst() outputs "first". Do not change or remove this line.
            Monitor.Enter(this);
            while (status != 0)
            {
                Monitor.Wait(this);
            }

            printFirst();
            status++;
            Monitor.PulseAll(this);
            Monitor.Exit(this);
        }

        public void Second(Action printSecond)
        {
            // printSecond() outputs "second". Do not change or remove this line.
            Monitor.Enter(this);
            while (status != 1)
            {
                Monitor.Wait(this);
            }

            printSecond();
            status++;
            Monitor.PulseAll(this);
            Monitor.Exit(this);
        }

        public void Third(Action printThird)
        {
            // printThird() outputs "third". Do not change or remove this line.
            Monitor.Enter(this);
            while (status != 2)
            {
                Monitor.Wait(this);
            }

            printThird();
            status++;
            Monitor.PulseAll(this);
            Monitor.Exit(this);
        }
    }
}