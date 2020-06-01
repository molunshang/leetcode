using System;
using System.Threading;

namespace leetcode.thread
{
    public class FooBar
    {
        private int n;
        AutoResetEvent foo = new AutoResetEvent(true);
        AutoResetEvent bar = new AutoResetEvent(false);

        public FooBar(int n)
        {
            this.n = n;
        }

        public void Foo(Action printFoo)
        {
            for (int i = 0; i < n; i++)
            {
                // printFoo() outputs "foo". Do not change or remove this line.
                foo.WaitOne();
                printFoo();
                bar.Set();
            }
        }

        public void Bar(Action printBar)
        {
            for (int i = 0; i < n; i++)
            {
                // printBar() outputs "bar". Do not change or remove this line.
                bar.WaitOne();
                printBar();
                foo.Set()
                    ;
            }
        }
    }
}