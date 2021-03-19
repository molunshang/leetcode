using System;
using System.Collections.Generic;

//https://leetcode-cn.com/problems/design-hashset/
//705.设计哈希集合
namespace leetcode
{
    public class MyHashSet
    {

        class Entry
        {
            public int value;
            public Entry next;
        }

        private Entry[] entries;

        /** Initialize your data structure here. */
        public MyHashSet()
        {
            entries = new Entry[32];
        }

        public void Add(int key)
        {
            var index = key % entries.Length;
            var entry = entries[index];
            if (entry == null)
            {
                entries[index] = new Entry() { value = key };
            }
            else if (entry.value != key)
            {
                while (entry.next != null)
                {
                    entry = entry.next;
                    if (entry.value == key)
                    {
                        return;
                    }
                }
                entry.next = new Entry() { value = key };
            }
        }

        public void Remove(int key)
        {
            var index = key % entries.Length;
            var entry = entries[index];
            Entry prev = null;
            while (entry != null)
            {
                if (entry.value == key)
                {
                    if (prev == null)
                    {
                        entries[index] = entry.next;
                        entry.next = null;
                    }
                    else
                    {
                        prev.next = entry.next;
                    }
                    break;
                }
                prev = entry;
                entry = entry.next;
            }
        }

        /** Returns true if this set contains the specified element */
        public bool Contains(int key)
        {
            var entry = entries[key % entries.Length];
            while (entry != null)
            {
                if (entry.value == key)
                {
                    return true;
                }
                entry = entry.next;
            }
            return false;
        }
    }
}