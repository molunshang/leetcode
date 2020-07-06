using System;

namespace leetcode
{
    #region 384. 打乱数组

    //https://leetcode-cn.com/problems/shuffle-an-array/
    public partial class Solution
    {
        private int[] data;
        private int[] randNums;
        private Random rand = new Random();

        public Solution(int[] nums)
        {
            data = nums;
            randNums = new int[nums.Length];
            Array.Copy(data, randNums, randNums.Length);
        }

        /** Resets the array to its original configuration and return it. */
        public int[] Reset()
        {
            Array.Copy(data, randNums, randNums.Length);
            return randNums;
        }

        /** Returns a random shuffling of the array. */
        public int[] Shuffle()
        {
            for (int i = 0; i < randNums.Length; i++)
            {
                var index = rand.Next(i, randNums.Length);
                var tmp = randNums[index];
                randNums[index] = randNums[i];
                randNums[i] = tmp;
            }

            return randNums;
        }
    }

    #endregion
}