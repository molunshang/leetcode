

//303. 区域和检索 - 数组不可变
//https://leetcode-cn.com/problems/range-sum-query-immutable/
namespace leetcode
{
    public class NumArray
    {
        private readonly int[] prefixs;
        private readonly int[] nums;

        public NumArray(int[] nums)
        {
            if (nums == null || nums.Length <= 0)
            {
                return;
            }

            this.nums = nums;
            prefixs = new int[nums.Length];
            prefixs[0] = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                prefixs[i] = prefixs[i - 1] + nums[i];
            }
        }

        public int SumRange(int i, int j)
        {
            if (nums == null || j >= nums.Length)
            {
                return 0;
            }

            if (i == 0)
            {
                return prefixs[j];
            }

            return prefixs[j] - prefixs[i] + nums[i];
        }
    }
}