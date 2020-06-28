namespace leetcode
{
    public class Rand
    {
        #region 470. 用 Rand7() 实现 Rand10()

        //https://leetcode-cn.com/problems/implement-rand10-using-rand7/
        int Rand7()
        {
            return -1;
        }

        private int[] nums;
        private int last;

        public Rand()
        {
            nums = new int[10];
            for (int i = 0; i < nums.Length; i++)
            {
                nums[i] = i + 1;
            }

            last = nums.Length - 1;
        }


        public int Rand10()
        {
            var index = Rand7() - 1;
            while (index > last)
            {
                index = Rand7() - 1;
            }

            var res = nums[index];
            var temp = nums[last];
            nums[last] = res;
            nums[index] = temp;
            last--;
            if (last < 0)
            {
                last = nums.Length - 1;
            }

            return res;
        }

        #endregion
    }
}