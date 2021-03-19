namespace leetcode
{
    #region 1603. 设计停车系统

    //https://leetcode-cn.com/problems/design-parking-system/
    public class ParkingSystem
    {
        private int[] cars;

        public ParkingSystem(int big, int medium, int small)
        {
            cars = new[] {big, medium, small};
        }

        public bool AddCar(int carType)
        {
            if (cars[carType - 1] <= 0)
            {
                return false;
            }

            cars[carType - 1]--;
            return true;
        }
    }

    #endregion
}