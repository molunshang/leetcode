using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 面试题 16.16. 部分排序

        //https://leetcode-cn.com/problems/sub-sort-lcci/
        public int[] SubSort(int[] array)
        {
            if (array.Length <= 0)
            {
                return new[] {-1, -1};
            }

            //1 5 3 7
            int min = int.MaxValue, max = int.MinValue;
            int left = -1, right = -1;
            for (int i = 0, j = array.Length - 1; i < array.Length; i++, j--)
            {
                if (array[i] < max)
                {
                    right = i;
                }
                else
                {
                    max = array[i];
                }

                if (array[j] > min)
                {
                    left = j;
                }
                else
                {
                    min = array[j];
                }
            }

            return new[] {left, right};
        }

        #endregion

        #region 17. 电话号码的字母组合

        //https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
        public IList<string> LetterCombinations(string digits)
        {
            if (string.IsNullOrEmpty(digits))
            {
                return new string[0];
            }

            var result = new List<string>();
            var dict = new IList<char>[8];
            var chars = new List<char>();
            for (int i = 0, j = 0; i < dict.Length; i++)
            {
                var step = i == 5 || i == 7 ? 4 : 3;
                while (step != 0)
                {
                    chars.Add((char) ('a' + j));
                    j++;
                    step--;
                }

                dict[i] = chars.ToArray();
                chars.Clear();
            }

            void Dfs(int index, List<char> sub)
            {
                if (index >= digits.Length)
                {
                    result.Add(new string(sub.ToArray()));
                    return;
                }

                var ch = digits[index];
                var next = dict[ch - 2];
                foreach (var c in next)
                {
                    sub.Add(c);
                    Dfs(index + 1, sub);
                    sub.RemoveAt(sub.Count - 1);
                }
            }

            Dfs(0, new List<char>());
            return result;
        }

        #endregion

        #region 947. 移除最多的同行或同列石头

        //https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/
        public int RemoveStones(int[][] stones)
        {
            Dictionary<int, int> rows = new Dictionary<int, int>(), cols = new Dictionary<int, int>();
            foreach (var stone in stones)
            {
                int r = stone[0], c = stone[1];
                int size;
                if (rows.TryGetValue(r, out size))
                {
                    rows[r] = size + 1;
                }
                else
                {
                    rows[r] = 1;
                }

                if (cols.TryGetValue(c, out size))
                {
                    cols[c] = size + 1;
                }
                else
                {
                    cols[c] = 1;
                }
            }

            var rm = new bool[stones.Length];

            int Dfs()
            {
                var res = 0;
                for (var i = 0; i < stones.Length; i++)
                {
                    if (rm[i])
                    {
                        continue;
                    }

                    int x = stones[i][0], y = stones[i][1];
                    if (rows[x] <= 1 && cols[y] <= 1)
                    {
                        continue;
                    }

                    rm[i] = true;
                    rows[x]--;
                    cols[y]--;
                    res = Math.Max(res, Dfs() + 1);
                    rows[x]++;
                    cols[y]++;
                    rm[i] = false;
                }

                return res;
            }

            return Dfs();
        }

        //将点连接为图进行DFS
        public int RemoveStonesDfsGraph(int[][] stones)
        {
            var graph = new Dictionary<int, IList<int>>();
            for (var i = 0; i < stones.Length; i++)
            {
                for (int j = i + 1; j < stones.Length; j++)
                {
                    if (stones[i][0] != stones[j][0] && stones[i][1] != stones[j][1])
                    {
                        continue;
                    }

                    if (!graph.TryGetValue(i, out var points))
                    {
                        points = new List<int>();
                        graph[i] = points;
                    }

                    points.Add(j);
                    if (!graph.TryGetValue(j, out points))
                    {
                        points = new List<int>();
                        graph[j] = points;
                    }

                    points.Add(i);
                }
            }

            var stack = new Stack<int>();
            var rm = new bool[stones.Length];
            var res = 0;
            for (int i = 0; i < stones.Length; i++)
            {
                if (rm[i])
                {
                    continue;
                }

                res--;
                stack.Push(i);
                while (stack.TryPop(out var point))
                {
                    if (rm[point])
                    {
                        continue;
                    }

                    rm[point] = true;
                    res++;
                    if (graph.TryGetValue(point, out var next))
                    {
                        foreach (var n in next)
                        {
                            stack.Push(n);
                        }
                    }
                }
            }

            return res;
        }

        #endregion

        #region 822. 翻转卡片游戏

        //https://leetcode-cn.com/problems/card-flipping-game/
        public int Flipgame(int[] fronts, int[] backs)
        {
            var set = new HashSet<int>();
            for (int i = 0; i < fronts.Length; i++)
            {
                if (fronts[i] == backs[i])
                {
                    set.Add(fronts[i]);
                }
            }

            var res = fronts.Concat(backs).Where(n => !set.Contains(n));
            return res.Any() ? res.Min() : 0;
        }

        #endregion

        #region 691. 贴纸拼词

        //https://leetcode-cn.com/problems/stickers-to-spell-word/
        public int MinStickers(string[] stickers, string target)
        {
            var targetDict = target.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            var stickerDict = stickers
                .Select(sticker => sticker.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count()))
                .Where(dict => targetDict.Any(kv => dict.ContainsKey(kv.Key))).ToList();

            var cache = new Dictionary<string, int>();

            int Dfs(Dictionary<char, int> dict, string key)
            {
                if (dict.Count <= 0)
                {
                    return 0;
                }

                if (cache.TryGetValue(key, out var res))
                {
                    return res;
                }

                res = int.MaxValue;
                var nextDict = new Dictionary<char, int>();
                var keyBuiler = new StringBuilder();
                for (var i = 0; i < stickerDict.Count; i++)
                {
                    nextDict.Clear();
                    keyBuiler.Clear();
                    var sticker = stickerDict[i];
                    var skip = true;
                    foreach (var kv in dict)
                    {
                        if (sticker.TryGetValue(kv.Key, out var size))
                        {
                            if (size < kv.Value)
                            {
                                nextDict.Add(kv.Key, kv.Value - size);
                                keyBuiler.Append(kv.Key).Append(',').Append(kv.Value - size);
                            }

                            skip = false;
                        }
                        else
                        {
                            nextDict.Add(kv.Key, kv.Value);
                            keyBuiler.Append(kv.Key).Append(',').Append(kv.Value);
                        }
                    }

                    if (skip)
                    {
                        continue;
                    }

                    var step = Dfs(nextDict, keyBuiler.ToString());
                    if (step != -1)
                    {
                        res = Math.Min(res, step + 1);
                    }
                }

                res = res == int.MaxValue ? -1 : res;
                cache[key] = res;
                return res;
            }

            return Dfs(targetDict, string.Empty);
        }

        #endregion

        #region 332. 重新安排行程

        //https://leetcode-cn.com/problems/reconstruct-itinerary/
        public IList<string> FindItinerary(IList<IList<string>> tickets)
        {
            if (tickets.Count <= 0)
            {
                return new string[0];
            }

            var ticketDict = new Dictionary<string, List<string>>();
            foreach (var ticket in tickets)
            {
                string from = ticket[0], to = ticket[1];
                if (!ticketDict.TryGetValue(from, out var tos))
                {
                    tos = new List<string>();
                    ticketDict[from] = tos;
                }

                tos.Add(to);
            }

            foreach (var list in ticketDict.Values)
            {
                list.Sort();
            }

            var paths = new List<string>();

            bool Dfs(string from)
            {
                if (ticketDict.Count <= 0)
                {
                    paths.Add(from);
                    return true;
                }

                if (!ticketDict.TryGetValue(from, out var tos) || tos.Count <= 0)
                {
                    return false;
                }

                paths.Add(from);
                for (var i = 0; i < tos.Count; i++)
                {
                    var to = tos[i];
                    tos.RemoveAt(i);
                    if (tos.Count <= 0)
                    {
                        ticketDict.Remove(from);
                    }

                    if (Dfs(to))
                    {
                        return true;
                    }

                    if (tos.Count <= 0)
                    {
                        ticketDict[from] = tos;
                    }

                    tos.Insert(i, to);
                }

                paths.RemoveAt(paths.Count - 1);
                return false;
            }

            Dfs("JFK");
            return paths;
        }

        #endregion

        #region 435. 无重叠区间

        //https://leetcode-cn.com/problems/non-overlapping-intervals/
        public int EraseOverlapIntervals(int[][] intervals)
        {
            if (intervals.Length <= 1)
            {
                return 0;
            }

            Array.Sort(intervals, Comparer<int[]>.Create((a, b) => a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]));
            //暴力解，超时
            int Dfs(int prev, int cur)
            {
                if (cur >= intervals.Length)
                {
                    return 0;
                }

                var no = int.MaxValue;
                if (prev < 0 || intervals[prev][1] <= intervals[cur][0])
                {
                    no = Dfs(cur, cur + 1);
                }

                var remove = Dfs(prev, cur + 1) + 1;
                return Math.Min(no, remove);
            }

            return Dfs(-1, 0);
        }

        #endregion
    }
}