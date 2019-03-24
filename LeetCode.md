### LeetCode1. Two Sum
~~~
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++)
        {
            if(map.containsKey(target-nums[i]))
            {
                res[1] = i;
                res[0] = map.get(target-nums[i]);
            }
            map.put(nums[i],i);
        }
        return res;
    }
~~~
### LeetCode15. 3Sum
找到和为`0`的所有三个数
~~~
public List<List<Integer>> threeSum(int[] num) {
    Arrays.sort(num);
    List<List<Integer>> res = new LinkedList<>(); 
    for (int i = 0; i < num.length-2; i++) {
        if (i == 0 || (i > 0 && num[i] != num[i-1])) {
            int lo = i+1, hi = num.length-1, sum = 0 - num[i];
            while (lo < hi) {
                if (num[lo] + num[hi] == sum) {
                    res.add(Arrays.asList(num[i], num[lo], num[hi]));
                    while (lo < hi && num[lo] == num[lo+1]) lo++;
                    while (lo < hi && num[hi] == num[hi-1]) hi--;
                    lo++; hi--;
                } else if (num[lo] + num[hi] < sum) lo++;
                else hi--;
           }
        }
    }
    return res;
}
~~~
### LeetCode16. 3Sum Closest
输入一组数组和`target`，找到距离`target`最近的三个数
题解：和上题类似，先随便确定一个sum值，每次设置三个位置，绝对值较小就更新结果
~~~
public class Solution {
    public int threeSumClosest(int[] num, int target) {
        int result = num[0] + num[1] + num[num.length - 1];
        Arrays.sort(num);
        for (int i = 0; i < num.length - 2; i++) {
            int start = i + 1, end = num.length - 1;
            while (start < end) {
                int sum = num[i] + num[start] + num[end];
                if (sum > target) {
                    end--;
                } else {
                    start++;
                }
                if (Math.abs(sum - target) < Math.abs(result - target)) {
                    result = sum;
                }
            }
        }
        return result;
    }
}
~~~
### LeetCode18. 4Sum
将题目转化为3Sum问题
~~~
public class Solution {
public List<List<Integer>> fourSum(int[] num, int target) {
    ArrayList<List<Integer>> ans = new ArrayList<>();
    if(num.length<4)return ans;
    Arrays.sort(num);
    for(int i=0; i<num.length-3; i++){
        if(num[i]+num[i+1]+num[i+2]+num[i+3]>target)break; //如果前四个数太大就可以直接结束
        if(num[i]+num[num.length-1]+num[num.length-2]+num[num.length-3]<target)continue; //第一个数太小也可以直接结束
        if(i>0&&num[i]==num[i-1])continue; //去除重复结果
        for(int j=i+1; j<num.length-2; j++){
            if(num[i]+num[j]+num[j+1]+num[j+2]>target)break; //第二个数太大
            if(num[i]+num[j]+num[num.length-1]+num[num.length-2]<target)continue; //第二个参与者太小
            if(j>i+1&&num[j]==num[j-1])continue; //prevents duplicate results in ans list
            int low=j+1, high=num.length-1;
            while(low<high){
                int sum=num[i]+num[j]+num[low]+num[high];
                if(sum==target){
                    ans.add(Arrays.asList(num[i], num[j], num[low], num[high]));
                    while(low<high&&num[low]==num[low+1])low++; //跳过左边重复的
                    while(low<high&&num[high]==num[high-1])high--; //跳过右边重复的
                    low++; 
                    high--;
                }
                //move window
                else if(sum<target)low++; 
                else high--;
            }
        }
    }
    return ans;
}
~~~
### LeetCode2. Add Two Numbers
~~~
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode c1 = l1;
        ListNode c2 = l2;
        ListNode sentinel = new ListNode(0);
        ListNode d = sentinel;
        int sum = 0;
        while (c1 != null || c2 != null) {
            sum /= 10;
            if (c1 != null) {
                sum += c1.val;
                c1 = c1.next;
            }
            if (c2 != null) {
                sum += c2.val;
                c2 = c2.next;
            }
            d.next = new ListNode(sum % 10);
            d = d.next;
        }
        if (sum / 10 == 1)
            d.next = new ListNode(1);
        return sentinel.next;
    }
~~~
### LeetCode3. Longest Substring Without Repeating Characters
~~~
    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        if(len == 0 || len == 1) return len;
        char[] str = s.toCharArray();
        int l = 0, r = -1;
        int res = 0;
        boolean[] flag = new boolean[256];
        while(l<len)
        {
            if(r<len-1 && flag[str[r+1]]==false)
            {
                r++;
                flag[str[r]] = true;
            }
            else
            {
                flag[str[l++]] = false;
                
            }
            res = Math.max(res,r-l+1);
        }
        return res;
    }
~~~
### LeetCode4. Median of Two Sorted Arrays
假设两个有序数组的长度分别为m和n，由于两个数组长度之和 m+n 的奇偶不确定，因此需要分情况来讨论，对于奇数的情况，直接找到最中间的数即可，偶数的话需要求最中间两个数的平均值。为了简化代码，不分情况讨论，
我们使用一个小trick，我们分别找第 `(m+n+1) / 2 `个，和 `(m+n+2) / 2 `个，然后求其平均值即可，这对奇偶数均适用。加入 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身。
其实要对K二分，意思是我们需要分别在nums1和nums2中查找第K/2个元素，注意这里由于两个数组的长度不定，所以有可能某个数组没有第K/2个数字，所以我们需要先check一下，数组中到底存不存在第K/2个数字，
如果存在就取出来，否则就赋值上一个整型最大值.</br>
如果某个数组没有第K/2个数字，那么我们就淘汰另一个数组的前K/2个数字即可。举个例子来说吧，比如 nums1 = {3}，nums2 = {2, 4, 5, 6, 7}，K=4，我们要找两个数组混合中第4个数字，那么我们分别在 nums1 和 nums2 中找第2个数字，我们发现 nums1 中只有一个数字，不存在第二个数字，那么 nums2 中的前2个数字可以直接跳过，为啥呢，因为我们要求整个混合数组的第4个数字，不管 nums1 中的那个数字是大是小，第4个数字绝不会出现在 nums2 的前两个数字中，所以可以直接跳过。
有没有可能两个数组都不存在第K/2个数字呢，这道题里是不可能的，因为我们的K不是任意给的，而是给的m+n的中间值，所以必定至少会有一个数组是存在第K/2个数字的。最后就是二分法的核心啦，比较这两个数组的第K/2小的数字midVal1和midVal2的大小，如果第一个数组的第K/2个数字小的话，那么说明我们要找的数字肯定不在nums1中的前K/2个数字，所以我们可以将其淘汰，将nums1的起始位置向后移动K/2个，并且此时的K也自减去K/2，调用递归。反之，我们淘汰nums2中的前K/2个数字，并将nums2的起始位置向后移动K/2个，并且此时的K也自减去K/2，调用递归即可，
~~~
    public double findMedianSortedArrays(int[] A, int[] B) {
          int m = A.length, n = B.length;
          int l = (m + n + 1) / 2;
          int r = (m + n + 2) / 2;
          return (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0;
      }

    public double getkth(int[] A, int aStart, int[] B, int bStart, int k) {
      if (aStart > A.length - 1) return B[bStart + k - 1];            
      if (bStart > B.length - 1) return A[aStart + k - 1];                
      if (k == 1) return Math.min(A[aStart], B[bStart]);

      int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
      if (aStart + k/2 - 1 < A.length) aMid = A[aStart + k/2 - 1]; 
      if (bStart + k/2 - 1 < B.length) bMid = B[bStart + k/2 - 1];        

      if (aMid < bMid) 
          return getkth(A, aStart + k/2, B, bStart, k - k/2);// Check: aRight + bLeft 
      else 
          return getkth(A, aStart, B, bStart + k/2, k - k/2);// Check: bRight + aLeft
    }
~~~
### LeetCode5. Longest Palindromic Substring最长回文子串
从0遍历到`s.length()-1` </br>
分为奇数和偶数两种情况扩展
~~~
public class Solution {
private int lo, maxLen;

public String longestPalindrome(String s) {
	int len = s.length();
	if (len < 2)
		return s;
	
    for (int i = 0; i < len-1; i++) {
     	extendPalindrome(s, i, i);  //如果是奇数长度，扩展回文数
     	extendPalindrome(s, i, i+1); //如果是偶数长度，扩展回文数
    }
    return s.substring(lo, lo + maxLen);
}

private void extendPalindrome(String s, int j, int k) {
	while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
		j--;
		k++;
	}
	if (maxLen < k - j - 1) {
		lo = j + 1;
		maxLen = k - j - 1;
	}
}}
~~~
### LeetCode6. ZigZag Conversion
假设字符串按照Z字排列
如： </br>
给定`"PAYPALISHIRING"` 和行数`3` </br>
~~~
P   A   H   N
A P L S I I G
Y   I   R
~~~
读时候按照横着读"PAHNAPLSIIGYIR"
解答：每行创建一个StringBuilder，将原始字符串的字符加入相对应的字符串中
~~~
public String convert(String s, int nRows) {
    char[] c = s.toCharArray();
    int len = c.length;
    StringBuffer[] sb = new StringBuffer[nRows];
    for (int i = 0; i < sb.length; i++) sb[i] = new StringBuffer();
    
    int i = 0;
    while (i < len) {
        for (int idx = 0; idx < nRows && i < len; idx++) // 竖直向下
            sb[idx].append(c[i++]);
        for (int idx = nRows-2; idx >= 1 && i < len; idx--) // 右斜向上
            sb[idx].append(c[i++]);
    }
    // 最后从上向下将StringBuilder整合到一起
    for (int idx = 1; idx < sb.length; idx++)
        sb[0].append(sb[idx]);
    return sb[0].toString();
}
~~~
### LeetCode7. Reverse Integer
给定一个32位有符号`Integer`，将其反转为一个整数
~~~
Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
~~~

~~~
    public int reverse(int x) {
        int res = 0,tail = 0;
        while(x != 0)
        {
            tail = x % 10;
            if(Math.abs(res)>Integer.MAX_VALUE/10) return 0;  // 逆运算验证，如果不等说明溢出了
            res = res*10 + tail;
            x /=10;
        }
        return res;
    }
~~~
### LeetCode9. Palindrome Number回文数字
不使用字符串解决
~~~
Example 1:

Input: 121
Output: true
Example 2:

Input: -121
Output: false
~~~
~~~
    public boolean isPalindrome(int x) {
        if(x<0) return false;
        int res = 0;
        int copy = x;
        int m = 10;
        while(x != 0)
        {
            res *= 10;
            res += x%m;
            x /= m;
        }
        if(res != copy) return false;
        return true;
    }
~~~
### LeetCode10. Regular Expression Matching正则表达式匹配
给定字符串`s`和`p` </br>
`s`中只有`a-z`的小写字符 </br>
`p`除了`a-z`的字符外，还有`. `和`* `</br>
`'.'` 匹配任意单个字符 </br>
`'*'` 匹配零个或者之前任意一个字符 </br>
~~~
public boolean isMatch(String s, String p) {

    if (s == null || p == null) {
        return false;
    }
    boolean[][] dp = new boolean[s.length()+1][p.length()+1];
    dp[0][0] = true;
    for (int i = 0; i < p.length(); i++) {
        if (p.charAt(i) == '*' && dp[0][i-1]) {
            dp[0][i+1] = true;
        }
    }
    for (int i = 0 ; i < s.length(); i++) {
        for (int j = 0; j < p.length(); j++) {
            if (p.charAt(j) == '.') {
                dp[i+1][j+1] = dp[i][j];
            }
            if (p.charAt(j) == s.charAt(i)) {
                dp[i+1][j+1] = dp[i][j];
            }
            if (p.charAt(j) == '*') {
                if (p.charAt(j-1) != s.charAt(i) && p.charAt(j-1) != '.') {
                    dp[i+1][j+1] = dp[i+1][j-1];
                } else {
                    dp[i+1][j+1] = (dp[i+1][j] || dp[i][j+1] || dp[i+1][j-1]);
                }
            }
        }
    }
    return dp[s.length()][p.length()];
}
~~~
### LeetCode44. Wildcard Matching
给定字符串`s`和`p` </br>
`s`中只有`a-z`的小写字符 </br>
`p`除了`a-z`的字符外，还有`?` 和`*` </br>
`'?'` 匹配任意单个字符 </br>
`'*'` 匹配任意连续字符，包括空字符 </br>
~~~
public class Solution {
    public boolean isMatch(String s, String p) {
        boolean[][] match=new boolean[s.length()+1][p.length()+1];
        match[s.length()][p.length()]=true;
        for(int i=p.length()-1;i>=0;i--){
            if(p.charAt(i)!='*')
                break;
            else
                match[s.length()][i]=true;
        }
        for(int i=s.length()-1;i>=0;i--){
            for(int j=p.length()-1;j>=0;j--){
                if(s.charAt(i)==p.charAt(j)||p.charAt(j)=='?')
                        match[i][j]=match[i+1][j+1];
                else if(p.charAt(j)=='*')
                        match[i][j]=match[i+1][j]||match[i][j+1];
                else
                    match[i][j]=false;
            }
        }
        return match[0][0];
    }
}
~~~
### LeetCode72. Edit Distance
~~~
Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation:
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
~~~
题目描述：修改一个字符串成为另一个字符串，使得修改次数最少。一次修改操作包括：插入一个字符、删除一个字符、替换一个字符。
~~~
public int minDistance(String word1, String word2) {
    if (word1 == null || word2 == null) {
        return 0;
    }
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
        dp[i][0] = i;
    }
    for (int i = 1; i <= n; i++) {
        dp[0][i] = i;
    }
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
            }
        }
    }
    return dp[m][n];
}
~~~
### LeetCode11. Container With Most Water
~~~
Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49
~~~
可以用一点点贪心去解这道题，一步步缩小子数组的大小。</br>
左边低就缩左边，右边低就缩右边
~~~
public int maxArea(int[] height) {
    int left = 0, right = height.length - 1;
	int maxArea = 0;

	while (left < right) {
		maxArea = Math.max(maxArea, Math.min(height[left], height[right])
				* (right - left));
		if (height[left] < height[right])
			left++;
		else
			right--;
	}

	return maxArea;
}
~~~
### LeetCode135. Candy
规则1：每个人手中至少一个 </br>
规则2：评分更高的孩子比左右邻居糖果数量多：分别遍历比较比左侧孩子多，比右侧孩子多。
~~~
Example 1:

Input: [1,0,2]
Output: 5
Explanation:可以给第一，第二，第三个孩子分别分配糖果为 2, 1, 2 

Example 2:

Input: [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
             The third child gets 1 candy because it satisfies the above two conditions.
~~~
使用贪心算法，从左到右遍历确保高分孩子比左侧糖果多，从右向左遍历确保高分孩子比右侧糖果多。</br>
算法步骤 </br>
- 每个孩子一个糖果
- 从左到右遍历，ratings[i+1]>ratings[i]时，candies[i+1]=candies[i] +1
- 从右到左遍历，ratings[i-1]>ratings[i]时, candies[i-1]=max(candies[i-1], candies[i]+1)
~~~
public int candy(int[] ratings) {
    int candies[] = new int[ratings.length];        
    Arrays.fill(candies, 1);// Give each child 1 candy 
    	
    for (int i = 1; i < candies.length; i++){// 确保右边分数更高的孩子比左边分数低的孩子糖果多1
        if (ratings[i] > ratings[i - 1]) candies[i] = (candies[i - 1] + 1);
    }
     
    for (int i = candies.length - 2; i >= 0; i--) {// 确保左边分数更高的孩子比右边分数低的孩子糖果多1
	    if (ratings[i] > ratings[i + 1]) candies[i] = Math.max(candies[i], (candies[i + 1] + 1));
    }
    
    int sum = 0;        
    for (int candy : candies)  
    	sum += candy;        
    return sum;
}
~~~
### LeetCode84. Largest Rectangle in Histogram
给定n个整数代表直方图的高度，每个直方图宽度默认为1，找到直方图中的最大矩形
题解：这道题可以用单调栈来求解，单调栈又两种：
 - 递增栈：递增栈是维护递增的顺序，当遇到小于栈顶元素的数就开始处理
 - 递减栈：递减栈正好相反，维护递减的顺序，当遇到大于栈顶元素的数开始处理。
   那么根据这道题的特点，我们需要按从高板子到低板子的顺序处理，先处理最高的板子，宽度为1，然后再处理旁边矮一些的板子，此时长度为2，因为之前的高板子可组成矮板子的矩形 ，`因此我们需要一个递增栈`，`当遇到大的数字直接进栈，而当遇到小于栈顶元素的数字时，就要取出栈顶元素进行处理了`，那取出的顺序就是从高板子到矮板子了，于是乎遇到的较小的数字只是一个触发，表示现在需要开始计算矩形面积了 </br>
   为了使得最后一块板子也被处理，这里用了个小trick，在高度数组最后面加上一个0，这样原先的最后一个板子也可以被处理了。</br>
   单调栈中不能放高度，而是需要放坐标。由于我们先取出栈中最高的板子，那么就可以先算出长度为1的矩形面积了，然后再取下一个板子，此时根据矮板子的高度算长度为2的矩形面积，以此类推，知道数字大于栈顶元素为止，再次进栈，巧妙的一比！
~~~
public class Solution {
    public int largestRectangleArea(int[] height) {
        int len = height.length;
        Stack<Integer> s = new Stack<Integer>();
        int maxArea = 0;
        for(int i = 0; i <= len; i++){
            int h = (i == len ? 0 : height[i]);
            if(s.isEmpty() || h >= height[s.peek()]){
                s.push(i);
            }else{
                int tp = s.pop();
                maxArea = Math.max(maxArea, height[tp] * (s.isEmpty() ? i : i - 1 - s.peek()));
                i--;
            }
        }
        return maxArea;
    }
}
~~~
### LeetCode85. Maximal Rectangle
给定由`0`，`1`组成的二维矩阵，找到最大的矩形并求得其面积
  此题是之前那道的 Largest Rectangle in Histogram 直方图中最大的矩形 的扩展，这道题的二维矩阵每一层向上都可以看做一个直方图，输入矩阵有多少行，就可以形成多少个直方图，对每个直方图都调用 Largest Rectangle in Histogram 直方图中最大的矩形 中的方法，就可以得到最大的矩形面积。那么这道题唯一要做的就是将每一层构成直方图，由于题目限定了输入矩阵的字符只有 '0' 和 '1' 两种，所以处理起来也相对简单。方法是，对于每一个点，如果是‘0’，则赋0，如果是 ‘1’，就赋 之前的height值加上1。
  而且可以边构建height数组边计算最大面积
~~~
    public int maximalRectangle(char[][] matrix) {
        if(matrix==null||matrix.length==0||matrix[0]==null||matrix[0].length==0) return 0;
        int row = matrix.length;
        int col = matrix[0].length;
        int max = 0;
        int[] heights = new int[col+1];
        for(int i=0;i<row;i++){
            Stack<Integer> stack = new Stack<>();
            for(int j=0;j<col+1;j++){
                if(j<col){
                    if(matrix[i][j]=='1')
                        heights[j]++;
                    else heights[j] = 0;
                }
                
                if(stack.isEmpty()||heights[stack.peek()]<=heights[j]){
                    stack.push(j);
                }else{
                    while(!stack.isEmpty() && heights[stack.peek()]>heights[j]){
                        int h = heights[stack.pop()];
                        int w = stack.isEmpty()?j:(j-stack.peek()-1);
                        max = Math.max(max,h*w);
                    }
                    stack.push(j);
                }

            }
        }
        return max;
    }
~~~
