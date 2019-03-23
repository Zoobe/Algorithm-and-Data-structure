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
### LeetCode42. Trapping Rain Water收集雨水

~~~

~~~
