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
### LeetCode26. Remove Duplicates from Sorted Array
~~~
    public int removeDuplicates(int[] nums) {
        if(nums==null) return 0;
        int len = nums.length;
        int count = 0;
        for(int i=0;i<len;i++)
        {
            if(nums[i] != nums[count])
                nums[++count] = nums[i];
        }
        return ++count;
    }
~~~
### LeetCode27. Remove Element
在数组中移除指定元素
~~~
public int removeElement(int[] A, int elem) {
   int m = 0;    
   for(int i = 0; i < A.length; i++){
       
       if(A[i] != elem){
           A[m] = A[i];
           m++;
       }
   }
   
   return m;
}
~~~
### LeetCode28. Implement strStr()实现indexOf函数
在字符串中第一次出现的位置，没有的话返回-1
此题可以练习KMP算法
~~~
	// next数组计算
	private void getNext(String pattern, int next[]) {
		int j = 0;
		int k = -1;
		int len = pattern.length();
		next[0] = -1;
 
		while (j < len - 1) {
			if (k == -1 || pattern.charAt(k) == pattern.charAt(j)) {
 
				j++;
				k++;
				next[j] = k;
			} else {
 
				// 比较到第K个字符，说明p[0——k-1]字符串和p[j-k——j-1]字符串相等，而next[k]表示
				// p[0——k-1]的前缀和后缀的最长共有长度，所以接下来可以直接比较p[next[k]]和p[j]
				k = next[k];
			}
		}
 
	}
 
	int kmp(String s, String pattern) {
		int i = 0;
		int j = 0;
		int slen = s.length();
		int plen = pattern.length();
 
		int[] next = new int[plen];
 
		getNext(pattern, next);
 
		while (i < slen && j < plen) {
 
			// 字符匹配，查找下一个字符
			if (s.charAt(i) == pattern.charAt(j)) {
				i++;
				j++;
			} else {
				// 如果为-1说明pattern串要重头匹配
				if (next[j] == -1) {
					i++;
					j = 0;
				} else {
					j = next[j];
				}
 
			}
			// 说明找到字符串了，返回所在位置
			if (j == plen) {
				return i - j;
			}
		}
		return -1;
	}
~~~
### LeetCode30. Substring with Concatenation of All Words
给定一个字符串和一组词，词的长度都一样，找到S子串包含，将word数组中的所有组合，每个词只能用一次
~~~
Example 1:

Input:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoor" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
Example 2:

Input:
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
Output: []
~~~
~~~
public static List<Integer> findSubstring(String S, String[] L) {
    List<Integer> res = new ArrayList<Integer>();
    if (S == null || L == null || L.length == 0) return res;
    int len = L[0].length(); // length of each word
    
    Map<String, Integer> map = new HashMap<String, Integer>(); // map for L
    for (String w : L) map.put(w, map.containsKey(w) ? map.get(w) + 1 : 1);
    
    for (int i = 0; i <= S.length() - len * L.length; i++) {
        Map<String, Integer> copy = new HashMap<String, Integer>(map);
        for (int j = 0; j < L.length; j++) { // checkc if match
            String str = S.substring(i + j*len, i + j*len + len); // next word
            if (copy.containsKey(str)) { // is in remaining words
                int count = copy.get(str);
                if (count == 1) copy.remove(str);
                else copy.put(str, count - 1);
                if (copy.isEmpty()) { // matches
                    res.add(i);
                    break;
                }
            } else break; // not in L
        }
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
### LeetCode32. Longest Valid Parentheses
给定一个链表，和一个值x，切分该链表，使得小于x以及本来就在x前面的值都在x前面
~~~
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
~~~
解法：基本思想是建立两个队列，一个存储小于x的，一个保存剩余元素，最后要将第二个队列置为0，防止死循环
~~~
public ListNode partition(ListNode head, int x) {
    ListNode dummy1 = new ListNode(0), dummy2 = new ListNode(0);  //dummy heads of the 1st and 2nd queues
    ListNode curr1 = dummy1, curr2 = dummy2;      //current tails of the two queues;
    while (head!=null){
        if (head.val<x) {
            curr1.next = head;
            curr1 = head;
        }else {
            curr2.next = head;
            curr2 = head;
        }
        head = head.next;
    }
    curr2.next = null;          //important! avoid cycle in linked list. otherwise u will get TLE.
    curr1.next = dummy2.next;
    return dummy1.next;
}
~~~
### LeetCode21. Merge Two Sorted Lists归并链表
~~~
public ListNode mergeTwoLists(ListNode l1, ListNode l2){
		if(l1 == null) return l2;
		if(l2 == null) return l1;
		if(l1.val < l2.val){
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else{
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}
}
~~~
### LeetCode23. Merge k Sorted Lists归并K个有序的链表
~~~
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
~~~
题解：使用优先队列，先将每个链表的头部放入优先队列，之后每次取出一个同时更新链表
~~~
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists==null || lists.length==0) return null;
        
        PriorityQueue<ListNode> queue= new PriorityQueue<ListNode>(lists.length, (a,b)-> a.val-b.val);
        
        ListNode dummy = new ListNode(0);
        ListNode tail=dummy;
        
        for (ListNode node:lists)
            if (node!=null)
                queue.add(node);
            
        while (!queue.isEmpty()){
            tail.next=queue.poll();
            tail=tail.next;
            
            if (tail.next!=null)
                queue.add(tail.next);
        }
        return dummy.next;
    }
~~~
### LeetCode24. Swap Nodes in Pairs
给定链表，交换相邻两个链表元素
~~~
Given 1->2->3->4, you should return the list as 2->1->4->3.
~~~
题解：因为返回的是交换之后的头部，可以用递归
~~~
    public ListNode swapPairs(ListNode head) {
        if(head==null || head.next==null) return head;
        ListNode l1 = head.next;
        head.next = swapPairs(head.next.next);
        l1.next = head;
        return l1;
    }
~~~
### LeetCode25. Reverse Nodes in k-Group交换相邻的K个链表元素
给定一个链表，每次交换K个元素，返回调整后的结果
~~~
Given this linked list: 1->2->3->4->5

For k = 2, you should return: 2->1->4->3->5

For k = 3, you should return: 3->2->1->4->5
~~~
题解：
~~~
public ListNode reverseKGroup(ListNode head, int k) {
    ListNode curr = head;
    int count = 0;
    while (curr != null && count != k) { // 找到第k+1个节点
        curr = curr.next;
        count++;
    }
    if (count == k) { // 如果k+1节点存在 
        curr = reverseKGroup(curr, k); // reverse list with k+1 node as head
        // head - 原始头结点
        // curr - 反转之后的头结点
        while (count-- > 0) { // 反转前面的K个节点
            ListNode tmp = head.next; // tmp - next head in direct part
            head.next = curr; // preappending "direct" head to the reversed list 
            curr = head; // move head of reversed part to a new node
            head = tmp; // move "direct" head to the next node in direct part
        }
        head = curr;
    }
    return head;
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
### LeetCode42. Trapping Rain Water收集雨水
解法：单调栈
  我们对低洼的地方感兴趣，就可以使用一个单调递减栈，将递减的边界存进去，一旦发现当前的数字大于栈顶元素了，那么就有可能会有能装水的地方产生。此时我们当前的数字是右边界，我们从栈中至少需要有两个数字，才能形成一个坑槽，先取出的那个最小的数字，就是坑槽的最低点，再次取出的数字就是左边界，我们比较左右边界，取其中较小的值为装水的边界，然后此高度减去水槽最低点的高度，乘以左右边界间的距离就是装水量了。由于需要知道左右边界的位置，所以我们虽然维护的是递减栈，但是栈中数字并不是存递减的高度，而是递减的高度的坐标。
~~~
  public int trap(int[] height) {
      Stack<Integer> s = new Stack<Integer>();
      int i = 0, n = height.length, res = 0;
      while (i < n) {
          if (s.isEmpty() || height[i] <= height[s.peek()]) {
              s.push(i++);
          } else {
              int t = s.pop();
              if (s.isEmpty()) continue;
              res += (Math.min(height[i], height[s.peek()]) - height[t]) * (i - s.peek() - 1);
          }
      }
      return res;
  }
~~~
### LeetCode14. Longest Common Prefix
找出最长公共前缀串，如果没有就输出`""`
~~~
    public String longestCommonPrefix(String[] strs) {
        StringBuilder result = new StringBuilder();
        
        if (strs!= null && strs.length > 0){
        
            Arrays.sort(strs);
            
            char [] a = strs[0].toCharArray();
            char [] b = strs[strs.length-1].toCharArray();
            
            for (int i = 0; i < a.length; i ++){
                if (b.length > i && b[i] == a[i]){
                    result.append(b[i]);
                }
                else {
                    return result.toString();
                }
            }
        return result.toString();
    }
~~~
### LeetCode146. LRU Cache
~~~
Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
~~~
~~~
class LRUCache {
    
    class Node{
        int key;
        int value;
        Node pre;
        Node next;
        public Node(int key,int value){
            this.key = key;
            this.value = value;
        }
    }
    private HashMap<Integer,Node> map;
    private Node head,tail;
    private int capacity;
    private int count;

    public LRUCache(int capacity) {
        this.map = new HashMap();
        this.capacity = capacity;
        head = new Node(0,0);
        tail = new Node(0,0);
        head.next = tail;
        head.next.pre = head;
        head.pre = null;
        tail.next = null;
    }
    
    private void insertTohead(Node node){
        Node second = this.head.next;
        head.next = node;
        node.pre = head;
        second.pre = node;
        node.next = second;
    }
    
    private void removeFromtail(){
        Node temp = this.tail.pre;
        temp.pre.next = this.tail;
        temp.pre = null;
        this.tail.pre = temp;
    }
    
    private void removeNode(Node node){
        Node pre = node.pre;
        Node next = node.next;
        node.next = null;
        node.pre = null;
        pre.next = next;
        next.pre = pre;
    }
    
    public int get(int key) {
        if(map.containsKey(key)){
            Node node = map.get(key);
            int value = node.value;
            removeNode(node);
            insertTohead(node);
            return value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node node = map.get(key);
            removeNode(node);
            node.value = value;
            node.key = key;
            insertTohead(node);
            map.put(key,node);
        }else{
            Node node = new Node(key,value);
            map.put(key,node);
            if(count<this.capacity){
                count++;
                insertTohead(node);
                
            }
            else{
                map.remove(this.tail.pre.key);
                removeNode(this.tail.pre);
                insertTohead(node);
            }
        }
    }
}
~~~
### LeetCode19. Remove Nth Node From End of List移除倒数第N个链表节点
使用双指针
~~~
public ListNode removeNthFromEnd(ListNode head, int n) {
    
    ListNode start = new ListNode(0);
    ListNode slow = start, fast = start;
    slow.next = head;
    
    //Move fast in front so that the gap between slow and fast becomes n
    for(int i=1; i<=n+1; i++)   {
        fast = fast.next;
    }
    //Move fast to the end, maintaining the gap
    while(fast != null) {
        slow = slow.next;
        fast = fast.next;
    }
    //Skip the desired node
    slow.next = slow.next.next;
    return start.next;
}
~~~
### LeetCode20. Valid Parentheses有效的括号
~~~
public boolean isValid(String s) {
	Stack<Character> stack = new Stack<Character>();
	for (char c : s.toCharArray()) {
		if (c == '(')
			stack.push(')');
		else if (c == '{')
			stack.push('}');
		else if (c == '[')
			stack.push(']');
		else if (stack.isEmpty() || stack.pop() != c)
			return false;
	}
	return stack.isEmpty();
}
~~~
### LeetCode22. Generate Parentheses
给定n对括号，写一个函数生成所有有效括号组合
比如给定n=3，结果为
~~~
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
~~~
~~~
public List<String> generateParenthesis(int n) {
    List<String> list = new ArrayList<String>();
    generateOneByOne("", list, n, n);
    return list;
}
public void generateOneByOne(String sublist, List<String> list, int left, int right){
    if(left > right){ //因为每次都是先增加左括号，所以要求left一定比right小
        return;
    }
    if(left > 0){
        generateOneByOne( sublist + "(" , list, left-1, right);
    }
    if(right > 0){
        generateOneByOne( sublist + ")" , list, left, right-1);
    }
    if(left == 0 && right == 0){
        list.add(sublist);
        return;
    }
}
~~~
### LeetCode32. Longest Valid Parentheses
给定一个字符串只包含`(`和`)`，找出最长的有效子串
解法：使用单调栈的思想
~~~
    public int longestValidParentheses(String s) {
        int len = s.length();
        if(s.length()==0) return 0;
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for(int i=0;i<len;i++)
        {
            if(s.charAt(i)==')'&& !stack.empty() && s.charAt(stack.peek())=='(')
            {
                stack.pop();
                if(stack.empty()) res = i+1;
                else res = Math.max(res,i-stack.peek());
            }
            else
                stack.push(i);
        }
        return res;
    }
~~~
### LeetCode131. Palindrome Partitioning
给定字符串，分割字符串使得所有部分都为回文串，返回所有结果
~~~
Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
~~~
~~~
    public List<List<String>> partition(String s) {
        List<List<String>> res = new LinkedList<>();
        if(s==null||s.length()==0) return res;
        List<String> part = new LinkedList<>();
        search(s,0,res,part);
        return res;
        
    }
    
    private void search(String s,int index,List<List<String>> res,List<String> part){
        if(index==s.length()){
            res.add(new LinkedList(part));
            return;
        }
        for(int i=index+1;i<=s.length();i++){
            String str = s.substring(index,i);
            if(isValid(str)){
                part.add(str);
                search(s,i,res,part);
                part.remove(part.size()-1);
            }
        }
        return;
    }
    
    private boolean isValid(String str){
        char[] ch = str.toCharArray();
        int left = 0;
        int right = ch.length-1;
        while(left<=right){
            if(ch[left]!=ch[right]){
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
~~~
### LeetCode132. Palindrome Partitioning II
给定字符串，找出最小分割次数，使得所有部分都为回文串
~~~
Input: "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
~~~
解法：
这里需要两个DP数组，cut数组记录最小分割次数，pal数组记录在[j,i]之间是否是回文串
DP1 如果[j,i]是回文串，只需要cut[i] = cut[j - 1] + 1 (j <= i)
DP2 如果[j,i]是回文串，[j + 1, i - 1]也是回文串，同时c[j] == c[i]
~~~
a   b   a   |   c  c
                j  i
       j-1  |  [j, i] is palindrome
   cut(j-1) +  1
~~~

~~~
public int minCut(String s) {
     boolean[][] isPalindr = new boolean[n + 1][n + 1]; //isPalindr[i][j] = true means s[i:j) is a valid palindrome
     int[] dp = new int[n + 1]; //dp[i] means the minCut for s[0:i) to be partitioned 

     for(int i = 0; i <= n; i++) dp[i] = i - 1;//initialize the value for each dp state.
     
     for(int i = 2; i <= n; i++){
         for(int j = i - 1; j >= 0; j--){//i从前往后遍历，j从后往前遍历
             //if(isPalindr[j][i]){
             if(s.charAt(i - 1) == s.charAt(j) && (i - 1 - j < 2 || isPalindr[j + 1][i - 1])){
                 isPalindr[j][i] = true;
                 dp[i] = Math.min(dp[i], dp[j] + 1);
             }
         }
     }
     
     return dp[n];
}
~~~
### LeetCode337. House Robber III
小偷偷的是按照二叉树来偷，但是不会连着偷父子，只会偷爷爷和孙子，或者兄弟节点
~~~
Example 1:

Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
Example 2:

Input: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
~~~
题解：如果左孩子存在，计算出左孩子的左子节点的最大值，还有左孩子右子节点的最大值</br>
如果右孩子存在，计算出右孩子的左子节点的最大值，还有右孩子右子节点的最大值</br>
比较当前节点加上孙子节点和不算当前节点(只算当前节点的左右子节点的和)的最大值</br>
为了防止溢出超时，采用HashMap保存当前节点的最大值
~~~
public class Solution {
    public int rob(TreeNode root) {
        if(root==null) return 0;
        if(root.left==null&&root.right==null) return root.val;
        
        int left=0, right=0;
        int subleft=0, subright=0;
    
    if(root.left!=null){
        left=rob(root.left);
        subleft=rob(root.left.left)+rob(root.left.right);
    }
    
    if(root.right!=null){
        right=rob(root.right);
        subright=rob(root.right.left)+rob(root.right.right);
    }
    
    int sum1=left+right;
    int sum2=subleft+subright+root.val;
    
    return (sum1>sum2)?sum1:sum2;
}
~~~
### LeetCode172. Factorial Trailing Zeroes
找出阶乘N！结果中的0的个数
题解：让求一个数的阶乘末尾0的个数，也就是要找乘数中10的个数，而10可分解为2和5，而我们可知2的数量又远大于5的数量，那么此题即便为找出5的个数。仍需注意的一点就是，像25,125，这样的不只含有一个5的数字需要考虑进去。
~~~
    public int trailingZeroes(int n) {
        int res = 0;
        while (n > 0) {
            res += n / 5;
            n /= 5;
        }
        return res;
    }
~~~
### LeetCode235. Lowest Common Ancestor of a Binary Search Tree二叉搜索树的最低公共祖先
题解：根据二叉搜索树的性质</br>
如果根节点的值大于p和q之间的较大值，说明p和q都在左子树中，那么此时我们就进入根节点的左子节点继续递归</br>
如果根节点小于p和q之间的较小值，说明p和q都在右子树中，那么此时我们就进入根节点的右子节点继续递归</br>
如果都不是，则说明当前根节点就是最小共同父节点，直接返回即可
~~~
public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root.val > p.val && root.val > q.val){
            return lowestCommonAncestor(root.left, p, q);
        }else if(root.val < p.val && root.val < q.val){
            return lowestCommonAncestor(root.right, p, q);
        }else{
            return root;
        }
    }
}
~~~
### LeetCode235. Lowest Common Ancestor of a Binary Search Tree搜索普通二叉树的最低公共祖先
图：
~~~
Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
Example 2:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
~~~
题解：
1. 首先关注递归返回条件
   在递归函数中，我们首先看当前结点是否为空，若为空则直接返回空，若为p或q中的任意一个，也直接返回当前结点。
2. 否则的话就对其左右子结点分别调用递归函数，由于这道题限制了p和q一定都在二叉树中存在，那么如果当前结点不等于p或q，p和q要么分别位于左右子树中，要么同时位于左子树，或者同时位于右子树，那么我们分别来讨论：
- 若p和q要么分别位于左右子树中，那么对左右子结点调用递归函数，会分别返回p和q结点的位置，而当前结点正好就是p和q的最小共同父结点，直接返回当前结点即可，这就是题目中的例子1的情况。
- 若p和q同时位于左子树，这里有两种情况，一种情况是left会返回p和q中较高的那个位置，而right会返回空，所以我们最终返回非空的left即可，这就是题目中的例子2的情况。还有一种情况是会返回p和q的最小父结点，就是说当前结点的左子树中的某个结点才是p和q的最小父结点，会被返回。
- 若p和q同时位于右子树，同样这里有两种情况，一种情况是right会返回p和q中较高的那个位置，而left会返回空，所以我们最终返回非空的right即可，还有一种情况是会返回p和q的最小父结点，就是说当前结点的右子树中的某个结点才是p和q的最小父结点，会被返回
~~~
public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q)  return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null)   return root;
        return left != null ? left : right;
    }
}
~~~
### LeetCode136. Single Number
只有一个数出现了一次，其他都出现两次
~~~
public int singleNumber(int[] nums) {
    int ans =0;
    
    int len = nums.length;
    for(int i=0;i!=len;i++)
        ans ^= nums[i];
    
    return ans;
    
}
~~~
### LeetCode137. Single Number II
只有一个数出现了一次，其他都出现了三次
题解：我们可以建立一个32位的数字，来统计每一位上1出现的个数，我们知道如果某一位上为1的话，那么如果该整数出现了三次，对3去余为0，我们把每个数的对应位都加起来对3取余，最终剩下来的那个数就是单独的数字。
~~~
public int singleNumber(int[] nums) {
    int ans = 0;
    for(int i = 0; i < 32; i++) {
        int sum = 0;
        for(int j = 0; j < nums.length; j++) {
            if(((nums[j] >> i) & 1) == 1) {
                sum++;
                sum %= 3;
            }
        }
        if(sum != 0) {
            ans |= sum << i;
        }
    }
    return ans;
}
~~~
### LeetCode260. Single Number III
有两个数出现了一次，其他数都出现两次
题解：能想办法把原数组分为两个小数组，不相同的两个数字分别在两个小数组中，这样分别调用 Single Number 的解法就可以得到答案。</br>
首先我们先把原数组全部异或起来，那么我们会得到一个数字，这个数字是两个不相同的数字异或的结果，我们取出其中任意一位为‘1’的位，为了方便起见，我们用 a &= -a 来取出最右端为‘1’的位,通过这一位分为两组，组内亦或
~~~
public class Solution {
    public int[] singleNumber(int[] nums) {
        int diff = 0;
        for (int num : nums) {
            diff ^= num;
        }
        // 获得最右一位的1
        diff &= -diff;
        
        // Pass 2 :
        int[] rets = {0, 0}; // 这个数组用来存储最后的结果
        for (int num : nums)
        {
            if ((num & diff) == 0) // 如果这个数该位不为1
            {
                rets[0] ^= num;
            }
            else // the bit is set
            {
                rets[1] ^= num;
            }
        }
        return rets;
    }
}
~~~
### LeetCode217. Contains Duplicate整型数组是否有重复
~~~
    public boolean containsDuplicate(int[] nums) {
        if(nums.length<=1) return false;
        Arrays.sort(nums);
        for(int i=1;i<nums.length;i++){
            if(nums[i]==nums[i-1]) return true;
        }
        return false;
    }
~~~
### LeetCode207. Course Schedule课程调度
**重点：**还是有向图环的检测
先建立好有向图，然后从第一个门课开始，找其可构成哪门课，暂时将当前课程标记为已访问，然后对新得到的课程调用DFS递归，直到出现新的课程已经访问过了，则返回false，没有冲突的话返回true，然后把标记为已访问的课程改为未访问。
~~~
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if(numCourses<=0) return false;
        ArrayList<Integer>[] graph = new ArrayList[numCourses];
        for(int i=0;i<graph.length;i++){
            graph[i] = new ArrayList<Integer>();
        }
        // int[] degree = new int[numCourses];
        for(int i=0;i<prerequisites.length;i++){
            // degree[prerequisites[i][1]]++;
            graph[prerequisites[i][1]].add(prerequisites[i][0]);
        }
        boolean[] visit = new boolean[numCourses];
        for(int i=0;i<numCourses;i++){
            if(!dfs(graph,visit,i))
                return false;
        }
        return true;
    }
    
    private boolean dfs(ArrayList<Integer>[] graph,boolean[] visit,int course ){
        if(visit[course]) return false;
        else visit[course] = true;
        for(int i=0;i<graph[course].size();i++){
            if(!dfs(graph,visit,graph[course].get(i)))
                return false;
        }
        
        visit[course] = false;
        return true;
    }
~~~
### LeetCode210. Course Schedule II
题解：这道题还需要计算所有课程的入度，从入读为0的课程开始遍历，将课程放入queue中，依次将该课程的下级课程入度减一，如果该课程入度为0，就添加进queue
最终若有向图中有环，则结果中元素的个数不等于总课程数，那我们将结果清空即可
~~~
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        ArrayList<Integer> res = new ArrayList<>();
        int[] indegree = new int[numCourses];
        ArrayList<Integer>[] graph = new ArrayList[numCourses];
        for(int i=0;i<numCourses;i++){
            graph[i] = new ArrayList<Integer>();
        }
        for(int[] depend:prerequisites){
            indegree[depend[0]]++;	//计算入度
            graph[depend[1]].add(depend[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for(int i=0;i<indegree.length;i++){
            if(indegree[i]==0) queue.offer(i);
        }
        
        while(!queue.isEmpty()){
            int course = queue.poll();
            res.add(course);
            for(int i=0;i<graph[course].size();i++){
                if(--indegree[graph[course].get(i)]==0){	//如果入度为0，就加入queue
                    queue.offer(graph[course].get(i));
                }
            }
        }
        
        int[] ans = new int[numCourses];
        if(res.size()!=numCourses) return new int[0];//如果最后结果内的课程数目不等于课程总个数，说明有环
        for(int i=0;i<res.size();i++){
            ans[i] = res.get(i);
        }
        return ans;
    }
~~~
### 编程之美3.8：求二叉树中节点的最大距离
二叉树中节点的最大距离必定是两个叶子节点的距离。求某个子树的节点的最大距离，有三种情况：
1. 两个叶子节点都出现在左子树；
2. 两个叶子节点都出现在右子树；
3. 一个叶子节点在左子树，一个叶子节点在右子树。
只要求得三种情况的最大值，结果就是这个子树的节点的最大距离。
~~~
private int height(TreeNode root){
	if(root==null) return 0;
	return Math.max(height(root.left),height(root.right))+1;
}

public int findMaxDis(TreeNode root){
	if(root==null) return 0;
	int leftMax = findMaxDis(root.left);
	int rightMax = findMax(root.right);
	return Math.max(Math.max(leftMax,rightMax),height(root.left)+height(root.right));
}
~~~
### LeetCode124. Binary Tree Maximum Path Sum
给定一棵树，找到最大的节点路径和，路径是指一条直线，不能分叉，不一定是从根节点或者叶子节点开始，可以是任意节点
~~~
Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42
~~~
题解：的递归函数返回值就可以定义为以当前结点为根结点，到叶节点的最大路径之和，然后全局路径最大值放在参数中，用结果res来表示。</br>
在递归函数中，如果当前结点不存在，那么直接返回0。否则就分别对其左右子节点调用递归函数，`由于路径和有可能为负数`，而我们当然不希望加上负的路径和，所以我们和0相比，取较大的那个，`就是要么不加，加就要加正数`。然后我们来更新全局最大值结果res，就是`以左子结点为终点的最大path之和加上以右子结点为终点的最大path之和，还要加上当前结点值，这样就组成了一个条完整的路径`。而我们返回值是取left和right中的较大值加上当前结点值，因为我们返回值的定义是以当前结点为终点的path之和，所以只能取left和right中较大的那个值，而不是两个都要
~~~
    private int max;
    
    public int maxPathSum(TreeNode root) {
        max = Integer.MIN_VALUE;
        helper(root);
        return max;
    }
    private int helper(TreeNode root)
    {
        if(root==null) return 0;
        int left = Math.max(0,maxPathSum(root.left));
        int right = Math.max(0,maxPathSum(root.right));
        max = Math.max(max,left+right+root.val);
        return Math.max(left+root.val,right+root.val);
    }
~~~
### LeetCode300. Longest Increasing Subsequence
动态规划Dynamic Programming的解法，这种解法的时间复杂度为O(n2)，我们维护一个一维dp数组，其中dp[i]表示以nums[i]为结尾的最长递增子串的长度，对于每一个nums[i]，我们从第一个数再搜索到i，如果发现某个数小于nums[i]，我们更新dp[i]，更新方法为dp[i] = max(dp[i], dp[j] + 1)，即比较当前dp[i]的值和那个小于num[i]的数的dp值加1的大小，我们就这样不断的更新dp数组，到最后dp数组中最大的值就是我们要返回的LIS的长度</br>
下面我们来看一种优化时间复杂度到O(nlgn)的解法，这里用到了二分查找法，所以才能加快运行时间。思路是，我们先建立一个数组ends，把首元素放进去，然后比较之后的元素，如果遍历到的新元素比ends数组中的首元素小的话，替换首元素为此新元素，如果遍历到的新元素比ends数组中的末尾元素还大的话，将此新元素添加到ends数组末尾(注意不覆盖原末尾元素)。如果遍历到的新元素比ends数组首元素大，比尾元素小时，此时用二分查找法找到**第一个不小于此新元素**的位置，覆盖掉位置的原来的数字，以此类推直至遍历完整个nums数组，此时ends数组的长度就是我们要求的LIS的长度，**特别注意的是ends数组的值可能不是一个真实的LIS**，比如若输入数组nums为{4, 2， 4， 5， 3， 7}，那么算完后的ends数组为{2， 3， 5， 7}，可以发现它不是一个原数组的LIS，只是长度相等而已，千万要注意这点。
~~~
public int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length];
    int size = 0;
    for (int x : nums) {
        int i = 0, j = size;
        while (i != j) {
            int m = (i + j) / 2;
            if (tails[m] < x)
                i = m + 1;//这是左闭右闭的区间，i先加1，所有i会停在大的地方
            else
                j = m;
        }
        tails[i] = x;
        if (i == size) ++size;//说明比最大的还要大，size++
    }
    return size;
}
~~~
### LeetCode301. Remove Invalid Parentheses
移除无效的字符串
题解：
- 如果合法的括号，左括号和右括号的数量应该是相等的，先计算左右括号应该删除的数目
- 深度递归搜索：
   - 如果是左括号：
	    - 去除这个左括号，就将rml-1
			- 不去除这个左括号，就rml不变，open+1
	 - 如果是右括号：
	    - 去除这个右括号，就将rmR-1
			- 不去除这个右括号，就rml不变，open-1，代表闭合了
	 - 如果是普通字符，就自动i+1
~~~
public List<String> removeInvalidParentheses(String s) {
    int rmL = 0, rmR = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            rmL++;
        } else if (s.charAt(i) == ')') {
            if (rmL != 0) {
                rmL--;
            } else {
                rmR++;
            }
        }
    }
    Set<String> res = new HashSet<>();
    dfs(s, 0, res, new StringBuilder(), rmL, rmR, 0);
    return new ArrayList<String>(res);
}

public void dfs(String s, int i, Set<String> res, StringBuilder sb, int rmL, int rmR, int open) {
    if (rmL < 0 || rmR < 0 || open < 0) {
        return;
    }
    if (i == s.length()) {
        if (rmL == 0 && rmR == 0 && open == 0) {
            res.add(sb.toString());
        }        
        return;
    }

    char c = s.charAt(i); 
    int len = sb.length();

    if (c == '(') {
        dfs(s, i + 1, res, sb, rmL - 1, rmR, open);		    // 不使用 (
    	dfs(s, i + 1, res, sb.append(c), rmL, rmR, open + 1);       // 使用 (

    } else if (c == ')') {
        dfs(s, i + 1, res, sb, rmL, rmR - 1, open);	            // 不使用  )
    	dfs(s, i + 1, res, sb.append(c), rmL, rmR, open - 1);  	    // 使用 )

    } else {
        dfs(s, i + 1, res, sb.append(c), rmL, rmR, open);	
    }

    sb.setLength(len);        
}
~~~
### LeetCode303. Range Sum Query - Immutable
给定数组，找到在指定索引之内的数的和
题解：初始化的时候计算所有的和，返回的时候直接区间相减
~~~
public class NumArray {
	int[] nums;

	public NumArray(int[] nums) {
			for(int i = 1; i < nums.length; i++)
					nums[i] += nums[i - 1];

			this.nums = nums;
	}

	public int sumRange(int i, int j) {
			if(i == 0)
					return nums[j];

			return nums[j] - nums[i - 1];
	}
}
~~~






















