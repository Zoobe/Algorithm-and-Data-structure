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
我们使用一个小trick，我们分别找第 (m+n+1) / 2 个，和 (m+n+2) / 2 个，然后求其平均值即可，这对奇偶数均适用。加入 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身。
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
          return getkth(A, aStart + k/2, B, bStart,       k - k/2);// Check: aRight + bLeft 
      else 
          return getkth(A, aStart,       B, bStart + k/2, k - k/2);// Check: bRight + aLeft
    }
~~~
