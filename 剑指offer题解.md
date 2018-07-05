# 合并两个排序的列表
## 题目要求：
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

### 递归解法：
~~~
public class Solution
{
	public ListNode Merge(ListNode list1,ListNode list2)
	{
		if(list1 == null) return list2;
		if(list2 == null) return list1;
		if(list1.val <= list2.val)
		{
			list1.next = Merge(list1.next,list2);
			return list1.next;
		}
		else
		{
			list2.next = Merge(list1,list2.next);
			return list2.next;
		}
			
	}
}
~~~

### 非递归解法：
~~~
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {//将list1设为小
        if(list1 == null) return list2;
        if(list2 == null) return list1;
        ListNode first = new ListNode(-1);
        if(list1.val > list2.val) 
        {
            ListNode temp = list1;
            list1 = list2;
            list2 = temp;
        }
        first.next = list1;
        while(list1.next != null && list2 != null)
        {
            while(list1.next.val < list2.val)
            {
                list1 = list1.next;
            }
            ListNode temp = list1.next;
            list1.next = list2;
            list1 = list1.next;
            list2 = temp;//此时list1为主线上
        }
        if(list2 != null)
        {
            list1.next = list2;
        }
        return first.next;
    }
}
~~~
