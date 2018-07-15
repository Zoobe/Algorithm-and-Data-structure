import java.util.Stack;
import java.util.ArrayDeque;
import java.util.ArrayList;

public class TreeNode{
	TreeNode.val=null;
	TreeNode.left=null;
	TreeNode.right=null;
	
	public TreeNode(int val){
		TreeNode.val =val;
	}
	
class BinaryTree{
	public BinaryTree()
	{
		ArrayList binarytree = new ArrayList<>();
	}
	/*二叉树的前序遍历*/
	
	public void PreOrder(TreeNode root)
	{
		if(root==null) return;
		binarytree.add(root.val);
		PreOrder(root.left);
		PreOrder(root.right);
		
	}
	
	/*二叉树的前序遍历---非递归实现*/
	public void PreOrder2(TreeNode root)
	{
		Stack<TreeNode> stack = new Stack<>();
		if(root!=null) stack.push(root);
		while(!stack.empty())
		{
			TreeNode node = stack.pop();
			binarytree.add(node.val);
			stack.push(node.right);  //右节点先入栈，后出栈
			stack.push(node.left);   //左节点后入栈，先出栈
		}
	}
	
	/*二叉树的中序遍历----递归实现*/
	public void MidOrder(TreeNode root)
	{
		if(root==null) return;
		MidOrder(root.left);
		binarytree.add(root.val);
		MidOrder(root.right);
	}
	
	/*二叉树中序遍历----非递归实现*/
	public void MidOrder2(TreeNode root)
	{
		Stack<TreeNode> stack =  new Stack<>();
		
		while( root!=null || !stack.empty() )
		{
			while(root!=null)
			{
				stack.push(root);
				root=root.left;      //从根节点开始一直往左走，把每个左节点入栈
			}
			
			if(!stack.empty())   
			{
				TreeNode node = stack.pop();    //当栈不为空时弹出一个节点
				binary.add(node.val);
				root=node.right;        //查找该节点是否有右节点，如果有右节点，再次循环，先将右节点入栈
			}                               //再将右节点的左节点循环加入栈
		}
	}
	
	/*二叉树的后续遍历-----递归实现*/
	public void PostOrder(TreeNode root)
	{
		if(root==null) return;
		PostOrder(root.left);
		binary.add(root.val);
		PostOrder(root.right);
	}
	
	/*二叉树的层序遍历（广度优先遍历） */
	public void LayerOrder(TreeNode root)
	{
		Deque<TreeNode> deque = new Deque<>();
		if(root!=null) deque.add(root);
		while( !deque.isEmpty() )
		{
			TreeNode node = deque.poll();
			binarytree.add(node.val);
			deque.add(node.left);
			deque.add(node.right);
		}
	}
			
  
  
  
  
  
  
  
  
