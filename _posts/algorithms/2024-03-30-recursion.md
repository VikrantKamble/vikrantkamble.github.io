---
layout: post
title:  Recursion
date:   2024-03-30 00:00:00 +0000
categories:
  - algorithms
usemathjax: True
---

A recursive algorithm is one which calls itself, on a subset of input. To setup a correct recursion and to avoid infinite loops, one
must take the following into account:
- setup the base case
- recursive calls should lead towards the base case

A classical example of recursion is the **Tower of Hanoi**. There are 3 poles, with the first pole containing stack of discs with decreasing diameter. The 
goal is to recreate this structure in the third pole, using the second pole as helper. At any time one can only move 1 disc and one can **never** stack a 
disc of higher diameter on top of a lower diameter one.

![](/images/hanoi.png)

- What is the base case?

What if we had only one disc, what do we do? Simple, we just move it to the destination pole in one move. What if we had no disc, 
well, in that case we don't do anything and simply stop.

- What is the recursive setup?

Assume that somehow we were able to get the top $n-1$ discs to the second pole, what do we do now. In that case, we move the last 
remaining disc from the first pole to the third pole and then somehow get the $n-1$ discs from the second pole to the third pole. 
But **who** is that somehow? It is the algorithm itself!

```python
def tower_of_hanoi(n, start, middle, end):
    if n > 0:
        tower_of_hanoi(n-1, start, end, middle)
        print(f"Move disc from {start} to {end}")
        tower_of_hanoi(n-1, middle, start, end)
```

As we saw [previously]({% post_url /algorithms/2024-03-30-binary-tree %}), a *depth-first* traversal of a *binary tree* can also be implemented recursively.

---

Recusive/iterative duality
----------

**Note:** A recursive algorithm can also be implemented as an iterative algorithm and vice-versa. Sometimes, one approach is much easier to help solve the problem than the other. So, if you feel stuck or having a hard time with one approach, using the complementary approach might help. 

Most of the graph and trees problem are easier to solve with the *recursive* approach; while problems involving arrays and dynamic programming are mostly easier with the *iterative* approach. The reason being that an array can be traversed from left to right or from right to left with the same effort. However, trees for example have a natural ordering; it is easier to go from *node* to its *children*, rather than from *children* to its *parent*. The base case in this type of data structure are the *leaf* children. Hence a *recusive* approach works best.

In this [post]({% post_url /algorithms/2024-03-08-arrays %}) we saw an iterative implementation of Kadane's algorithm. Here we turn it into a recursion. The strategy is the same - we find, for each index, the max sum subarray that ends on that index, and then we choose the max out of these. **H**owever the direction of traversal is opposite; whereas in the iterative case, we started filling from `left` to `right`; for the recursive case we start the recursive from the last index and recurse to the base case - the base case being the single first element.

```python
def max_sum_subarray_rec(nums):
    def util(i):
        if i == 0:  # base case
            curr_sum = nums[i]
        else:  # recursive case
            curr_sum = max(nums[i], nums[i] + util(i-1))
        util.res = max(util.res, curr_sum)
        return curr_sum
    
    util.res = float('-inf')
    util(len(nums) - 1)
    return util.res
```

---

Depth Sum
--------

Given a list where each location can contain either an integer or a list, compute the depth sum?

E.g. $nums = [1, 2, [3, 4], 5]$, the depth sum is $ 1\*1 + 1\*2 + 2\*3 + 2\*4 + 1\*5 = 22 $.

For $nums = [1, [2, [3]], 4]$, the depth sum is $1\*1 + 2\*2 + 3\*3 + 1\*4 = 18$.

Let' say we already have an algorithm that computes the depth sum; how will it look like. We can iterate over the input 
argument, in this case a list. If the element is an integer we can simply update the sum. If however, the element is a list, we 
can use the algorithm **itself** to get the depth sum of the element; with depth **one** level **deeper**.

What will the base case? The base case will be when the input consists of all integers and no nested lists.

```python
def depth_sum(nums):
    def _depth_sum(nums, depth=0):
        curr_sum = 0
        for ele in nums:
            if isinstance(ele, list):
                curr_sum += _depth_sum(ele, depth+1)  # recurse one level deeper
            else:
                curr_sum += ele
        return curr_sum
    return _depth_sum(nums)
```

Merge Sort
--------

Let's take the classic problem of sorting an array and see how we can apply recursion to solve the task. 

We basically want a function that takes an input and sorts it. Let's say we already have a function to do this; how will it look like. We can 
break the input array into parts and sort each of the part. This is the recursive aspect - we can use the function **itself** to accomplish this. 
The base case being when there is a single element, in which case nothing needs to be done.

Now that we have two sorted arrays, how to put them together into one sorted array? We can use the *pointer* technique as discussed here. We keep 
two pointers at the start of each of the two sorted arrays. We then compare the values at these pointer locations and choose the smallest one and keep going.
This leads to the **mergesort** algorithm as follows:

```python
def merge_sort(nums)
    def _merge_sort(nums, start, end):
        if len(nums) == 0:
            raise ValueError("Empty input array!") 
        
        if start == end:  # base case
            return [nums[start]]

        mid = (start + end) // 2
        _merge_sort(nums, start, mid)  # recurse on the left
        _merge_sort(nums, mid + 1, end)  # recurse on the right

        # merge two sorted arrays, in-place
        i = 0
        while i < mid + 1:
            if nums[i] > nums[mid + 1]:
                nums[mid + 1], nums[i] = nums[i], nums[mid + 1]
            i += 1
```
---

Split BST
----------

Let us take the following problem. We want to split a BST into two separate BSTs given a target. One subtree has nodes that are all smaller or equal to the target value, while the other subtree has all nodes that are greater than the target value.

Again, let's apply the recursive formulation:
- What is the base case?
    - Well, if the root itself is `NULL`, then we can simply return `[None, None]`. 
    - What if the root itself is equal to the target? Since its a BST, we know that the right subtree starting at *root* will all have values greater than the *root*. Hence we can simply detach the right subtree and return `[root, root.right]` as the result.

- What is the recursive case?
Depending on the value of the root, we recurse on the *left* subtree or the *right* subtree.

```python
def split_bst(root, target):
    if not root:
        return None, None
    if root.val == target:
        rightchild = root.right
        root.right = None
        return root, rightchild
    elif root.val < target:
        small, large = split_bst(root.right, target)
        root.right = small
        return root, large
    else:
        small, large = split_bst(root.left, target)
        root.left = large
        return small, root
```

---

Max Sum Path BST
--------

Given a binary tree, find the path with the maximum sum. The path can start at any node and end at any node.

Here we will apply the *recusive* version of Kadane's algorithm. Let's say, for a given node, we have the two *max paths*, one ending on the *left* child and one ending on the *right* child. Now what is the max path **ending on the node**? We have three options: either we extend the `left` path (green), or we extend the `right` path (blue); or we simply go solo and do not extend either of the paths (red).

![](/images/posts/max_path_bt.png)

But what about the max path in general, that can start at any node and end at any node. Well, for this case we also need to take the path that can be obtaining the *left* and the *right* path (shown in orange). That is:

$$max sum = max(max sum, green, blue, red, orange)$$

```python
def max_sum_path_util(root):
    """ Returns max sum path ending on the root.
    """
    if root is None: # Base case
        return 0
    
    left_end_max_sum = max_sum_path_util(root.left)  # recursive case
    right_end_max_sum = max_sum_path_util(root.right)  # recursive case

    max_single_left_right = max(
        left_end_max_sum + root.val,    # green path
        right_end_max_sum + root.val,   # blue path
        root.val,                       # red path
    )

    # global max path
    max_sum = max(
        left_end_max_sum + right_end_max_sum + root.val, # orange path
        max_single_left_right
    )
    max_sum_path_util.res = max(max_sum_path_util.res, max_sum)

    return max_single_left_right

def max_sum_path(root):
    max_sum_path_util.res = float('-inf')

    max_sum_path_util(root)
    return max_sum_path_util.res
```