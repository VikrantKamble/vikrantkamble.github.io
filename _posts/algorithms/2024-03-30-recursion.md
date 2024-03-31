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

As we saw [previously]({% post_url /algorithms/2024-03-30-traversals %}), a *depth-first* traversal of a *binary tree* can also be implemented recursively.

---
**Note:** A recursive algorithm can also be implemented as an iterative algorithm and vice-versa. Sometimes, one approach is much easier to help solve the problem than the other. So, if you feel stuck or having a hard time with one approach, using the complementary approach might help. Most of the graph and trees problem are easier to solve with the *recursive* approach; while problems involving arrays and dynamic programming are mostly easier with the *iterative* approach. 

---

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