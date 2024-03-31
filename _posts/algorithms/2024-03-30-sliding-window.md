---
layout: post
title:  Sliding window
date:   2024-03-30 00:00:00 +0000
categories:
  - algorithms
  - arrays
usemathjax: True
---

This **iterative** technique is useful when dealing with problems that involve finding a (longest/shortest) subsequence that satisfies a given criteria in a
string or an array. We maintain two pointers, *left* and *right*; initially located at the first index. 

$\rightarrow$ For the case of finding the **longest** subarray/substrig, the *right* pointer is moved *as long as* the subarray defined by the range [left, right] satisfies the window (subarray) criterion. Once the window 
condition is broken, the *left* pointer is moved until the window condition gets satisfied or it catches up to the *right* pointer.
A typical logic will look like this:

```python
def solution(nums, *args, **kwargs):
    left, res = 0, 0
    for right in range(len(nums)):
        while (window_condition_broken) and (left < right):
            # do some logic here
            left += 1
        # update res
    return res
```

$\rightarrow$For the case of finding **shortest** subsequence, the strategy is exactly reversed. The right pointer is moved until the 
subarray condition is satisfied; at which point the left pointer is incremented until the condition is broken. 

Let's apply the sliding window technique to this problem: **Given a string find the longest substring without repeating characters.**
- We will maintain a data structure to keep track of characters present in the substring
defined by [left, right]. A obvious choice here is a *set*.  
- We then keep moving the right pointer increasing the subarray length.
- The moment we reach a position where the character has already been encountered, the condition of *non-repeatedness* is broken. 
- We then move the *left* pointer until we reach a position such that the character at that *left* location is the one causing the broken window condition. 
- As we move the left pointer, we also need to make sure that we remove the previous characters from the set.

```python
def longest_non_repeating(s):
    left, max_length = 0, 0
    encountered = set()
    for right in range(len(s)):
        while (s[right] in encountered) and (left < right):
            encountered.remove(s[left])
            left += 1

        encountered.add(s[right])
        max_length = max(max_length, right - left + 1)
    return max_length
```

Let's consider another problem: 
**Given an array of positive integers nums and a positive integer target, return the minimal length of a 
subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.**

```python
def min_subarray_sum(nums, target):
    left = 0

    curr_sum, min_length = 0, float('inf')
    for right in range(len(nums)):
        curr_sum += nums[right]
        while (curr_sum >= target) and (left <= right):
            min_length = min(min_length, right - left + 1)

            curr_sum -= nums[left]
            left += 1
        
    return 0 if min_length == float('inf') else min_length
```
---

The questions we have considered above were windows that had variable size. Let us now consider a problem, where the 
size of the window is fixed - **given two strings s and p, return an array of all the start indices of p's anagrams in s**.
Both *s* and *p* only consists of lowercase english letters.

An anagram of a string is another string of equal length but with the characters jumbled up. Every though the ordering changes, the 
count map, i.e. the number of times each character appears remains unchanged. Thus our sliding window will always have the length 
equal to that of the target string, in this case *p*. We will check if the count map of the window matches that of the target, if yes 
we have a match; if not we slide the window one step to the right and keep going.

Note that when we slide the window the only characters that are being affected is the newly added rightmost character and the leftmost dropped character. 
Hence for checking the match we don't need to scan the entire hashmap.

The count map will be maintained across each of the lowercase letter; hence for the string `abbad`, the map will look like `a:2, b:2, c:0, d:1, e:0, ..... `.
We will maintain a variable called `matches` that counts how many out of the 26 characters match in their counts between the target and the current window. Let's
say that for the current window, the count of letter `f` matches and that when we slide the window the newly added character was also `f`. In that case the match 
count will decrease. Conversely, if the count of letter `f` did not match previously, then we need to check if after adding, it matches or not. If it still 
doesn't match, we do nothing, else if it matches, we increase the match by 1.

```python
def findAnagrams(self, s: str, p: str) -> List[int]:
    if len(p) > len(s):
        return []
        
    length = len(p)

    hashmap_p = defaultdict(int)
    hashmap_window = defaultdict(int)
    for i in range(len(p)):
        hashmap_p[p[i]] += 1
        hashmap_window[s[i]] += 1

    # check matches for the first window.
    matches = 0
    for ele in string.ascii_lowercase:
        if hashmap_p[ele] == hashmap_window[ele]:
            matches += 1

    res = []
    if matches == len(string.ascii_lowercase):
        res.append(0)
    
    for ptr in range(len(p), len(s)):
        # handle newly added rightmost element
        right_element = s[ptr]
        right_already_match = hashmap_p[right_element] == hashmap_window[right_element]

        hashmap_window[right_element] += 1
        if right_already_match:
            matches -= 1
        else:
            if hashmap_p[right_element] == hashmap_window[right_element]:
                matches += 1

        # handle leftmost dropped element
        left_element = s[ptr - length]
        left_already_match = hashmap_p[left_element] == hashmap_window[left_element]

        hashmap_window[left_element] -= 1
        if left_already_match:
            matches -= 1
        else:
            if hashmap_p[left_element] == hashmap_window[left_element]:
                matches += 1

        # Check if this new window matches
        if matches == len(string.ascii_lowercase):
            res.append(ptr - length + 1)

    return res
```