---
sidebar_position: 14
title: 回溯算法
---

# 回溯算法

回溯算法通过递归尝试所有可能的解，遇到不满足条件时回退。

## 📖 框架模板

```java
void backtrack(路径, 选择列表) {
    if (满足结束条件) {
        结果.add(路径);
        return;
    }
    for (选择 : 选择列表) {
        做选择;
        backtrack(路径, 选择列表);
        撤销选择;
    }
}
```

## 🎯 经典问题

### 全排列

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backtrack(nums, new ArrayList<>(), new boolean[nums.length], res);
    return res;
}

private void backtrack(int[] nums, List<Integer> path, boolean[] used,
                       List<List<Integer>> res) {
    if (path.size() == nums.length) {
        res.add(new ArrayList<>(path));
        return;
    }
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;
        path.add(nums[i]);
        used[i] = true;
        backtrack(nums, path, used, res);
        path.remove(path.size() - 1);
        used[i] = false;
    }
}
```

### 子集

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), res);
    return res;
}

private void backtrack(int[] nums, int start, List<Integer> path,
                       List<List<Integer>> res) {
    res.add(new ArrayList<>(path));
    for (int i = start; i < nums.length; i++) {
        path.add(nums[i]);
        backtrack(nums, i + 1, path, res);
        path.remove(path.size() - 1);
    }
}
```

### N 皇后

```java
public List<List<String>> solveNQueens(int n) {
    List<List<String>> res = new ArrayList<>();
    char[][] board = new char[n][n];
    for (char[] row : board) Arrays.fill(row, '.');
    backtrack(board, 0, res);
    return res;
}

private void backtrack(char[][] board, int row, List<List<String>> res) {
    if (row == board.length) {
        res.add(construct(board));
        return;
    }
    for (int col = 0; col < board.length; col++) {
        if (!isValid(board, row, col)) continue;
        board[row][col] = 'Q';
        backtrack(board, row + 1, res);
        board[row][col] = '.';
    }
}
```
