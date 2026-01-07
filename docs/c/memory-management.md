---
sidebar_position: 7
title: 内存管理
---

# 内存管理

C 语言允许程序员直接管理内存，这提供了极大的灵活性，但也需要谨慎操作。

## 内存布局

```
+------------------+
|       栈         |  <- 局部变量、函数参数
+------------------+
|        ↓         |
|                  |
|        ↑         |
+------------------+
|       堆         |  <- 动态分配的内存
+------------------+
|   未初始化数据   |  <- 全局变量（未初始化）
+------------------+
|   已初始化数据   |  <- 全局变量（已初始化）
+------------------+
|      代码段      |  <- 程序代码
+------------------+
```

## 动态内存分配

### malloc

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 分配单个整数
    int *p = (int*)malloc(sizeof(int));
    if (p == NULL) {
        printf("内存分配失败\n");
        return 1;
    }

    *p = 42;
    printf("*p = %d\n", *p);
    free(p);

    // 分配数组
    int n = 5;
    int *arr = (int*)malloc(n * sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    free(arr);
    arr = NULL;  // 防止悬空指针

    return 0;
}
```

### calloc

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // calloc 分配并初始化为0
    int n = 5;
    int *arr = (int*)calloc(n, sizeof(int));

    if (arr == NULL) {
        return 1;
    }

    // 所有元素都是0
    printf("初始值: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}
```

### realloc

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int *arr = (int*)malloc(3 * sizeof(int));
    arr[0] = 1; arr[1] = 2; arr[2] = 3;

    // 扩展数组
    int *new_arr = (int*)realloc(arr, 5 * sizeof(int));
    if (new_arr == NULL) {
        free(arr);
        return 1;
    }
    arr = new_arr;

    arr[3] = 4;
    arr[4] = 5;

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}
```

## 动态数据结构

### 动态数组

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} DynamicArray;

DynamicArray* createArray(int capacity) {
    DynamicArray *arr = malloc(sizeof(DynamicArray));
    arr->data = malloc(capacity * sizeof(int));
    arr->size = 0;
    arr->capacity = capacity;
    return arr;
}

void append(DynamicArray *arr, int value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, arr->capacity * sizeof(int));
    }
    arr->data[arr->size++] = value;
}

void freeArray(DynamicArray *arr) {
    free(arr->data);
    free(arr);
}

int main(void) {
    DynamicArray *arr = createArray(2);

    for (int i = 0; i < 10; i++) {
        append(arr, i * 10);
    }

    printf("Size: %d, Capacity: %d\n", arr->size, arr->capacity);
    for (int i = 0; i < arr->size; i++) {
        printf("%d ", arr->data[i]);
    }
    printf("\n");

    freeArray(arr);
    return 0;
}
```

### 链表

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node* createNode(int data) {
    Node *node = malloc(sizeof(Node));
    node->data = data;
    node->next = NULL;
    return node;
}

void append(Node **head, int data) {
    Node *newNode = createNode(data);
    if (*head == NULL) {
        *head = newNode;
        return;
    }
    Node *curr = *head;
    while (curr->next != NULL) {
        curr = curr->next;
    }
    curr->next = newNode;
}

void printList(Node *head) {
    while (head != NULL) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

void freeList(Node *head) {
    while (head != NULL) {
        Node *temp = head;
        head = head->next;
        free(temp);
    }
}

int main(void) {
    Node *list = NULL;

    append(&list, 10);
    append(&list, 20);
    append(&list, 30);

    printList(list);
    freeList(list);

    return 0;
}
```

## 内存问题

### 内存泄漏

```c
#include <stdlib.h>

void memoryLeak(void) {
    int *p = malloc(100 * sizeof(int));
    // 忘记 free(p)，内存泄漏！
}

void noLeak(void) {
    int *p = malloc(100 * sizeof(int));
    // 使用内存...
    free(p);  // 正确释放
}
```

### 使用 Valgrind 检测

```bash
# 编译
gcc -g program.c -o program

# 检测内存泄漏
valgrind --leak-check=full ./program
```

## 最佳实践

1. **始终检查分配结果**
2. **每个 malloc 对应一个 free**
3. **释放后将指针置为 NULL**
4. **不要释放同一内存两次**
5. **使用工具检测内存问题**

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 安全的内存管理模式
    int *p = malloc(sizeof(int) * 10);

    if (p == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    // 使用内存
    for (int i = 0; i < 10; i++) {
        p[i] = i;
    }

    // 释放并置空
    free(p);
    p = NULL;

    return 0;
}
```

掌握内存管理后，就可以继续学习结构体了！
