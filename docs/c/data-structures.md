---
sidebar_position: 12
title: 数据结构实现
---

# 数据结构实现

使用 C 语言实现常用数据结构。

## 链表

### 单链表

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *head;
    int size;
} LinkedList;

LinkedList* createList(void) {
    LinkedList *list = malloc(sizeof(LinkedList));
    list->head = NULL;
    list->size = 0;
    return list;
}

void append(LinkedList *list, int data) {
    Node *node = malloc(sizeof(Node));
    node->data = data;
    node->next = NULL;

    if (list->head == NULL) {
        list->head = node;
    } else {
        Node *curr = list->head;
        while (curr->next) curr = curr->next;
        curr->next = node;
    }
    list->size++;
}

void prepend(LinkedList *list, int data) {
    Node *node = malloc(sizeof(Node));
    node->data = data;
    node->next = list->head;
    list->head = node;
    list->size++;
}

void removeAt(LinkedList *list, int index) {
    if (index < 0 || index >= list->size) return;

    Node *temp;
    if (index == 0) {
        temp = list->head;
        list->head = list->head->next;
    } else {
        Node *curr = list->head;
        for (int i = 0; i < index - 1; i++) {
            curr = curr->next;
        }
        temp = curr->next;
        curr->next = temp->next;
    }
    free(temp);
    list->size--;
}

void printList(LinkedList *list) {
    Node *curr = list->head;
    while (curr) {
        printf("%d -> ", curr->data);
        curr = curr->next;
    }
    printf("NULL\n");
}

void freeList(LinkedList *list) {
    Node *curr = list->head;
    while (curr) {
        Node *temp = curr;
        curr = curr->next;
        free(temp);
    }
    free(list);
}

int main(void) {
    LinkedList *list = createList();

    append(list, 10);
    append(list, 20);
    append(list, 30);
    prepend(list, 5);

    printList(list);  // 5 -> 10 -> 20 -> 30 -> NULL

    removeAt(list, 1);
    printList(list);  // 5 -> 20 -> 30 -> NULL

    freeList(list);
    return 0;
}
```

## 栈

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 100

typedef struct {
    int data[MAX_SIZE];
    int top;
} Stack;

Stack* createStack(void) {
    Stack *s = malloc(sizeof(Stack));
    s->top = -1;
    return s;
}

bool isEmpty(Stack *s) { return s->top == -1; }
bool isFull(Stack *s) { return s->top == MAX_SIZE - 1; }

void push(Stack *s, int value) {
    if (!isFull(s)) {
        s->data[++s->top] = value;
    }
}

int pop(Stack *s) {
    if (!isEmpty(s)) {
        return s->data[s->top--];
    }
    return -1;
}

int peek(Stack *s) {
    if (!isEmpty(s)) {
        return s->data[s->top];
    }
    return -1;
}

int main(void) {
    Stack *s = createStack();

    push(s, 10);
    push(s, 20);
    push(s, 30);

    printf("栈顶: %d\n", peek(s));
    printf("弹出: %d\n", pop(s));
    printf("弹出: %d\n", pop(s));

    free(s);
    return 0;
}
```

## 队列

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 100

typedef struct {
    int data[MAX_SIZE];
    int front, rear;
} Queue;

Queue* createQueue(void) {
    Queue *q = malloc(sizeof(Queue));
    q->front = 0;
    q->rear = -1;
    return q;
}

bool isEmpty(Queue *q) { return q->rear < q->front; }
bool isFull(Queue *q) { return q->rear >= MAX_SIZE - 1; }

void enqueue(Queue *q, int value) {
    if (!isFull(q)) {
        q->data[++q->rear] = value;
    }
}

int dequeue(Queue *q) {
    if (!isEmpty(q)) {
        return q->data[q->front++];
    }
    return -1;
}

int main(void) {
    Queue *q = createQueue();

    enqueue(q, 10);
    enqueue(q, 20);
    enqueue(q, 30);

    printf("出队: %d\n", dequeue(q));
    printf("出队: %d\n", dequeue(q));

    free(q);
    return 0;
}
```

## 二叉树

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct TreeNode {
    int data;
    struct TreeNode *left, *right;
} TreeNode;

TreeNode* createNode(int data) {
    TreeNode *node = malloc(sizeof(TreeNode));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}

// 前序遍历
void preorder(TreeNode *root) {
    if (root == NULL) return;
    printf("%d ", root->data);
    preorder(root->left);
    preorder(root->right);
}

// 中序遍历
void inorder(TreeNode *root) {
    if (root == NULL) return;
    inorder(root->left);
    printf("%d ", root->data);
    inorder(root->right);
}

// 后序遍历
void postorder(TreeNode *root) {
    if (root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->data);
}

void freeTree(TreeNode *root) {
    if (root == NULL) return;
    freeTree(root->left);
    freeTree(root->right);
    free(root);
}

int main(void) {
    TreeNode *root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    printf("前序: "); preorder(root); printf("\n");
    printf("中序: "); inorder(root); printf("\n");
    printf("后序: "); postorder(root); printf("\n");

    freeTree(root);
    return 0;
}
```

## 哈希表

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 100

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct {
    Entry *buckets[SIZE];
} HashMap;

unsigned int hash(const char *key) {
    unsigned int h = 0;
    while (*key) h = h * 31 + *key++;
    return h % SIZE;
}

HashMap* createMap(void) {
    HashMap *map = calloc(1, sizeof(HashMap));
    return map;
}

void put(HashMap *map, const char *key, int value) {
    unsigned int idx = hash(key);
    Entry *e = map->buckets[idx];

    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return;
        }
        e = e->next;
    }

    Entry *new = malloc(sizeof(Entry));
    new->key = strdup(key);
    new->value = value;
    new->next = map->buckets[idx];
    map->buckets[idx] = new;
}

int get(HashMap *map, const char *key, int *found) {
    unsigned int idx = hash(key);
    Entry *e = map->buckets[idx];

    while (e) {
        if (strcmp(e->key, key) == 0) {
            if (found) *found = 1;
            return e->value;
        }
        e = e->next;
    }
    if (found) *found = 0;
    return 0;
}

int main(void) {
    HashMap *map = createMap();

    put(map, "apple", 100);
    put(map, "banana", 200);

    int found;
    printf("apple: %d\n", get(map, "apple", &found));
    printf("banana: %d\n", get(map, "banana", &found));

    return 0;
}
```

掌握数据结构后，就可以继续学习项目实战了！
