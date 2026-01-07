---
sidebar_position: 15
title: 项目实战
---

# 项目实战

通过实际项目巩固 C 语言知识。

## 项目 1：命令行计算器

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

double calculate(double a, char op, double b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return b != 0 ? a / b : 0;
        default: return 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s <数字> <运算符> <数字>\n", argv[0]);
        printf("示例: %s 10 + 5\n", argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    char op = argv[2][0];
    double b = atof(argv[3]);

    double result = calculate(a, op, b);
    printf("%.2f %c %.2f = %.2f\n", a, op, b, result);

    return 0;
}
```

## 项目 2：通讯录管理

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20

typedef struct {
    char name[NAME_LEN];
    char phone[PHONE_LEN];
} Contact;

typedef struct {
    Contact contacts[MAX_CONTACTS];
    int count;
} AddressBook;

void addContact(AddressBook *book) {
    if (book->count >= MAX_CONTACTS) {
        printf("通讯录已满\n");
        return;
    }

    Contact *c = &book->contacts[book->count];
    printf("姓名: ");
    scanf("%49s", c->name);
    printf("电话: ");
    scanf("%19s", c->phone);
    book->count++;
    printf("添加成功\n");
}

void listContacts(AddressBook *book) {
    printf("\n=== 通讯录 (%d人) ===\n", book->count);
    for (int i = 0; i < book->count; i++) {
        printf("%d. %s - %s\n", i + 1,
               book->contacts[i].name,
               book->contacts[i].phone);
    }
}

void searchContact(AddressBook *book) {
    char name[NAME_LEN];
    printf("搜索姓名: ");
    scanf("%49s", name);

    for (int i = 0; i < book->count; i++) {
        if (strstr(book->contacts[i].name, name)) {
            printf("找到: %s - %s\n",
                   book->contacts[i].name,
                   book->contacts[i].phone);
            return;
        }
    }
    printf("未找到\n");
}

int main(void) {
    AddressBook book = {.count = 0};
    int choice;

    while (1) {
        printf("\n1.添加 2.列表 3.搜索 0.退出\n选择: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1: addContact(&book); break;
            case 2: listContacts(&book); break;
            case 3: searchContact(&book); break;
            case 0: return 0;
        }
    }
}
```

## 项目 3：简易文件加密

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void xorEncrypt(const char *in, const char *out, const char *key) {
    FILE *fin = fopen(in, "rb");
    FILE *fout = fopen(out, "wb");

    if (!fin || !fout) {
        printf("文件打开失败\n");
        return;
    }

    int keyLen = strlen(key);
    int keyIdx = 0;
    int ch;

    while ((ch = fgetc(fin)) != EOF) {
        fputc(ch ^ key[keyIdx], fout);
        keyIdx = (keyIdx + 1) % keyLen;
    }

    fclose(fin);
    fclose(fout);
    printf("完成: %s -> %s\n", in, out);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s <输入文件> <输出文件> <密钥>\n", argv[0]);
        return 1;
    }

    xorEncrypt(argv[1], argv[2], argv[3]);
    return 0;
}
```

## 项目 4：词频统计

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORDS 1000
#define WORD_LEN 50

typedef struct {
    char word[WORD_LEN];
    int count;
} WordCount;

WordCount words[MAX_WORDS];
int wordCount = 0;

void addWord(const char *word) {
    // 查找是否已存在
    for (int i = 0; i < wordCount; i++) {
        if (strcasecmp(words[i].word, word) == 0) {
            words[i].count++;
            return;
        }
    }

    // 添加新词
    if (wordCount < MAX_WORDS) {
        strcpy(words[wordCount].word, word);
        words[wordCount].count = 1;
        wordCount++;
    }
}

int compare(const void *a, const void *b) {
    return ((WordCount*)b)->count - ((WordCount*)a)->count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("用法: %s <文件名>\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        printf("无法打开文件\n");
        return 1;
    }

    char word[WORD_LEN];
    int idx = 0;
    int ch;

    while ((ch = fgetc(fp)) != EOF) {
        if (isalpha(ch)) {
            if (idx < WORD_LEN - 1) {
                word[idx++] = tolower(ch);
            }
        } else if (idx > 0) {
            word[idx] = '\0';
            addWord(word);
            idx = 0;
        }
    }
    fclose(fp);

    // 排序并输出前10
    qsort(words, wordCount, sizeof(WordCount), compare);

    printf("\n=== 词频统计 Top 10 ===\n");
    for (int i = 0; i < 10 && i < wordCount; i++) {
        printf("%2d. %-15s %d\n", i+1, words[i].word, words[i].count);
    }

    return 0;
}
```

## 项目 5：简易内存池

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POOL_SIZE 4096
#define BLOCK_SIZE 64

typedef struct Block {
    struct Block *next;
} Block;

typedef struct {
    char memory[POOL_SIZE];
    Block *freeList;
} MemoryPool;

MemoryPool* pool_create(void) {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    pool->freeList = NULL;

    // 初始化空闲链表
    int numBlocks = POOL_SIZE / BLOCK_SIZE;
    for (int i = 0; i < numBlocks; i++) {
        Block *block = (Block*)(pool->memory + i * BLOCK_SIZE);
        block->next = pool->freeList;
        pool->freeList = block;
    }

    return pool;
}

void* pool_alloc(MemoryPool *pool) {
    if (pool->freeList == NULL) {
        return NULL;
    }

    Block *block = pool->freeList;
    pool->freeList = block->next;
    return block;
}

void pool_free(MemoryPool *pool, void *ptr) {
    Block *block = (Block*)ptr;
    block->next = pool->freeList;
    pool->freeList = block;
}

void pool_destroy(MemoryPool *pool) {
    free(pool);
}

int main(void) {
    MemoryPool *pool = pool_create();

    // 分配一些内存
    void *p1 = pool_alloc(pool);
    void *p2 = pool_alloc(pool);
    void *p3 = pool_alloc(pool);

    printf("分配了 3 块内存\n");

    // 释放
    pool_free(pool, p2);
    printf("释放了 1 块内存\n");

    // 再次分配
    void *p4 = pool_alloc(pool);
    printf("再次分配: %s\n", p4 == p2 ? "复用了之前的块" : "新块");

    pool_destroy(pool);
    return 0;
}
```

祝贺！你已经完成了 C 语言的学习！🎉
