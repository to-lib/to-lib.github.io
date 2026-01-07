---
sidebar_position: 20
title: 多线程编程
---

# C 语言多线程编程

使用 POSIX 线程 (pthread) 进行并发编程。

## 线程基础

### 创建线程

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void *thread_func(void *arg) {
    int id = *(int*)arg;
    printf("线程 %d 开始\n", id);
    sleep(1);
    printf("线程 %d 结束\n", id);
    return NULL;
}

int main(void) {
    pthread_t threads[3];
    int ids[3] = {1, 2, 3};

    // 创建线程
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, thread_func, &ids[i]);
    }

    // 等待线程结束
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("所有线程完成\n");
    return 0;
}

// 编译: gcc -pthread program.c -o program
```

### 线程返回值

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

void *compute(void *arg) {
    int n = *(int*)arg;
    int *result = malloc(sizeof(int));
    *result = n * n;
    return result;
}

int main(void) {
    pthread_t thread;
    int num = 5;
    int *result;

    pthread_create(&thread, NULL, compute, &num);
    pthread_join(thread, (void**)&result);

    printf("%d² = %d\n", num, *result);
    free(result);

    return 0;
}
```

## 互斥锁

### 基本使用

```c
#include <stdio.h>
#include <pthread.h>

int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void *increment(void *arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main(void) {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d (期望 200000)\n", counter);

    pthread_mutex_destroy(&lock);
    return 0;
}
```

### 死锁避免

```c
#include <pthread.h>

pthread_mutex_t lock_a = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_b = PTHREAD_MUTEX_INITIALIZER;

// 错误：可能死锁
void *bad_thread1(void *arg) {
    pthread_mutex_lock(&lock_a);
    pthread_mutex_lock(&lock_b);  // 等待 lock_b
    // ...
    pthread_mutex_unlock(&lock_b);
    pthread_mutex_unlock(&lock_a);
    return NULL;
}

void *bad_thread2(void *arg) {
    pthread_mutex_lock(&lock_b);
    pthread_mutex_lock(&lock_a);  // 等待 lock_a -> 死锁！
    // ...
    pthread_mutex_unlock(&lock_a);
    pthread_mutex_unlock(&lock_b);
    return NULL;
}

// 正确：保持一致的锁顺序
void *good_thread(void *arg) {
    pthread_mutex_lock(&lock_a);  // 总是先锁 a
    pthread_mutex_lock(&lock_b);  // 再锁 b
    // ...
    pthread_mutex_unlock(&lock_b);
    pthread_mutex_unlock(&lock_a);
    return NULL;
}
```

## 条件变量

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool ready = false;
int data = 0;

void *producer(void *arg) {
    pthread_mutex_lock(&mutex);

    data = 42;
    ready = true;
    printf("生产者: 数据准备好了\n");

    pthread_cond_signal(&cond);  // 唤醒消费者
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void *consumer(void *arg) {
    pthread_mutex_lock(&mutex);

    while (!ready) {  // 使用 while 防止虚假唤醒
        printf("消费者: 等待数据...\n");
        pthread_cond_wait(&cond, &mutex);
    }

    printf("消费者: 收到数据 %d\n", data);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(void) {
    pthread_t prod, cons;

    pthread_create(&cons, NULL, consumer, NULL);
    sleep(1);  // 确保消费者先等待
    pthread_create(&prod, NULL, producer, NULL);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    return 0;
}
```

## 读写锁

```c
#include <stdio.h>
#include <pthread.h>

int shared_data = 0;
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

void *reader(void *arg) {
    int id = *(int*)arg;

    pthread_rwlock_rdlock(&rwlock);
    printf("读者 %d: 数据 = %d\n", id, shared_data);
    pthread_rwlock_unlock(&rwlock);

    return NULL;
}

void *writer(void *arg) {
    int id = *(int*)arg;

    pthread_rwlock_wrlock(&rwlock);
    shared_data++;
    printf("写者 %d: 数据更新为 %d\n", id, shared_data);
    pthread_rwlock_unlock(&rwlock);

    return NULL;
}

int main(void) {
    pthread_t threads[5];
    int ids[5] = {1, 2, 3, 4, 5};

    // 创建读者和写者
    pthread_create(&threads[0], NULL, reader, &ids[0]);
    pthread_create(&threads[1], NULL, writer, &ids[1]);
    pthread_create(&threads[2], NULL, reader, &ids[2]);
    pthread_create(&threads[3], NULL, reader, &ids[3]);
    pthread_create(&threads[4], NULL, writer, &ids[4]);

    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_rwlock_destroy(&rwlock);
    return 0;
}
```

## 线程池

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

#define POOL_SIZE 4
#define QUEUE_SIZE 100

typedef struct {
    void (*func)(void*);
    void *arg;
} Task;

typedef struct {
    pthread_t threads[POOL_SIZE];
    Task queue[QUEUE_SIZE];
    int front, rear, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    bool shutdown;
} ThreadPool;

void *worker(void *arg) {
    ThreadPool *pool = (ThreadPool*)arg;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->not_empty, &pool->mutex);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // 取任务
        Task task = pool->queue[pool->front];
        pool->front = (pool->front + 1) % QUEUE_SIZE;
        pool->count--;

        pthread_cond_signal(&pool->not_full);
        pthread_mutex_unlock(&pool->mutex);

        // 执行任务
        task.func(task.arg);
    }

    return NULL;
}

ThreadPool* pool_create(void) {
    ThreadPool *pool = calloc(1, sizeof(ThreadPool));
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->not_empty, NULL);
    pthread_cond_init(&pool->not_full, NULL);

    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_create(&pool->threads[i], NULL, worker, pool);
    }

    return pool;
}

void pool_submit(ThreadPool *pool, void (*func)(void*), void *arg) {
    pthread_mutex_lock(&pool->mutex);

    while (pool->count == QUEUE_SIZE) {
        pthread_cond_wait(&pool->not_full, &pool->mutex);
    }

    pool->queue[pool->rear].func = func;
    pool->queue[pool->rear].arg = arg;
    pool->rear = (pool->rear + 1) % QUEUE_SIZE;
    pool->count++;

    pthread_cond_signal(&pool->not_empty);
    pthread_mutex_unlock(&pool->mutex);
}

void pool_destroy(ThreadPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->not_empty);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    free(pool);
}

// 使用示例
void task_func(void *arg) {
    int id = *(int*)arg;
    printf("执行任务 %d\n", id);
    free(arg);
}

int main(void) {
    ThreadPool *pool = pool_create();

    for (int i = 0; i < 20; i++) {
        int *id = malloc(sizeof(int));
        *id = i;
        pool_submit(pool, task_func, id);
    }

    sleep(2);
    pool_destroy(pool);

    return 0;
}
```

## 编译和注意事项

```bash
# 编译时需要链接 pthread 库
gcc -pthread program.c -o program

# 或
gcc program.c -o program -lpthread
```

**注意事项：**

- 共享数据访问要加锁
- 避免死锁（保持锁顺序一致）
- 使用 `while` 检查条件变量
- 及时释放锁和销毁资源

多线程编程是构建高性能应用的关键技能！
