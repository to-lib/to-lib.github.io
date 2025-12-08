---
sidebar_position: 14
---

# 定时任务和异步处理

## 启用定时任务和异步

```java
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableScheduling    // 启用定时任务
@EnableAsync         // 启用异步处理
public class TaskConfig {
}
```

## 定时任务（@Scheduled）

### 固定延迟执行

```java
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ScheduledTaskService {
    
    // 上次执行完成后延迟 5 秒再执行
    @Scheduled(fixedDelay = 5000)
    public void taskWithFixedDelay() {
        log.info("执行固定延迟任务");
    }
    
    // 初始延迟 2 秒，之后每 5 秒执行一次
    @Scheduled(initialDelay = 2000, fixedDelay = 5000)
    public void taskWithInitialDelay() {
        log.info("执行带初始延迟的任务");
    }
}
```

### 固定速率执行

```java
@Service
@Slf4j
public class ScheduledTaskService {
    
    // 固定速率：每 5 秒执行一次，不管上次执行是否完成
    @Scheduled(fixedRate = 5000)
    public void taskWithFixedRate() {
        log.info("执行固定速率任务");
        // 如果任务耗时 2 秒，5 秒后会再执行，不会等待
    }
    
    // 初始延迟 2 秒，之后以固定速率执行
    @Scheduled(initialDelay = 2000, fixedRate = 5000)
    public void taskWithRateAndDelay() {
        log.info("执行带初始延迟的固定速率任务");
    }
}
```

### Cron 表达式

```java
@Service
@Slf4j
public class ScheduledTaskService {
    
    // 每天凌晨 2 点执行
    @Scheduled(cron = "0 0 2 * * *")
    public void dailyTask() {
        log.info("执行每日任务");
    }
    
    // 每周一上午 10 点执行
    @Scheduled(cron = "0 0 10 ? * MON")
    public void weeklyTask() {
        log.info("执行每周一任务");
    }
    
    // 每月 1 日午夜执行
    @Scheduled(cron = "0 0 0 1 * *")
    public void monthlyTask() {
        log.info("执行每月任务");
    }
    
    // 工作日每 30 分钟执行一次
    @Scheduled(cron = "0 */30 * ? * MON-FRI")
    public void workdayTask() {
        log.info("执行工作日任务");
    }
    
    // 每小时执行一次
    @Scheduled(cron = "0 0 * * * *")
    public void hourlyTask() {
        log.info("执行每小时任务");
    }
}
```

### Cron 表达式说明

```
表达式格式：秒 分 时 日 月 周 [年]

0 0 0 * * *     → 每天午夜
0 0 12 * * *    → 每天中午
0 0 9-17 * * *  → 工作时间每小时
0 */30 * * * *  → 每 30 分钟
0 0 0 1 * *     → 每月 1 日
0 0 0 ? * MON   → 每周一
0 0 0 ? * 1     → 每周日
```

### 动态定时任务

```java
import org.springframework.scheduling.support.CronTrigger;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DynamicScheduleConfig {
    
    @Bean
    public ThreadPoolTaskScheduler taskScheduler() {
        ThreadPoolTaskScheduler scheduler = new ThreadPoolTaskScheduler();
        scheduler.setPoolSize(10);
        scheduler.setThreadNamePrefix("scheduled-task-");
        scheduler.initialize();
        return scheduler;
    }
}

@Service
@RequiredArgsConstructor
public class DynamicScheduleService {
    
    private final ThreadPoolTaskScheduler taskScheduler;
    private final ScheduleRepository scheduleRepository;
    private ScheduledFuture<?> scheduledFuture;
    
    public void startDynamicSchedule(String cronExpression) {
        // 停止之前的任务
        if (scheduledFuture != null) {
            scheduledFuture.cancel(false);
        }
        
        // 创建新的定时任务
        scheduledFuture = taskScheduler.schedule(
            this::executeDynamicTask,
            new CronTrigger(cronExpression)
        );
    }
    
    private void executeDynamicTask() {
        log.info("执行动态定时任务");
    }
    
    public void stopDynamicSchedule() {
        if (scheduledFuture != null) {
            scheduledFuture.cancel(false);
            scheduledFuture = null;
        }
    }
}
```

## 异步处理（@Async）

### 基本异步方法

```java
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class AsyncTaskService {
    
    // 异步执行，无返回值
    @Async
    public void asyncTask(String taskName) {
        log.info("开始执行异步任务：{}", taskName);
        try {
            Thread.sleep(2000);  // 模拟耗时操作
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        log.info("异步任务完成：{}", taskName);
    }
    
    // 异步执行，有返回值
    @Async
    public CompletableFuture<String> asyncTaskWithReturn(String data) {
        return CompletableFuture.supplyAsync(() -> {
            log.info("处理数据：{}", data);
            return "处理结果：" + data;
        });
    }
    
    // 异步执行，返回 Future
    @Async
    public Future<String> asyncTaskWithFuture(String data) {
        return new AsyncResult<>(data + " - 异步处理完成");
    }
}
```

### 使用异步任务

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;

@RestController
@RequestMapping("/api/tasks")
@RequiredArgsConstructor
public class TaskController {
    
    private final AsyncTaskService asyncTaskService;
    
    // 无返回值的异步任务
    @GetMapping("/start/{name}")
    public ResponseEntity<String> startTask(@PathVariable String name) {
        asyncTaskService.asyncTask(name);
        return ResponseEntity.ok("任务已启动，后台执行中...");
    }
    
    // 带返回值的异步任务
    @GetMapping("/process/{data}")
    public CompletableFuture<String> processData(@PathVariable String data) {
        return asyncTaskService.asyncTaskWithReturn(data);
    }
    
    // 等待异步任务完成
    @GetMapping("/wait/{data}")
    public ResponseEntity<String> waitForTask(@PathVariable String data) 
            throws Exception {
        Future<String> result = asyncTaskService.asyncTaskWithFuture(data);
        return ResponseEntity.ok(result.get());  // 阻塞直到完成
    }
}
```

### 自定义异步执行器

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import java.util.concurrent.Executor;

@Configuration
public class AsyncConfig {
    
    @Bean(name = "asyncExecutor")
    public Executor asyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);           // 核心线程数
        executor.setMaxPoolSize(10);          // 最大线程数
        executor.setQueueCapacity(100);       // 队列容量
        executor.setThreadNamePrefix("async-");
        executor.setAwaitTerminationSeconds(60);
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.initialize();
        return executor;
    }
}

@Service
@Slf4j
public class AsyncTaskService {
    
    @Async("asyncExecutor")
    public void asyncTaskWithCustomExecutor(String taskName) {
        log.info("使用自定义执行器执行异步任务：{}", taskName);
    }
}
```

## 异步异常处理

```java
import org.springframework.scheduling.annotation.Async;
import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.AsyncConfigurer;
import org.springframework.context.annotation.Bean;
import java.util.concurrent.Executor;

@Configuration
public class AsyncExceptionConfig implements AsyncConfigurer {
    
    @Override
    public Executor getAsyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setThreadNamePrefix("async-");
        executor.initialize();
        return executor;
    }
    
    @Override
    public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
        return new CustomAsyncExceptionHandler();
    }
}

@Slf4j
public class CustomAsyncExceptionHandler implements AsyncUncaughtExceptionHandler {
    
    @Override
    public void handleUncaughtException(Throwable throwable, Method method, Object... params) {
        log.error("异步任务异常 - 方法：{}，参数：{}", method.getName(), params, throwable);
        // 发送告警、记录日志等
    }
}
```

## CompletableFuture 组合

```java
@Service
@Slf4j
public class ComplexAsyncService {
    
    // 组合多个异步任务
    public CompletableFuture<String> complexAsyncFlow() {
        return CompletableFuture
            .supplyAsync(() -> {
                log.info("步骤 1: 获取用户数据");
                return "用户数据";
            })
            .thenApply(userData -> {
                log.info("步骤 2: 处理用户数据");
                return userData + " - 已处理";
            })
            .thenApply(processedData -> {
                log.info("步骤 3: 保存结果");
                return processedData + " - 已保存";
            })
            .exceptionally(ex -> {
                log.error("任务执行异常", ex);
                return "默认返回值";
            });
    }
    
    // 并行执行多个任务
    public CompletableFuture<Void> parallelTasks() {
        CompletableFuture<String> task1 = CompletableFuture.supplyAsync(() -> "任务1完成");
        CompletableFuture<String> task2 = CompletableFuture.supplyAsync(() -> "任务2完成");
        CompletableFuture<String> task3 = CompletableFuture.supplyAsync(() -> "任务3完成");
        
        // 等待所有任务完成
        return CompletableFuture.allOf(task1, task2, task3);
    }
    
    // 任意一个任务完成则继续
    public CompletableFuture<Object> anyOfTasks() {
        CompletableFuture<String> task1 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
                return "任务1完成";
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });
        
        CompletableFuture<String> task2 = CompletableFuture.supplyAsync(() -> "任务2快速完成");
        
        return CompletableFuture.anyOf(task1, task2);
    }
}
```

## 线程池配置最佳实践

```java
@Configuration
public class ThreadPoolConfig {
    
    @Bean
    public Executor asyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        
        // 核心线程数：建议为 CPU 核心数
        int coreSize = Runtime.getRuntime().availableProcessors();
        executor.setCorePoolSize(coreSize);
        
        // 最大线程数：通常为核心线程数的 2 倍
        executor.setMaxPoolSize(coreSize * 2);
        
        // 队列容量：过小会频繁创建线程，过大会占用内存
        executor.setQueueCapacity(coreSize * 100);
        
        // 线程名前缀
        executor.setThreadNamePrefix("async-pool-");
        
        // 拒绝策略：当队列满时如何处理
        executor.setRejectedExecutionHandler(
            new ThreadPoolTaskExecutor.CallerRunsPolicy()
        );
        
        // 优雅关闭
        executor.setAwaitTerminationSeconds(60);
        executor.setWaitForTasksToCompleteOnShutdown(true);
        
        executor.initialize();
        return executor;
    }
}
```

## 定时任务监控

```java
@Service
@Slf4j
public class TaskMonitorService {
    
    private final MeterRegistry meterRegistry;
    private AtomicInteger successCount = new AtomicInteger(0);
    private AtomicInteger failureCount = new AtomicInteger(0);
    
    @Autowired
    public TaskMonitorService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        
        // 注册指标
        meterRegistry.gauge("task.success.count", successCount);
        meterRegistry.gauge("task.failure.count", failureCount);
    }
    
    @Scheduled(fixedRate = 5000)
    public void monitoredTask() {
        try {
            log.info("执行受监控的任务");
            successCount.incrementAndGet();
        } catch (Exception e) {
            log.error("任务执行失败", e);
            failureCount.incrementAndGet();
        }
    }
}
```

## 总结

- **定时任务** - 使用 @Scheduled 处理定期执行的任务
- **异步处理** - 使用 @Async 处理耗时的异步操作
- **CompletableFuture** - 更灵活地组合和管理异步任务
- **线程池** - 合理配置线程池参数以获得最佳性能
- **异常处理** - 为异步任务添加完善的异常处理机制
- **监控** - 使用指标监控定时任务和异步任务的执行情况

下一步学习 [安全认证](./security.md)。
