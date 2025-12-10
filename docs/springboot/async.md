---
sidebar_position: 17
---

# 异步处理

> [!TIP]
> **异步的价值**: 异步处理可以显著提升应用性能和响应速度，特别是在处理耗时操作（如发送邮件、调用外部API、大数据处理）时，避免阻塞主线程。

## 启用异步支持

### 添加 @EnableAsync

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

@Configuration
@EnableAsync
public class AsyncConfig {
}
```

## @Async 基本用法

### 异步方法

```java
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class EmailService {
    
    @Async
    public void sendEmail(String to, String subject, String content) {
        log.info("开始发送邮件到: {}, 线程: {}", to, Thread.currentThread().getName());
        
        try {
            // 模拟耗时操作
            Thread.sleep(3000);
            // 实际发送邮件逻辑
            log.info("邮件发送成功: {}", to);
        } catch (InterruptedException e) {
            log.error("邮件发送失败", e);
        }
        
        log.info("邮件发送完成, 线程: {}", Thread.currentThread().getName());
    }
    
    @Async
    public void sendBulkEmails(List<String> recipients) {
        log.info("批量发送邮件, 共{}个收件人", recipients.size());
        
        for (String recipient : recipients) {
            sendSingleEmail(recipient);
        }
    }
    
    private void sendSingleEmail(String recipient) {
        // 发送单个邮件
    }
}
```

### 使用异步方法

```java
@RestController
@RequestMapping("/api")
public class UserController {
    
    @Autowired
    private EmailService emailService;
    
    @PostMapping("/register")
    public ResponseEntity<String> register(@RequestBody User user) {
        // 1. 保存用户
        userService.save(user);
        
        // 2. 异步发送欢迎邮件（不阻塞）
        emailService.sendEmail(
            user.getEmail(),
            "欢迎注册",
            "欢迎加入我们的平台！"
        );
        
        // 3. 立即返回响应，不等待邮件发送完成
        return ResponseEntity.ok("注册成功");
    }
}
```

## 带返回值的异步方法

### Future

```java
import java.util.concurrent.Future;

@Service
@Slf4j
public class DataService {
    
    @Async
    public Future<String> processDataAsync() {
        log.info("开始处理数据, 线程: {}", Thread.currentThread().getName());
        
        try {
            Thread.sleep(2000);
            String result = "处理完成的数据";
            return new AsyncResult<>(result);
        } catch (InterruptedException e) {
            return new AsyncResult<>("处理失败");
        }
    }
}

// 使用
@Service
public class BusinessService {
    
    @Autowired
    private DataService dataService;
    
    public void doSomething() throws Exception {
        Future<String> future = dataService.processDataAsync();
        
        // 做其他事情
        log.info("继续执行其他任务...");
        
        // 获取异步结果（阻塞直到完成）
        String result = future.get();
        log.info("异步结果: {}", result);
        
        // 带超时的获取
        String resultWithTimeout = future.get(5, TimeUnit.SECONDS);
    }
}
```

### CompletableFuture（推荐）

```java
import java.util.concurrent.CompletableFuture;

@Service
@Slf4j
public class DataService {
    
    @Async
    public CompletableFuture<String> processDataAsync() {
        log.info("开始处理数据, 线程: {}", Thread.currentThread().getName());
        
        try {
            Thread.sleep(2000);
            String result = "处理完成的数据";
            return CompletableFuture.completedFuture(result);
        } catch (InterruptedException e) {
            CompletableFuture<String> future = new CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }
    
    @Async
    public CompletableFuture<User> getUserAsync(Long id) {
        User user = userRepository.findById(id).orElse(null);
        return CompletableFuture.completedFuture(user);
    }
    
    @Async
    public CompletableFuture<List<Order>> getOrdersAsync(Long userId) {
        List<Order> orders = orderRepository.findByUserId(userId);
        return CompletableFuture.completedFuture(orders);
    }
}
```

### 组合多个异步操作

```java
@Service
@Slf4j
public class UserProfileService {
    
    @Autowired
    private DataService dataService;
    
    public UserProfile getUserProfile(Long userId) {
        // 并行执行多个异步操作
        CompletableFuture<User> userFuture = dataService.getUserAsync(userId);
        CompletableFuture<List<Order>> ordersFuture = dataService.getOrdersAsync(userId);
        CompletableFuture<Statistics> statsFuture = dataService.getStatisticsAsync(userId);
        
        // 等待所有操作完成
        CompletableFuture.allOf(userFuture, ordersFuture, statsFuture).join();
        
        try {
            User user = userFuture.get();
            List<Order> orders = ordersFuture.get();
            Statistics stats = statsFuture.get();
            
            return new UserProfile(user, orders, stats);
        } catch (Exception e) {
            log.error("获取用户档案失败", e);
            return null;
        }
    }
    
    // 使用 thenCombine 组合结果
    public CompletableFuture<UserProfile> getUserProfileAsync(Long userId) {
        CompletableFuture<User> userFuture = dataService.getUserAsync(userId);
        CompletableFuture<List<Order>> ordersFuture = dataService.getOrdersAsync(userId);
        
        return userFuture.thenCombine(ordersFuture, (user, orders) -> {
            return new UserProfile(user, orders);
        });
    }
    
    // 链式调用
    public CompletableFuture<String> processUserChain(Long userId) {
        return dataService.getUserAsync(userId)
            .thenApply(user -> {
                // 处理用户数据
                log.info("处理用户: {}", user.getName());
                return user;
            })
            .thenCompose(user -> {
                // 根据用户获取订单
                return dataService.getOrdersAsync(user.getId());
            })
            .thenApply(orders -> {
                // 处理订单数据
                return "用户有 " + orders.size() + " 个订单";
            })
            .exceptionally(ex -> {
                // 异常处理
                log.error("处理失败", ex);
                return "处理失败";
            });
    }
}
```

## 线程池配置

### 自定义线程池

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

@Configuration
@EnableAsync
public class AsyncConfig {
    
    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        
        // 核心线程数
        executor.setCorePoolSize(10);
        
        // 最大线程数
        executor.setMaxPoolSize(20);
        
        // 队列容量
        executor.setQueueCapacity(100);
        
        // 线程名称前缀
        executor.setThreadNamePrefix("async-");
        
        // 拒绝策略
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        
        // 等待所有任务完成后再关闭线程池
        executor.setWaitForTasksToCompleteOnShutdown(true);
        
        // 等待时间
        executor.setAwaitTerminationSeconds(60);
        
        executor.initialize();
        return executor;
    }
    
    // 创建多个线程池用于不同场景
    @Bean(name = "emailExecutor")
    public Executor emailExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(50);
        executor.setThreadNamePrefix("email-");
        executor.initialize();
        return executor;
    }
    
    @Bean(name = "reportExecutor")
    public Executor reportExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(2);
        executor.setMaxPoolSize(5);
        executor.setQueueCapacity(20);
        executor.setThreadNamePrefix("report-");
        executor.initialize();
        return executor;
    }
}
```

### 使用指定线程池

```java
@Service
@Slf4j
public class MultiThreadPoolService {
    
    // 使用 emailExecutor 线程池
    @Async("emailExecutor")
    public void sendEmail(String to) {
        log.info("发送邮件, 线程池: emailExecutor, 线程: {}", 
            Thread.currentThread().getName());
        // 邮件发送逻辑
    }
    
    // 使用 reportExecutor 线程池
    @Async("reportExecutor")
    public CompletableFuture<Report> generateReport() {
        log.info("生成报表, 线程池: reportExecutor, 线程: {}", 
            Thread.currentThread().getName());
        // 报表生成逻辑
        return CompletableFuture.completedFuture(new Report());
    }
    
    // 使用默认线程池 taskExecutor
    @Async
    public void processData() {
        log.info("处理数据, 线程池: taskExecutor, 线程: {}", 
            Thread.currentThread().getName());
        // 数据处理逻辑
    }
}
```

### 配置文件方式

```yaml
spring:
  task:
    execution:
      pool:
        core-size: 10
        max-size: 20
        queue-capacity: 100
        keep-alive: 60s
      thread-name-prefix: async-task-
```

## 异常处理

### AsyncUncaughtExceptionHandler

```java
import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.AsyncConfigurerSupport;
import org.springframework.scheduling.annotation.EnableAsync;
import java.lang.reflect.Method;
import java.util.concurrent.Executor;

@Configuration
@EnableAsync
public class AsyncConfig extends AsyncConfigurerSupport {
    
    @Override
    public Executor getAsyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(20);
        executor.setQueueCapacity(100);
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
    public void handleUncaughtException(Throwable ex, Method method, Object... params) {
        log.error("异步方法执行异常 - 方法: {}, 参数: {}, 异常: {}", 
            method.getName(), 
            Arrays.toString(params), 
            ex.getMessage(), 
            ex);
        
        // 可以发送告警、记录到数据库等
        sendAlertEmail(method, ex);
    }
    
    private void sendAlertEmail(Method method, Throwable ex) {
        // 发送告警邮件
    }
}
```

### 方法内异常处理

```java
@Service
@Slf4j
public class RobustAsyncService {
    
    @Async
    public CompletableFuture<String> processWithErrorHandling() {
        try {
            // 业务逻辑
            String result = doSomething();
            return CompletableFuture.completedFuture(result);
            
        } catch (Exception e) {
            log.error("异步处理失败", e);
            
            CompletableFuture<String> future = new CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }
    
    @Async
    public void processWithRetry() {
        int maxRetries = 3;
        int attempt = 0;
        
        while (attempt < maxRetries) {
            try {
                attempt++;
                doSomething();
                log.info("处理成功, 尝试次数: {}", attempt);
                return;
                
            } catch (Exception e) {
                log.warn("处理失败, 尝试次数: {}, 异常: {}", attempt, e.getMessage());
                
                if (attempt >= maxRetries) {
                    log.error("达到最大重试次数, 处理失败", e);
                    // 记录失败，发送告警等
                } else {
                    try {
                        Thread.sleep(1000 * attempt); // 递增等待时间
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }
    }
    
    private String doSomething() throws Exception {
        // 业务逻辑
        return "result";
    }
}
```

## 异步与事务

> [!WARNING]
> **异步方法中的事务**: @Async 方法在独立线程中执行，与调用者的事务是分离的。需要特别注意事务边界。

```java
@Service
@Slf4j
public class AsyncTransactionService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private EmailService emailService;
    
    // ❌ 错误示例：异步方法与外部事务分离
    @Transactional
    public void createOrder(Order order) {
        // 1. 保存订单（在事务中）
        orderRepository.save(order);
        
        // 2. 异步发送邮件
        emailService.sendOrderConfirmationAsync(order);
        // 注意：此时订单可能还未提交到数据库
        // 如果邮件发送很快，可能读取不到订单
    }
    
    // ✓ 正确示例1：确保事务提交后再调用异步方法
    @Transactional
    public void createOrderCorrect1(Order order) {
        // 1. 保存订单
        orderRepository.save(order);
        
        // 2. 注册事务提交后的回调
        TransactionSynchronizationManager.registerSynchronization(
            new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    // 事务提交后再发送邮件
                    emailService.sendOrderConfirmationAsync(order);
                }
            }
        );
    }
    
    // ✓ 正确示例2：使用 @TransactionalEventListener
    @Transactional
    public void createOrderCorrect2(Order order) {
        orderRepository.save(order);
        
        // 发布事件
        applicationEventPublisher.publishEvent(new OrderCreatedEvent(order));
    }
}

@Service
public class EmailService {
    
    // 监听事件，在事务提交后执行
    @Async
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void handleOrderCreated(OrderCreatedEvent event) {
        sendOrderConfirmation(event.getOrder());
    }
}

// 异步方法需要自己的事务
@Service
public class AsyncDatabaseService {
    
    @Async
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveLogAsync(AuditLog log) {
        // 使用独立事务保存日志
        auditLogRepository.save(log);
    }
}
```

## 实战场景

### 1. 并行处理提升性能

```java
@Service
@Slf4j
public class ProductService {
    
    @Async
    public CompletableFuture<ProductDetails> getProductDetails(Long productId) {
        return CompletableFuture.completedFuture(
            productRepository.findById(productId).orElse(null)
        );
    }
    
    @Async
    public CompletableFuture<List<Review>> getProductReviews(Long productId) {
        return CompletableFuture.completedFuture(
            reviewRepository.findByProductId(productId)
        );
    }
    
    @Async
    public CompletableFuture<Inventory> getProductInventory(Long productId) {
        return CompletableFuture.completedFuture(
            inventoryRepository.findByProductId(productId)
        );
    }
    
    public ProductPage getProductPage(Long productId) {
        long startTime = System.currentTimeMillis();
        
        // 并行获取数据
        CompletableFuture<ProductDetails> detailsFuture = getProductDetails(productId);
        CompletableFuture<List<Review>> reviewsFuture = getProductReviews(productId);
        CompletableFuture<Inventory> inventoryFuture = getProductInventory(productId);
        
        // 等待所有操作完成
        CompletableFuture.allOf(detailsFuture, reviewsFuture, inventoryFuture).join();
        
        try {
            ProductPage page = new ProductPage(
                detailsFuture.get(),
                reviewsFuture.get(),
                inventoryFuture.get()
            );
            
            long duration = System.currentTimeMillis() - startTime;
            log.info("商品页面数据加载完成, 耗时: {}ms", duration);
            
            return page;
        } catch (Exception e) {
            log.error("获取商品页面数据失败", e);
            return null;
        }
    }
}
```

### 2. 批量异步处理

```java
@Service
@Slf4j
public class BatchService {
    
    @Async
    public CompletableFuture<Void> processItemAsync(Item item) {
        try {
            // 处理单个项目
            processItem(item);
            return CompletableFuture.completedFuture(null);
        } catch (Exception e) {
            CompletableFuture<Void> future = new CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }
    
    public void batchProcess(List<Item> items) {
        log.info("开始批量处理, 共{}个项目", items.size());
        
        // 创建异步任务列表
        List<CompletableFuture<Void>> futures = items.stream()
            .map(this::processItemAsync)
            .collect(Collectors.toList());
        
        // 等待所有任务完成
        CompletableFuture<Void> allOf = CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0])
        );
        
        allOf.join();
        
        // 统计结果
        long successCount = futures.stream()
            .filter(f -> !f.isCompletedExceptionally())
            .count();
        
        log.info("批量处理完成, 成功: {}/{}", successCount, items.size());
    }
    
    private void processItem(Item item) {
        // 处理逻辑
    }
}
```

### 3. 定时异步任务

```java
@Service
@Slf4j
public class ScheduledAsyncService {
    
    @Async
    @Scheduled(fixedRate = 60000) // 每分钟执行一次
    public void cleanupExpiredData() {
        log.info("开始清理过期数据, 线程: {}", Thread.currentThread().getName());
        
        try {
            int deletedCount = dataRepository.deleteExpiredData();
            log.info("清理完成, 删除{}条数据", deletedCount);
        } catch (Exception e) {
            log.error("清理过期数据失败", e);
        }
    }
    
    @Async
    @Scheduled(cron = "0 0 2 * * ?") // 每天凌晨2点执行
    public void generateDailyReport() {
        log.info("开始生成日报, 线程: {}", Thread.currentThread().getName());
        
        try {
            Report report = reportService.generateDailyReport();
            reportService.saveReport(report);
            emailService.sendReport(report);
            log.info("日报生成完成");
        } catch (Exception e) {
            log.error("生成日报失败", e);
        }
    }
}
```

## 最佳实践

> [!TIP]
> **异步处理最佳实践**：
>
> 1. **合理配置线程池** - 根据任务类型和系统资源配置合适的线程池大小
> 2. **避免滥用** - 不是所有方法都需要异步，评估性能收益
> 3. **异常处理** - 必须处理异步方法中的异常
> 4. **监控线程池** - 监控线程池状态，防止线程耗尽
> 5. **事务边界** - 注意异步方法与事务的分离
> 6. **避免自调用** - 同一个类内部调用 @Async 方法不生效
> 7. **返回值使用 CompletableFuture** - 比 Future 更强大灵活

### 线程池监控

```java
@Component
@Slf4j
public class ThreadPoolMonitor {
    
    @Autowired
    @Qualifier("taskExecutor")
    private ThreadPoolTaskExecutor executor;
    
    @Scheduled(fixedRate = 60000) // 每分钟记录一次
    public void monitorThreadPool() {
        ThreadPoolExecutor threadPool = executor.getThreadPoolExecutor();
        
        log.info("线程池状态 - " +
            "核心线程数: {}, " +
            "最大线程数: {}, " +
            "当前线程数: {}, " +
            "活跃线程数: {}, " +
            "队列大小: {}, " +
            "已完成任务数: {}",
            threadPool.getCorePoolSize(),
            threadPool.getMaximumPoolSize(),
            threadPool.getPoolSize(),
            threadPool.getActiveCount(),
            threadPool.getQueue().size(),
            threadPool.getCompletedTaskCount()
        );
        
        // 告警：队列接近满
        if (threadPool.getQueue().size() > 80) {
            log.warn("线程池队列接近满! 当前大小: {}", threadPool.getQueue().size());
        }
        
        // 告警：活跃线程接近最大值
        if (threadPool.getActiveCount() >= threadPool.getMaximumPoolSize() * 0.9) {
            log.warn("活跃线程数接近最大值! 当前: {}/{}", 
                threadPool.getActiveCount(), 
                threadPool.getMaximumPoolSize());
        }
    }
}
```

## 总结

- **@Async** - 异步方法调用，提升性能
- **CompletableFuture** - 强大的异步编程工具
- **线程池配置** - 自定义线程池，隔离不同类型的任务
- **异常处理** - AsyncUncaughtExceptionHandler 处理未捕获异常
- **异步与事务** - 注意事务边界，使用事件监听器
- **实战场景** - 并行处理、批量处理、定时任务

下一步学习 [事件机制](./events)。
