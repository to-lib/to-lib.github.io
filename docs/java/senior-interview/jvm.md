---
sidebar_position: 2
title: JVM 深度
---

# 🎯 JVM 深度（高级）

## 1. 详细描述 JVM 内存模型，各区域的作用和特点？

**答案要点：**

**JDK 8+ 运行时数据区域：**

| 区域 | 线程共享 | 作用 | 异常 |
|------|---------|------|------|
| **堆（Heap）** | 共享 | 存储对象实例和数组 | OutOfMemoryError |
| **方法区/元空间** | 共享 | 存储类信息、常量、静态变量 | OutOfMemoryError |
| **虚拟机栈** | 私有 | 存储局部变量、操作数栈、方法出口 | StackOverflowError/OOM |
| **本地方法栈** | 私有 | 为 native 方法服务 | StackOverflowError/OOM |
| **程序计数器** | 私有 | 记录当前执行的字节码指令地址 | 无 |

**堆内存分代结构：**

```
堆内存
├── 新生代（Young Generation）- 1/3 堆
│   ├── Eden 区 - 8/10 新生代
│   ├── Survivor From (S0) - 1/10 新生代
│   └── Survivor To (S1) - 1/10 新生代
└── 老年代（Old Generation）- 2/3 堆
```

**代码示例 - 查看内存分配：**

```java
public class MemoryDemo {
    public static void main(String[] args) {
        // 堆内存信息
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();      // 最大堆内存
        long totalMemory = runtime.totalMemory();  // 当前堆内存
        long freeMemory = runtime.freeMemory();    // 空闲堆内存
        
        System.out.println("Max: " + maxMemory / 1024 / 1024 + "MB");
        System.out.println("Total: " + totalMemory / 1024 / 1024 + "MB");
        System.out.println("Free: " + freeMemory / 1024 / 1024 + "MB");
    }
}
```

**JVM 参数配置：**

```bash
# 堆内存配置
-Xms512m          # 初始堆大小
-Xmx1024m         # 最大堆大小
-Xmn256m          # 新生代大小

# 元空间配置（JDK 8+）
-XX:MetaspaceSize=128m
-XX:MaxMetaspaceSize=256m

# 栈大小配置
-Xss256k          # 每个线程栈大小
```

**延伸：** 参考 [JVM 基础 - 内存模型](/docs/java/jvm-basics#内存模型)

---

## 2. 对比 CMS、G1、ZGC 垃圾回收器的特点和适用场景？

**答案要点：**

| 特性 | CMS | G1 | ZGC |
|------|-----|----|----|
| **算法** | 标记-清除 | 标记-整理 | 染色指针+读屏障 |
| **停顿时间** | 不可预测 | 可预测（-XX:MaxGCPauseMillis） | <10ms |
| **内存碎片** | 有 | 无 | 无 |
| **堆大小** | <32GB | 4GB-64GB | 8MB-16TB |
| **JDK版本** | JDK 5+ | JDK 7+ | JDK 11+ |
| **适用场景** | 低延迟、中小堆 | 大堆、可控停顿 | 超大堆、极低延迟 |

**G1 收集器工作原理：**

```
G1 堆内存布局（Region 化）
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ Eden│ Eden│ Sur │ Old │ Old │ Hum │
├─────┼─────┼─────┼─────┼─────┼─────┤
│ Old │ Free│ Eden│ Old │ Sur │ Old │
└─────┴─────┴─────┴─────┴─────┴─────┘
每个 Region 大小：1MB-32MB（2的幂次）
```

**GC 选择建议：**

```bash
# CMS（JDK 8 默认可用，JDK 14 移除）
-XX:+UseConcMarkSweepGC

# G1（JDK 9+ 默认）
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200

# ZGC（JDK 15+ 生产可用）
-XX:+UseZGC
-XX:+ZGenerational  # JDK 21+ 分代 ZGC
```

**延伸：** 参考 [JVM 基础 - 垃圾回收](/docs/java/jvm-basics#垃圾回收gc)

---

## 3. 如何进行 GC 调优？请描述一个实际的调优案例

**答案要点：**

**GC 调优步骤：**

1. **收集 GC 日志**
2. **分析 GC 行为**
3. **调整 JVM 参数**
4. **验证优化效果**

**开启 GC 日志：**

```bash
# JDK 8
-XX:+PrintGCDetails
-XX:+PrintGCDateStamps
-Xloggc:/path/to/gc.log

# JDK 9+
-Xlog:gc*:file=/path/to/gc.log:time,uptime,level,tags
```

**实际调优案例 - Full GC 频繁：**

```bash
# 问题现象：每隔几分钟发生 Full GC，停顿 2-3 秒

# 原始配置
-Xms2g -Xmx2g -Xmn512m

# 分析发现：
# 1. 新生代太小，对象过早晋升到老年代
# 2. 老年代很快被填满，触发 Full GC

# 优化后配置
-Xms4g -Xmx4g -Xmn1536m
-XX:SurvivorRatio=8
-XX:MaxTenuringThreshold=15
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
```

**GC 日志分析关键指标：**

```
[GC (Allocation Failure) [PSYoungGen: 524288K->87654K(611840K)] 
 524288K->87654K(2010112K), 0.0876543 secs]
 
关键指标：
- GC 原因：Allocation Failure
- Young GC 前后：524288K -> 87654K
- 堆总量变化：524288K -> 87654K
- GC 耗时：0.0876543 秒
```

**延伸：** 参考 [JVM 基础 - 性能调优](/docs/java/jvm-basics)

---

## 4. 解释类加载机制和双亲委派模型，如何打破双亲委派？

**答案要点：**

**类加载过程：**

```
加载 → 验证 → 准备 → 解析 → 初始化 → 使用 → 卸载
```

**双亲委派模型：**

```
                    ┌─────────────────┐
                    │ Bootstrap       │ 加载 rt.jar
                    │ ClassLoader     │ (C++ 实现)
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Extension       │ 加载 ext/*.jar
                    │ ClassLoader     │
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Application     │ 加载 classpath
                    │ ClassLoader     │
                    └────────┬────────┘
                             │ 委派
                    ┌────────▼────────┐
                    │ Custom          │ 自定义加载
                    │ ClassLoader     │
                    └─────────────────┘
```

**打破双亲委派的场景：**

1. **SPI 机制**（JDBC、JNDI）
2. **热部署**（Tomcat、OSGi）
3. **代码隔离**（不同版本类库）

**自定义类加载器示例：**

```java
public class HotSwapClassLoader extends ClassLoader {
    
    @Override
    protected Class<?> loadClass(String name, boolean resolve) 
            throws ClassNotFoundException {
        // 打破双亲委派：先尝试自己加载
        if (name.startsWith("com.myapp.")) {
            return findClass(name);
        }
        // 其他类仍走双亲委派
        return super.loadClass(name, resolve);
    }
    
    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        byte[] classData = loadClassData(name);
        if (classData == null) {
            throw new ClassNotFoundException(name);
        }
        return defineClass(name, classData, 0, classData.length);
    }
    
    private byte[] loadClassData(String name) {
        // 从文件/网络加载类字节码
        String path = name.replace('.', '/') + ".class";
        try (InputStream is = new FileInputStream(path)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int len;
            while ((len = is.read(buffer)) != -1) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        } catch (IOException e) {
            return null;
        }
    }
}
```

**延伸：** 参考 [JVM 基础 - 类加载机制](/docs/java/jvm-basics#类加载机制)

---

## 5. JIT 编译器有哪些优化技术？什么是逃逸分析？

**答案要点：**

**JIT 主要优化技术：**

| 优化技术 | 说明 | 效果 |
|---------|------|------|
| **方法内联** | 将小方法代码直接嵌入调用处 | 减少方法调用开销 |
| **逃逸分析** | 分析对象作用域 | 栈上分配、锁消除 |
| **锁消除** | 消除不必要的同步 | 提升并发性能 |
| **锁粗化** | 合并连续的加锁操作 | 减少锁开销 |
| **标量替换** | 将对象拆解为基本类型 | 减少内存分配 |

**逃逸分析详解：**

```java
// 不逃逸 - 可以栈上分配
public void noEscape() {
    Point p = new Point(1, 2);  // 对象只在方法内使用
    System.out.println(p.x + p.y);
}

// 方法逃逸 - 不能栈上分配
public Point methodEscape() {
    Point p = new Point(1, 2);
    return p;  // 对象被返回，逃逸到方法外
}

// 线程逃逸 - 不能栈上分配
public void threadEscape() {
    Point p = new Point(1, 2);
    new Thread(() -> System.out.println(p)).start();  // 被其他线程访问
}
```

**锁消除示例：**

```java
// JIT 会消除这个同步，因为 sb 不会逃逸
public String concat(String s1, String s2) {
    StringBuffer sb = new StringBuffer();  // 局部变量，不逃逸
    sb.append(s1);
    sb.append(s2);
    return sb.toString();
}
// 优化后等价于使用 StringBuilder（无同步）
```

**JIT 相关 JVM 参数：**

```bash
# 开启/关闭逃逸分析
-XX:+DoEscapeAnalysis    # 默认开启
-XX:-DoEscapeAnalysis    # 关闭

# 开启/关闭锁消除
-XX:+EliminateLocks      # 默认开启

# 开启/关闭标量替换
-XX:+EliminateAllocations

# 查看 JIT 编译日志
-XX:+PrintCompilation
-XX:+UnlockDiagnosticVMOptions
-XX:+PrintInlining
```

**延伸：** 参考 [JVM 基础 - JIT 编译](/docs/java/jvm-basics)
