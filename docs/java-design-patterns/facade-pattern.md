---
sidebar_position: 13
---

# 外观模式 (Facade Pattern)

## 模式定义

**外观模式**是一种结构型设计模式，它为复杂的子系统提供一个统一的、简单的界面，隐藏系统的复杂性。

## 问题分析

当一个系统由多个复杂的子系统组成时，直接使用这些子系统会导致：

- 客户端代码复杂
- 需要了解多个类的接口
- 系统之间的依赖关系复杂
- 难以维护和扩展

## 解决方案

提供一个简单的Facade类来封装复杂的子系统：

```
┌─────────────────────┐
│     Facade         │
│  (简化接口)        │
└──────────┬──────────┘
           │
    ┌──────┴──────────┐
    │                 │
┌────────┐      ┌────────┐
│SubSystem│      │SubSystem│
│    1    │      │    2    │
└────────┘      └────────┘
    │                 │
    └────────┬────────┘
           │
    ┌────────────────┐
    │  SubSystem 3   │
    └────────────────┘
```

## 代码实现

### 1. 定义复杂的子系统

```java
public class CPU {
    public void freeze() {
        System.out.println("CPU冻结");
    }
    
    public void jump(long position) {
        System.out.println("CPU跳转到位置: " + position);
    }
    
    public void execute() {
        System.out.println("CPU执行");
    }
}

public class Memory {
    public void load(long position, byte[] data) {
        System.out.println("内存在位置 " + position + " 加载数据");
    }
}

public class HardDrive {
    public byte[] read(long lba, int size) {
        System.out.println("从硬盘读取数据: LBA=" + lba + ", 大小=" + size);
        return new byte[size];
    }
}

public class Display {
    public void displayBIOS() {
        System.out.println("显示BIOS");
    }
    
    public void displayLoading() {
        System.out.println("显示加载界面");
    }
}
```

### 2. 创建外观类

```java
public class ComputerFacade {
    private CPU cpu;
    private Memory memory;
    private HardDrive hardDrive;
    private Display display;
    
    public ComputerFacade() {
        this.cpu = new CPU();
        this.memory = new Memory();
        this.hardDrive = new HardDrive();
        this.display = new Display();
    }
    
    // 简化的启动方法
    public void start() {
        System.out.println("=== 计算机启动 ===");
        display.displayBIOS();
        cpu.freeze();
        memory.load(0, hardDrive.read(0, 1024));
        cpu.jump(0);
        cpu.execute();
        display.displayLoading();
        System.out.println("启动完成");
    }
    
    // 简化的关闭方法
    public void shutdown() {
        System.out.println("=== 计算机关闭 ===");
        cpu.freeze();
    }
}
```

### 3. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        ComputerFacade computer = new ComputerFacade();
        
        // 用户只需要调用简单的方法
        computer.start();
        // 使用计算机...
        computer.shutdown();
    }
}
```

## 实际应用示例

### 视频转换外观

```java
public class VideoFile {
    private String name;
    
    public VideoFile(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
}

public class Codec {
}

public class MPEG4CompressionCodec extends Codec {
}

public class OggCompressionCodec extends Codec {
}

public class CodecFactory {
    public static Codec extract(VideoFile file) {
        String name = file.getName();
        if (name.endsWith(".mp4")) {
            return new MPEG4CompressionCodec();
        }
        return new OggCompressionCodec();
    }
}

public class BitrateReader {
    public static VideoFile read(VideoFile file, Codec codec) {
        System.out.println("使用 " + codec.getClass().getSimpleName() + " 读取文件");
        return file;
    }
}

public class AudioMixer {
    public File fix(VideoFile result) {
        System.out.println("修复音频...");
        return new File("output_" + result.getName());
    }
}

// 外观类
public class VideoConversionFacade {
    public File convertVideo(String fileName, String format) {
        System.out.println("VideoConversionFacade: 转换视频 " + fileName);
        
        VideoFile file = new VideoFile(fileName);
        Codec codec = CodecFactory.extract(file);
        VideoFile videoWithAudio = BitrateReader.read(file, codec);
        File result = new AudioMixer().fix(videoWithAudio);
        
        System.out.println("VideoConversionFacade: 转换完成");
        return result;
    }
}

// 使用
VideoConversionFacade converter = new VideoConversionFacade();
File video = converter.convertVideo("movie.mp4", "avi");
```

### 数据库查询外观

```java
public class SQLBuilder {
    private StringBuilder sql = new StringBuilder();
    
    public SQLBuilder select(String... columns) {
        sql.append("SELECT ").append(String.join(", ", columns));
        return this;
    }
    
    public SQLBuilder from(String table) {
        sql.append(" FROM ").append(table);
        return this;
    }
    
    public SQLBuilder where(String condition) {
        sql.append(" WHERE ").append(condition);
        return this;
    }
    
    public String build() {
        return sql.toString();
    }
}

public class ConnectionPool {
    public Connection getConnection() {
        System.out.println("获取数据库连接");
        return null;
    }
}

public class QueryExecutor {
    public List<Map<String, Object>> execute(String sql) {
        System.out.println("执行SQL: " + sql);
        return new ArrayList<>();
    }
}

// 外观类
public class DatabaseFacade {
    private ConnectionPool pool = new ConnectionPool();
    private QueryExecutor executor = new QueryExecutor();
    
    public List<Map<String, Object>> query(String table, String... columns) {
        Connection conn = pool.getConnection();
        String sql = new SQLBuilder()
            .select(columns)
            .from(table)
            .build();
        return executor.execute(sql);
    }
}

// 使用
DatabaseFacade db = new DatabaseFacade();
List<Map<String, Object>> users = db.query("users", "id", "name", "email");
```

### 家居智能控制外观

```java
public class Light {
    public void on() {
        System.out.println("灯打开");
    }
    
    public void off() {
        System.out.println("灯关闭");
    }
}

public class AirConditioner {
    public void on() {
        System.out.println("空调打开");
    }
    
    public void setTemperature(int temp) {
        System.out.println("空调温度设置为: " + temp);
    }
}

public class Television {
    public void on() {
        System.out.println("电视打开");
    }
    
    public void setChannel(int channel) {
        System.out.println("电视切换到频道: " + channel);
    }
}

public class Door {
    public void lock() {
        System.out.println("门锁定");
    }
}

// 外观类
public class SmartHomeFacade {
    private Light light;
    private AirConditioner ac;
    private Television tv;
    private Door door;
    
    public SmartHomeFacade() {
        this.light = new Light();
        this.ac = new AirConditioner();
        this.tv = new Television();
        this.door = new Door();
    }
    
    public void leavingHome() {
        System.out.println("=== 离家模式 ===");
        light.off();
        ac.on();
        tv.on();
        door.lock();
    }
    
    public void welcomeHome() {
        System.out.println("=== 欢迎回家 ===");
        light.on();
        ac.setTemperature(24);
        tv.setChannel(1);
    }
    
    public void goToSleep() {
        System.out.println("=== 睡眠模式 ===");
        light.off();
        ac.setTemperature(20);
        door.lock();
    }
}

// 使用
SmartHomeFacade home = new SmartHomeFacade();
home.leavingHome();
home.welcomeHome();
home.goToSleep();
```

## 外观模式 vs 其他模式

| 模式 | 目的 | 职责 |
|------|------|------|
| 外观 | 简化复杂系统 | 简化接口 |
| 适配器 | 转换接口 | 兼容不同接口 |
| 代理 | 控制访问 | 增强功能 |
| 装饰器 | 添加功能 | 增强对象 |

## 优缺点

### 优点
- ✅ 简化客户端使用
- ✅ 降低系统耦合
- ✅ 隐藏实现细节
- ✅ 易于维护和修改

### 缺点
- ❌ 违反接口隔离原则
- ❌ 增加了间接性
- ❌ 可能造成功能不完全

## 适用场景

- ✓ 简化复杂的库或框架
- ✓ 为系统提供统一入口
- ✓ 解耦客户端和子系统
- ✓ 分层设计
- ✓ 提供简单的API

## Java中的应用

```java
// JDBC是外观模式
DriverManager.getConnection("jdbc:mysql://localhost/db");

// Spring Framework的很多服务类
HibernateTemplate.find(query);

// SLF4J是外观模式
Logger logger = LoggerFactory.getLogger(MyClass.class);
```
