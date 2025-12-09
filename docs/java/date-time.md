---
sidebar_position: 10
title: 日期时间 API
---

# 日期时间 API

Java 8 引入了新的日期时间 API（java.time 包），提供了更好的日期时间处理方式。本文介绍 LocalDate、LocalTime、LocalDateTime 等核心类的使用。

## 为什么需要新的日期时间 API

### 旧 API 的问题

Java 8 之前的 Date 和 Calendar 存在很多问题：

```java
import java.util.Date;
import java.util.Calendar;

public class OldDateTimeProblems {
    public static void main(String[] args) {
        // 问题1：Date 的月份从 0 开始
        Date date = new Date();
        System.out.println(date.getMonth());  // 返回 0-11，容易出错
        
        // 问题2：Date 是可变的，不线程安全
        Date d1 = new Date();
        Date d2 = d1;
        d2.setTime(0);  // 修改 d2 会影响 d1
        
        // 问题3：Calendar API 复杂且不直观
        Calendar calendar = Calendar.getInstance();
        calendar.set(2024, 11, 25);  // 月份是 0-11
        System.out.println(calendar.getTime());
    }
}
```

## 新 API 核心类

### LocalDate（日期）

LocalDate 表示一个不可变的日期对象（年-月-日）。

```java
import java.time.LocalDate;
import java.time.Month;

public class LocalDateExample {
    public static void main(String[] args) {
        // 创建当前日期
        LocalDate today = LocalDate.now();
        System.out.println("今天: " + today);  // 2024-12-09
        
        // 创建指定日期
        LocalDate christmas = LocalDate.of(2024, 12, 25);
        System.out.println("圣诞节: " + christmas);
        
        // 使用 Month 枚举
        LocalDate newYear = LocalDate.of(2025, Month.JANUARY, 1);
        System.out.println("新年: " + newYear);
        
        // 解析日期字符串
        LocalDate parsed = LocalDate.parse("2024-12-09");
        System.out.println("解析的日期: " + parsed);
        
        // 获取日期信息
        System.out.println("年: " + today.getYear());
        System.out.println("月: " + today.getMonthValue());  // 1-12
        System.out.println("日: " + today.getDayOfMonth());
        System.out.println("星期: " + today.getDayOfWeek());  // MONDAY...
        System.out.println("本年第几天: " + today.getDayOfYear());
        
        // 日期加减
        LocalDate tomorrow = today.plusDays(1);
        LocalDate nextMonth = today.plusMonths(1);
        LocalDate nextYear = today.plusYears(1);
        
        LocalDate yesterday = today.minusDays(1);
        LocalDate lastMonth = today.minusMonths(1);
        
        // 判断日期
        System.out.println("是否是闰年: " + today.isLeapYear());
        System.out.println("是否在某日期之前: " + today.isBefore(christmas));
        System.out.println("是否在某日期之后: " + today.isAfter(christmas));
    }
}
```

### LocalTime（时间）

LocalTime 表示一个不可变的时间对象（时-分-秒-纳秒）。

```java
import java.time.LocalTime;

public class LocalTimeExample {
    public static void main(String[] args) {
        // 创建当前时间
        LocalTime now = LocalTime.now();
        System.out.println("当前时间: " + now);  // 14:30:45.123456789
        
        // 创建指定时间
        LocalTime noon = LocalTime.of(12, 0, 0);
        LocalTime morningTime = LocalTime.of(8, 30, 45);
        
        // 解析时间字符串
        LocalTime parsed = LocalTime.parse("14:30:45");
        System.out.println("解析的时间: " + parsed);
        
        // 获取时间信息
        System.out.println("小时: " + now.getHour());
        System.out.println("分钟: " + now.getMinute());
        System.out.println("秒: " + now.getSecond());
        System.out.println("纳秒: " + now.getNano());
        
        // 时间加减
        LocalTime oneHourLater = now.plusHours(1);
        LocalTime thirtyMinutesLater = now.plusMinutes(30);
        
        LocalTime oneHourBefore = now.minusHours(1);
        
        // 判断时间
        System.out.println("是否在某时间之前: " + now.isBefore(noon));
        System.out.println("是否在某时间之后: " + now.isAfter(noon));
    }
}
```

### LocalDateTime（日期时间）

LocalDateTime 包含日期和时间信息。

```java
import java.time.LocalDateTime;

public class LocalDateTimeExample {
    public static void main(String[] args) {
        // 创建当前日期时间
        LocalDateTime now = LocalDateTime.now();
        System.out.println("当前日期时间: " + now);
        
        // 创建指定日期时间
        LocalDateTime christmas = LocalDateTime.of(2024, 12, 25, 19, 30, 0);
        
        // 从 LocalDate 和 LocalTime 创建
        LocalDateTime combined = LocalDateTime.of(
            LocalDate.of(2024, 12, 25),
            LocalTime.of(19, 30, 0)
        );
        
        // 解析日期时间字符串
        LocalDateTime parsed = LocalDateTime.parse("2024-12-25T19:30:45");
        
        // 获取日期和时间部分
        LocalDate date = now.toLocalDate();
        LocalTime time = now.toLocalTime();
        
        // 日期时间加减
        LocalDateTime tomorrow = now.plusDays(1);
        LocalDateTime inTwoHours = now.plusHours(2);
        LocalDateTime lastMonth = now.minusMonths(1);
    }
}
```

### ZonedDateTime（带时区的日期时间）

ZonedDateTime 包含时区信息。

```java
import java.time.ZonedDateTime;
import java.time.ZoneId;
import java.time.LocalDateTime;

public class ZonedDateTimeExample {
    public static void main(String[] args) {
        // 创建带时区的日期时间
        ZonedDateTime now = ZonedDateTime.now();
        System.out.println("当前日期时间（默认时区）: " + now);
        
        // 指定时区创建
        ZonedDateTime tokyoTime = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        ZonedDateTime newYorkTime = ZonedDateTime.now(ZoneId.of("America/New_York"));
        
        System.out.println("东京时间: " + tokyoTime);
        System.out.println("纽约时间: " + newYorkTime);
        
        // 获取所有可用时区
        ZoneId.getAvailableZoneIds().forEach(System.out::println);
    }
}
```

### Instant（时刻）

Instant 表示时间线上的一个点，通常用于记录事件发生的时刻。

```java
import java.time.Instant;

public class InstantExample {
    public static void main(String[] args) {
        // 获取当前时刻
        Instant now = Instant.now();
        System.out.println("当前时刻: " + now);  // 2024-12-09T06:30:45.123456Z
        
        // 从纪元秒创建
        Instant epoch = Instant.ofEpochSecond(0);
        System.out.println("纪元: " + epoch);
        
        // 从毫秒时间戳创建
        long timeStamp = System.currentTimeMillis();
        Instant fromTimestamp = Instant.ofEpochMilli(timeStamp);
        
        // 获取秒数和纳秒
        System.out.println("秒: " + now.getEpochSecond());
        System.out.println("纳秒: " + now.getNano());
    }
}
```

## 时间间隔和期间

### Duration（时间间隔）

Duration 表示两个时间点之间的秒数和纳秒数。

```java
import java.time.LocalDateTime;
import java.time.Duration;

public class DurationExample {
    public static void main(String[] args) {
        LocalDateTime start = LocalDateTime.now();
        LocalDateTime end = start.plusHours(2).plusMinutes(30);
        
        // 计算持续时间
        Duration duration = Duration.between(start, end);
        
        System.out.println("总秒数: " + duration.getSeconds());
        System.out.println("纳秒: " + duration.getNano());
        System.out.println("总分钟数: " + duration.toMinutes());
        System.out.println("总小时数: " + duration.toHours());
        System.out.println("总天数: " + duration.toDays());
        
        // 创建 Duration
        Duration d1 = Duration.ofHours(2);
        Duration d2 = Duration.ofMinutes(30);
        Duration d3 = Duration.ofSeconds(3600);
        Duration d4 = Duration.ofMillis(1000);
        
        // 计算
        Duration sum = d1.plus(d2);
        System.out.println("2小时 + 30分钟 = " + sum.toMinutes() + "分钟");
    }
}
```

### Period（日期期间）

Period 表示两个日期之间的年、月、日。

```java
import java.time.LocalDate;
import java.time.Period;

public class PeriodExample {
    public static void main(String[] args) {
        LocalDate start = LocalDate.of(2020, 1, 1);
        LocalDate end = LocalDate.of(2024, 12, 9);
        
        // 计算期间
        Period period = Period.between(start, end);
        
        System.out.println("年: " + period.getYears());
        System.out.println("月: " + period.getMonths());
        System.out.println("日: " + period.getDays());
        System.out.println("总天数: " + period.getDays());
        
        // 创建 Period
        Period p1 = Period.ofYears(1);
        Period p2 = Period.ofMonths(2);
        Period p3 = Period.ofWeeks(3);
        Period p4 = Period.ofDays(7);
        
        // 计算
        LocalDate today = LocalDate.now();
        LocalDate nextYear = today.plus(Period.ofYears(1));
        System.out.println("明年今天: " + nextYear);
    }
}
```

## 日期时间格式化

### DateTimeFormatter

DateTimeFormatter 用于格式化和解析日期时间。

```java
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatter;

public class DateTimeFormatterExample {
    public static void main(String[] args) {
        LocalDateTime now = LocalDateTime.now();
        
        // 使用预定义的格式
        System.out.println("ISO_DATE: " + now.format(DateTimeFormatter.ISO_DATE));
        System.out.println("ISO_TIME: " + now.format(DateTimeFormatter.ISO_TIME));
        System.out.println("ISO_DATE_TIME: " + now.format(DateTimeFormatter.ISO_DATE_TIME));
        
        // 使用自定义格式
        DateTimeFormatter formatter1 = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        System.out.println("自定义格式1: " + now.format(formatter1));
        
        DateTimeFormatter formatter2 = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        System.out.println("自定义格式2: " + now.format(formatter2));
        
        DateTimeFormatter formatter3 = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss");
        System.out.println("自定义格式3: " + now.format(formatter3));
        
        // 解析日期时间
        String dateStr = "2024-12-25";
        LocalDateTime parsed = LocalDateTime.parse(
            dateStr + " 19:30:00",
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
        );
        System.out.println("解析的日期时间: " + parsed);
    }
}
```

### 常用格式化模式

```java
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

public class FormatterPatterns {
    public static void main(String[] args) {
        LocalDateTime now = LocalDateTime.now();
        
        // 常用格式
        System.out.println("yyyy-MM-dd: " + now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
        System.out.println("HH:mm:ss: " + now.format(DateTimeFormatter.ofPattern("HH:mm:ss")));
        System.out.println("yyyy-MM-dd HH:mm:ss: " + now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        
        // 包含星期和月份名称
        System.out.println("EEEE, MMMM dd, yyyy: " + 
            now.format(DateTimeFormatter.ofPattern("EEEE, MMMM dd, yyyy", Locale.ENGLISH)));
        System.out.println("EEEE, MMMM dd, yyyy: " + 
            now.format(DateTimeFormatter.ofPattern("EEEE, MMMM dd, yyyy", Locale.CHINESE)));
        
        // 时间戳格式
        System.out.println("yyyy年MM月dd日 HH时mm分ss秒: " + 
            now.format(DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH时mm分ss秒")));
    }
}
```

## 实际应用示例

### 计算日期差异

```java
import java.time.LocalDate;
import java.time.Period;
import java.time.temporal.ChronoUnit;

public class DateDifferenceExample {
    public static void main(String[] args) {
        LocalDate birthDate = LocalDate.of(1990, 5, 15);
        LocalDate today = LocalDate.now();
        
        // 方式1：使用 Period
        Period age = Period.between(birthDate, today);
        System.out.println("年龄: " + age.getYears() + "岁");
        
        // 方式2：使用 ChronoUnit
        long daysOld = ChronoUnit.DAYS.between(birthDate, today);
        long monthsOld = ChronoUnit.MONTHS.between(birthDate, today);
        long yearsOld = ChronoUnit.YEARS.between(birthDate, today);
        
        System.out.println("出生以来经过的天数: " + daysOld);
        System.out.println("出生以来经过的月数: " + monthsOld);
        System.out.println("出生以来经过的年数: " + yearsOld);
    }
}
```

### 判断工作日和周末

```java
import java.time.LocalDate;
import java.time.DayOfWeek;

public class WorkdayExample {
    public static void main(String[] args) {
        LocalDate date = LocalDate.now();
        DayOfWeek dayOfWeek = date.getDayOfWeek();
        
        if (dayOfWeek == DayOfWeek.SATURDAY || dayOfWeek == DayOfWeek.SUNDAY) {
            System.out.println(date + " 是周末");
        } else {
            System.out.println(date + " 是工作日");
        }
        
        // 计算下一个工作日
        LocalDate nextWorkday = date.plusDays(1);
        while (nextWorkday.getDayOfWeek() == DayOfWeek.SATURDAY ||
               nextWorkday.getDayOfWeek() == DayOfWeek.SUNDAY) {
            nextWorkday = nextWorkday.plusDays(1);
        }
        System.out.println("下一个工作日: " + nextWorkday);
    }
}
```

### 计算月末日期

```java
import java.time.LocalDate;
import java.time.YearMonth;

public class MonthEndExample {
    public static void main(String[] args) {
        LocalDate today = LocalDate.now();
        
        // 获取这个月的最后一天
        YearMonth yearMonth = YearMonth.from(today);
        LocalDate lastDayOfMonth = yearMonth.atEndOfMonth();
        
        System.out.println("当前日期: " + today);
        System.out.println("本月最后一天: " + lastDayOfMonth);
        System.out.println("还剩 " + (lastDayOfMonth.getDayOfMonth() - today.getDayOfMonth()) + " 天");
    }
}
```

### 定时任务示例

```java
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

public class ScheduleExample {
    public static void main(String[] args) throws InterruptedException {
        // 获取明天的特定时间
        LocalDateTime tomorrow = LocalDateTime.now()
            .plusDays(1)
            .withHour(10)
            .withMinute(0)
            .withSecond(0)
            .withNano(0);
        
        // 计算延迟时间
        LocalDateTime now = LocalDateTime.now();
        long delaySeconds = ChronoUnit.SECONDS.between(now, tomorrow);
        
        System.out.println("当前时间: " + now);
        System.out.println("任务执行时间: " + tomorrow);
        System.out.println("延迟秒数: " + delaySeconds);
        
        // 延迟执行任务
        if (delaySeconds > 0) {
            System.out.println("等待 " + delaySeconds + " 秒后执行任务");
            Thread.sleep(delaySeconds * 1000);
            System.out.println("执行定时任务");
        }
    }
}
```

## 最佳实践

### 1. 优先使用新 API

```java
// ❌ 不推荐：使用旧 API
import java.util.Date;
Date date = new Date();

// ✅ 推荐：使用新 API
import java.time.LocalDateTime;
LocalDateTime dateTime = LocalDateTime.now();
```

### 2. LocalDateTime vs ZonedDateTime

```java
// 使用 LocalDateTime：没有时区关系的本地日期时间
LocalDateTime meetTime = LocalDateTime.of(2024, 12, 25, 19, 30);

// 使用 ZonedDateTime：涉及多个时区的日期时间
ZonedDateTime startTime = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
```

### 3. 不可变和线程安全

```java
// LocalDateTime 是不可变的，可以安全地在多线程间使用
LocalDateTime time = LocalDateTime.now();
// time.plusDays(1) 返回新对象，不修改 time
LocalDateTime tomorrow = time.plusDays(1);

System.out.println(time);      // 原对象不变
System.out.println(tomorrow);  // 新对象
```

## 总结

本文介绍了 Java 8+ 的新日期时间 API：

- ✅ LocalDate、LocalTime、LocalDateTime 的使用
- ✅ ZonedDateTime 和 Instant 处理时区
- ✅ Duration 和 Period 计算时间差
- ✅ DateTimeFormatter 格式化和解析
- ✅ 实际应用示例和最佳实践

掌握日期时间 API 后，可以更方便地处理日期时间相关的业务逻辑。
