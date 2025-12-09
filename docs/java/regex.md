---
sidebar_position: 12
title: 正则表达式
---

# 正则表达式

正则表达式（Regular Expression）是用于匹配字符串模式的强大工具。本文介绍 Java 中正则表达式的语法和常见应用。

## 正则表达式基础

### 字符类

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class CharacterClasses {
    public static void main(String[] args) {
        // [abc]：任意一个字符
        System.out.println("b".matches("[abc]"));  // true
        System.out.println("d".matches("[abc]"));  // false
        
        // [^abc]：不是 abc 中任何一个
        System.out.println("d".matches("[^abc]"));  // true
        System.out.println("a".matches("[^abc]"));  // false
        
        // [a-z]：范围内的任意一个字符
        System.out.println("m".matches("[a-z]"));  // true
        System.out.println("5".matches("[a-z]"));  // false
        
        // [a-zA-Z0-9]：字母和数字
        System.out.println("a".matches("[a-zA-Z0-9]"));  // true
        System.out.println("-".matches("[a-zA-Z0-9]"));  // false
    }
}
```

### 预定义字符类

```java
public class PredefinedClasses {
    public static void main(String[] args) {
        // \d：数字 [0-9]
        System.out.println("5".matches("\\d"));      // true
        System.out.println("abc".matches("\\d"));    // false
        
        // \D：非数字 [^0-9]
        System.out.println("a".matches("\\D"));      // true
        System.out.println("5".matches("\\D"));      // false
        
        // \w：字母数字下划线 [a-zA-Z0-9_]
        System.out.println("a".matches("\\w"));      // true
        System.out.println("_".matches("\\w"));      // true
        System.out.println("-".matches("\\w"));      // false
        
        // \W：非字母数字下划线
        System.out.println("-".matches("\\W"));      // true
        System.out.println("a".matches("\\W"));      // false
        
        // \s：空白字符（空格、制表符、换行符等）
        System.out.println(" ".matches("\\s"));      // true
        System.out.println("\t".matches("\\s"));     // true
        System.out.println("a".matches("\\s"));      // false
        
        // \S：非空白字符
        System.out.println("a".matches("\\S"));      // true
        System.out.println(" ".matches("\\S"));      // false
    }
}
```

### 量词

```java
public class Quantifiers {
    public static void main(String[] args) {
        // *：出现 0 次或多次
        System.out.println("abc".matches("ab*c"));   // true
        System.out.println("ac".matches("ab*c"));    // true
        System.out.println("abbbc".matches("ab*c")); // true
        System.out.println("adc".matches("ab*c"));   // false
        
        // +：出现 1 次或多次
        System.out.println("abc".matches("ab+c"));   // true
        System.out.println("ac".matches("ab+c"));    // false
        System.out.println("abbbc".matches("ab+c")); // true
        
        // ?：出现 0 次或 1 次
        System.out.println("ac".matches("ab?c"));    // true
        System.out.println("abc".matches("ab?c"));   // true
        System.out.println("abbc".matches("ab?c"));  // false
        
        // {n}：恰好出现 n 次
        System.out.println("abbb".matches("ab{3}c"));    // false
        System.out.println("abbbc".matches("ab{3}c"));   // true
        
        // {n,}：出现 n 次或以上
        System.out.println("abbbc".matches("ab{2,}c"));  // true
        System.out.println("abbbbc".matches("ab{2,}c")); // true
        
        // {n,m}：出现 n 到 m 次
        System.out.println("abc".matches("ab{2,3}c"));   // false
        System.out.println("abbbc".matches("ab{2,3}c")); // true
        System.out.println("abbbbc".matches("ab{2,3}c"));// false
    }
}
```

## Pattern 和 Matcher

### 匹配

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class PatternMatchingExample {
    public static void main(String[] args) {
        // 创建 Pattern（编译正则表达式）
        Pattern pattern = Pattern.compile("\\d{3}-\\d{3}-\\d{4}");
        
        // 电话号码格式：123-456-7890
        String phone1 = "123-456-7890";
        String phone2 = "1234567890";
        
        // 方式1：直接匹配
        boolean matches1 = pattern.matcher(phone1).matches();
        boolean matches2 = pattern.matcher(phone2).matches();
        
        System.out.println(phone1 + " 是有效电话号码: " + matches1);  // true
        System.out.println(phone2 + " 是有效电话号码: " + matches2);  // false
        
        // 方式2：使用 Pattern.matches() 静态方法（每次都编译）
        boolean matches3 = Pattern.matches("\\d{3}-\\d{3}-\\d{4}", phone1);
        System.out.println(matches3);  // true
    }
}
```

### 查找

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class PatternFindingExample {
    public static void main(String[] args) {
        Pattern pattern = Pattern.compile("\\d+");  // 匹配一个或多个数字
        String text = "I have 2 apples and 10 oranges";
        
        Matcher matcher = pattern.matcher(text);
        
        // find()：查找下一个匹配
        System.out.println("查找数字:");
        while (matcher.find()) {
            System.out.println("找到: " + matcher.group());  // 匹配的文本
            System.out.println("位置: " + matcher.start() + "-" + matcher.end());
        }
    }
}
```

### 替换

```java
import java.util.regex.Pattern;

public class PatternReplacingExample {
    public static void main(String[] args) {
        // 替换所有数字为 *
        String text = "My phone is 123-456-7890";
        String result = text.replaceAll("\\d", "*");
        System.out.println(result);  // My phone is ***-***-****
        
        // 替换第一个匹配
        String result2 = text.replaceFirst("\\d{3}", "***");
        System.out.println(result2);  // My phone is ***-456-7890
        
        // 使用 Pattern
        Pattern pattern = Pattern.compile("\\d{3}-\\d{3}-(\\d{4})");
        String result3 = pattern.matcher(text).replaceAll("***-***-$1");
        System.out.println(result3);  // My phone is ***-***-7890
    }
}
```

### 分割

```java
public class PatternSplittingExample {
    public static void main(String[] args) {
        // 按数字分割
        String text = "apple2orange5banana8grape";
        String[] parts = text.split("\\d");
        for (String part : parts) {
            System.out.println(part);
        }
        
        // 输出：
        // apple
        // orange
        // banana
        // grape
        
        // 按多个空格分割
        String csv = "name   age    city";
        String[] fields = csv.split("\\s+");
        for (String field : fields) {
            System.out.println(field);
        }
    }
}
```

## 常见正则表达式

### 验证电子邮件

```java
public class EmailValidation {
    public static boolean isValidEmail(String email) {
        String pattern = "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$";
        return email.matches(pattern);
    }
    
    public static void main(String[] args) {
        System.out.println(isValidEmail("user@example.com"));      // true
        System.out.println(isValidEmail("invalid.email@"));        // false
        System.out.println(isValidEmail("user.name+tag@example.co.uk"));  // true
    }
}
```

### 验证电话号码

```java
public class PhoneValidation {
    // 验证中国大陆手机号
    public static boolean isChinaPhone(String phone) {
        String pattern = "^1[3456789]\\d{9}$";
        return phone.matches(pattern);
    }
    
    // 验证国际格式
    public static boolean isInternationalPhone(String phone) {
        String pattern = "^\\+?\\d{1,3}[\\s-]?\\d{3,14}$";
        return phone.matches(pattern);
    }
    
    public static void main(String[] args) {
        System.out.println(isChinaPhone("13812345678"));      // true
        System.out.println(isChinaPhone("138123456789"));     // false
        System.out.println(isInternationalPhone("+1-202-555-0173"));  // true
    }
}
```

### 验证 URL

```java
public class URLValidation {
    public static boolean isValidURL(String url) {
        String pattern = "^(https?://)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([/\\w \\.-]*)*/?$";
        return url.matches(pattern);
    }
    
    public static void main(String[] args) {
        System.out.println(isValidURL("http://www.example.com"));           // true
        System.out.println(isValidURL("https://github.com/user/repo"));     // true
        System.out.println(isValidURL("not a url"));                        // false
    }
}
```

### 验证身份证号

```java
public class IDCardValidation {
    public static boolean isValidIDCard(String id) {
        // 18位身份证
        if (id.length() == 18) {
            String pattern = "^[1-9]\\d{5}[12]\\d{3}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{3}[\\dXx]$";
            return id.matches(pattern);
        }
        // 15位身份证
        else if (id.length() == 15) {
            String pattern = "^[1-9]\\d{5}\\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{2}$";
            return id.matches(pattern);
        }
        return false;
    }
    
    public static void main(String[] args) {
        System.out.println(isValidIDCard("110101199003074219"));  // true
        System.out.println(isValidIDCard("invalid"));            // false
    }
}
```

### 验证IP地址

```java
public class IPAddressValidation {
    public static boolean isValidIPv4(String ip) {
        String pattern = "^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$";
        return ip.matches(pattern);
    }
    
    public static void main(String[] args) {
        System.out.println(isValidIPv4("192.168.1.1"));          // true
        System.out.println(isValidIPv4("256.1.1.1"));            // false
        System.out.println(isValidIPv4("192.168.1"));            // false
    }
}
```

## 高级用法

### 捕获组

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class CapturingGroups {
    public static void main(String[] args) {
        // 提取日期中的年、月、日
        Pattern pattern = Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");
        String dateStr = "2024-12-09";
        
        Matcher matcher = pattern.matcher(dateStr);
        if (matcher.matches()) {
            System.out.println("完整匹配: " + matcher.group(0));  // 2024-12-09
            System.out.println("年: " + matcher.group(1));       // 2024
            System.out.println("月: " + matcher.group(2));       // 12
            System.out.println("日: " + matcher.group(3));       // 09
        }
    }
}
```

### 非捕获组

```java
import java.util.regex.Pattern;

public class NonCapturingGroups {
    public static void main(String[] args) {
        // (?:...) 非捕获组
        Pattern pattern = Pattern.compile("(?:https?|ftp)://([\\w.-]+)");
        
        String url = "https://www.example.com";
        String replaced = pattern.matcher(url).replaceAll("[$1]");
        System.out.println(replaced);  // [www.example.com]
    }
}
```

### 前向和后向查找

```java
public class LookaroundExample {
    public static void main(String[] args) {
        // 正向查找：(?=pattern)
        // 查找后面跟着 "€" 的数字
        String text = "100$, 200€, 300¥";
        String[] prices = text.split(",");
        
        for (String price : prices) {
            if (price.trim().matches("\\d+(?=€)")) {
                System.out.println("欧元: " + price.trim());
            }
        }
        
        // 反向查找：(?<=pattern)
        // 查找前面是 "¥" 的数字
        if ("300¥".matches("(?<=¥)\\d+")) {
            System.out.println("找到人民币价格");
        }
    }
}
```

## 实际应用

### HTML 标签处理

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class HTMLTagProcessing {
    // 移除 HTML 标签
    public static String removeHTMLTags(String html) {
        String pattern = "<[^>]*>";
        return html.replaceAll(pattern, "");
    }
    
    // 提取 HTML 标签内容
    public static void extractTagContent(String html, String tagName) {
        String pattern = "<" + tagName + "[^>]*>([^<]*)</" + tagName + ">";
        Pattern p = Pattern.compile(pattern);
        Matcher m = p.matcher(html);
        
        while (m.find()) {
            System.out.println(m.group(1));
        }
    }
    
    public static void main(String[] args) {
        String html = "<p>Hello <b>World</b></p>";
        System.out.println(removeHTMLTags(html));  // Hello World
        
        extractTagContent(html, "b");  // World
    }
}
```

### 日志解析

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class LogParsing {
    public static void main(String[] args) {
        String logLine = "2024-12-09 14:30:45 ERROR Database connection failed";
        
        // 解析日志格式
        Pattern pattern = Pattern.compile(
            "(\\d{4}-\\d{2}-\\d{2})\\s+" +    // 日期
            "(\\d{2}:\\d{2}:\\d{2})\\s+" +    // 时间
            "(ERROR|WARN|INFO)\\s+" +          // 日志级别
            "(.+)"                              // 消息
        );
        
        Matcher matcher = pattern.matcher(logLine);
        if (matcher.matches()) {
            System.out.println("日期: " + matcher.group(1));
            System.out.println("时间: " + matcher.group(2));
            System.out.println("级别: " + matcher.group(3));
            System.out.println("消息: " + matcher.group(4));
        }
    }
}
```

## 性能优化

```java
import java.util.regex.Pattern;

public class RegexPerformance {
    // ❌ 不推荐：每次都编译正则表达式
    public static void inefficientRegex(String[] texts) {
        for (String text : texts) {
            if (text.matches("\\d+")) {  // 每次都编译
                System.out.println(text);
            }
        }
    }
    
    // ✅ 推荐：提前编译，重复使用
    private static final Pattern PATTERN = Pattern.compile("\\d+");
    
    public static void efficientRegex(String[] texts) {
        for (String text : texts) {
            if (PATTERN.matcher(text).matches()) {  // 复用编译好的 Pattern
                System.out.println(text);
            }
        }
    }
}
```

## 总结

本文介绍了 Java 正则表达式的核心内容：

- ✅ 字符类和量词
- ✅ Pattern 和 Matcher 的使用
- ✅ 常见的验证场景
- ✅ 捕获组和高级查找
- ✅ 实际应用和性能优化

掌握正则表达式后，可以更高效地进行字符串处理和数据验证。
