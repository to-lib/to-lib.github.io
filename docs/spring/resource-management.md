---
id: resource-management
title: Spring 资源管理
sidebar_label: 资源管理
sidebar_position: 11
---

# Spring 资源管理

> [!TIP]
> **统一资源访问**: Spring 的 Resource 接口提供了统一的方式访问各种资源。无论是文件系统、类路径还是 URL，都可以使用相同的 API。

## 1. Resource 接口概述

**Resource** 接口是 Spring 对底层资源的抽象，提供统一的资源访问方式。

### 1.1 Resource 层次结构

```
Resource (接口)
    ├── UrlResource          - URL 资源
    ├── ClassPathResource    - 类路径资源
    ├── FileSystemResource   - 文件系统资源
    ├── ServletContextResource - Web 应用资源
    ├── InputStreamResource  - 输入流资源
    └── ByteArrayResource    - 字节数组资源
```

### 1.2 核心方法

```java
public interface Resource extends InputStreamSource {
    boolean exists();                  // 资源是否存在
    boolean isReadable();             // 资源是否可读
    boolean isOpen();                 // 资源是否已打开
    URL getURL() throws IOException;  // 获取 URL
    URI getURI() throws IOException;  // 获取 URI
    File getFile() throws IOException; // 获取 File 对象
    long contentLength() throws IOException; // 内容长度
    long lastModified() throws IOException;  // 最后修改时间
    Resource createRelative(String relativePath); // 创建相对资源
    String getFilename();             // 获取文件名
    String getDescription();          // 获取描述
}
```

## 2. 资源类型

### 2.1 ClassPathResource

从类路径加载资源：

```java
@Service
public class ResourceService {
    
    public void loadClassPathResource() throws IOException {
        // 从类路径加载
        Resource resource = new ClassPathResource("config/application.properties");
        
        if (resource.exists()) {
            InputStream inputStream = resource.getInputStream();
            // 读取资源内容
            String content = new String(inputStream.readAllBytes());
            System.out.println(content);
        }
    }
    
    public void loadFromPackage() throws IOException {
        // 从特定包加载
        Resource resource = new ClassPathResource("templates/email.html");
        
        // 或者使用类加载器
        ClassLoader classLoader = getClass().getClassLoader();
        InputStream inputStream = classLoader.getResourceAsStream("config/app.yml");
    }
}
```

### 2.2 FileSystemResource

从文件系统加载资源：

```java
@Service
public class FileResourceService {
    
    public void loadFileResource() throws IOException {
        // 绝对路径
        Resource resource = new FileSystemResource("/var/data/config.txt");
        
        // 相对路径
        Resource relativeResource = new FileSystemResource("data/config.txt");
        
        if (resource.exists()) {
            File file = resource.getFile();
            String content = Files.readString(file.toPath());
            System.out.println(content);
        }
    }
}
```

### 2.3 UrlResource

从 URL 加载资源：

```java
@Service
public class UrlResourceService {
    
    public void loadUrlResource() throws IOException {
        // HTTP URL
        Resource resource = new UrlResource("https://example.com/config.json");
        
        // FTP URL
        Resource ftpResource = new UrlResource("ftp://ftp.example.com/file.txt");
        
        // File URL
        Resource fileResource = new UrlResource("file:///var/data/file.txt");
        
        if (resource.exists() && resource.isReadable()) {
            InputStream inputStream = resource.getInputStream();
            // 处理内容
        }
    }
}
```

### 2.4 ByteArrayResource

从字节数组创建资源：

```java
@Service
public class ByteResourceService {
    
    public void createByteResource() throws IOException {
        byte[] data = "Hello, Spring!".getBytes();
        Resource resource = new ByteArrayResource(data);
        
        InputStream inputStream = resource.getInputStream();
        String content = new String(inputStream.readAllBytes());
        System.out.println(content);
    }
}
```

## 3. ResourceLoader

### 3.1 使用 ResourceLoader

```java
@Service
public class ResourceLoaderService {
    
    @Autowired
    private ResourceLoader resourceLoader;
    
    public void loadResource() throws IOException {
        // 从类路径加载
        Resource resource = resourceLoader.getResource("classpath:config/app.yml");
        
        // 从文件系统加载
        Resource fileResource = resourceLoader.getResource("file:/var/data/config.txt");
        
        // 从 URL 加载
        Resource urlResource = resourceLoader.getResource("https://example.com/data.json");
        
        if (resource.exists()) {
            String content = new String(resource.getInputStream().readAllBytes());
            System.out.println(content);
        }
    }
}
```

### 3.2 资源前缀

| 前缀 | 说明 | 示例 |
|------|------|------|
| `classpath:` | 类路径资源 | `classpath:config/app.yml` |
| `file:` | 文件系统资源 | `file:/var/data/config.txt` |
| `http:` | HTTP URL | `http://example.com/data.json` |
| `https:` | HTTPS URL | `https://example.com/data.json` |
| `ftp:` | FTP URL | `ftp://ftp.example.com/file.txt` |

## 4. @Value 注入资源

### 4.1 注入资源对象

```java
@Component
public class ConfigReader {
    
    @Value("classpath:config/application.properties")
    private Resource configResource;
    
    @Value("${app.data.file}")  // 从配置文件读取路径
    private Resource dataFile;
    
    public void readConfig() throws IOException {
        if (configResource.exists()) {
            Properties properties = new Properties();
            properties.load(configResource.getInputStream());
            System.out.println(properties);
        }
    }
}
```

### 4.2 注入资源内容

```java
@Component
public class ContentReader {
    
    // 直接注入文件内容（小文件）
    @Value("${spring.application.name}")
    private String appName;
    
    // 注入资源
    @Value("classpath:banner.txt")
    private Resource banner;
    
    public void printBanner() throws IOException {
        String content = new String(banner.getInputStream().readAllBytes());
        System.out.println(content);
    }
}
```

## 5. ApplicationContext 和资源

### 5.1 ApplicationContext 作为 ResourceLoader

```java
@Component
public class ContextResourceService {
    
    @Autowired
    private ApplicationContext context;
    
    public void loadResources() throws IOException {
        // ApplicationContext 实现了 ResourceLoader
        Resource resource = context.getResource("classpath:data/users.json");
        
        // 加载多个资源
        Resource[] resources = context.getResources("classpath*:config/*.yml");
        
        for (Resource r : resources) {
            System.out.println("Found: " + r.getFilename());
        }
    }
}
```

### 5.2 通配符支持

```java
@Service
public class WildcardResourceService {
    
    @Autowired
    private ResourcePatternResolver resourcePatternResolver;
    
    public void loadMultipleResources() throws IOException {
        // 加载所有 .properties 文件
        Resource[] resources = resourcePatternResolver.getResources(
            "classpath*:config/*.properties"
        );
        
        for (Resource resource : resources) {
            System.out.println("File: " + resource.getFilename());
            
            // 读取内容
            Properties props = new Properties();
            props.load(resource.getInputStream());
            System.out.println(props);
        }
    }
}
```

## 6. 读取配置文件

### 6.1 Properties 文件

```java
@Service
public class PropertiesReader {
    
    @Value("classpath:app.properties")
    private Resource propertiesResource;
    
    public Properties loadProperties() throws IOException {
        Properties properties = new Properties();
        properties.load(propertiesResource.getInputStream());
        return properties;
    }
    
    public String getProperty(String key) throws IOException {
        Properties props = loadProperties();
        return props.getProperty(key);
    }
}
```

### 6.2 YAML 文件

```java
@Service
public class YamlReader {
    
    @Autowired
    private ResourceLoader resourceLoader;
    
    public Map<String, Object> loadYaml() throws IOException {
        Resource resource = resourceLoader.getResource("classpath:config/app.yml");
        
        Yaml yaml = new Yaml();
        InputStream inputStream = resource.getInputStream();
        
        Map<String, Object> data = yaml.load(inputStream);
        return data;
    }
}
```

### 6.3 JSON 文件

```java
@Service
public class JsonReader {
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Value("classpath:data/users.json")
    private Resource jsonResource;
    
    public List<User> loadUsers() throws IOException {
        InputStream inputStream = jsonResource.getInputStream();
        
        return objectMapper.readValue(
            inputStream,
            new TypeReference<List<User>>() {}
        );
    }
}
```

## 7. 写入资源

### 7.1 写入文件

```java
@Service
public class ResourceWriter {
    
    public void writeToFile(String content, String filename) throws IOException {
        // 创建文件资源
        Resource resource = new FileSystemResource(filename);
        
        // 获取文件
        File file = resource.getFile();
        
        // 写入内容
        Files.writeString(file.toPath(), content);
    }
    
    public void appendToFile(String content, String filename) throws IOException {
        Resource resource = new FileSystemResource(filename);
        File file = resource.getFile();
        
        Files.writeString(
            file.toPath(),
            content,
            StandardOpenOption.CREATE,
            StandardOpenOption.APPEND
        );
    }
}
```

## 8. 实际应用

### 8.1 模板加载

```java
@Service
public class TemplateService {
    
    @Value("classpath:templates/email-welcome.html")
    private Resource emailTemplate;
    
    public String loadEmailTemplate(Map<String, String> variables) throws IOException {
        // 读取模板
        String template = new String(emailTemplate.getInputStream().readAllBytes());
        
        // 替换变量
        for (Map.Entry<String, String> entry : variables.entrySet()) {
            template = template.replace("${" + entry.getKey() + "}", entry.getValue());
        }
        
        return template;
    }
}
```

### 8.2 静态资源服务

```java
@RestController
@RequestMapping("/files")
public class FileController {
    
    @Autowired
    private ResourceLoader resourceLoader;
    
    @GetMapping("/{filename}")
    public ResponseEntity<Resource> downloadFile(@PathVariable String filename) {
        Resource resource = resourceLoader.getResource("classpath:static/downloads/" + filename);
        
        if (!resource.exists()) {
            return ResponseEntity.notFound().build();
        }
        
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + filename + "\"")
            .body(resource);
    }
}
```

### 8.3 多环境配置

```java
@Configuration
public class MultiEnvConfig {
    
    @Autowired
    private Environment environment;
    
    @Autowired
    private ResourceLoader resourceLoader;
    
    @Bean
    public Properties appProperties() throws IOException {
        String profile = environment.getActiveProfiles()[0];
        String filename = "classpath:config/app-" + profile + ".properties";
        
        Resource resource = resourceLoader.getResource(filename);
        Properties properties = new Properties();
        properties.load(resource.getInputStream());
        
        return properties;
    }
}
```

## 9. 最佳实践

### 9.1 使用 ResourceLoader

```java
// ✅ 推荐：注入 ResourceLoader
@Service
public class GoodService {
    @Autowired
    private ResourceLoader resourceLoader;
    
    public void loadResource() {
        Resource resource = resourceLoader.getResource("classpath:data.json");
    }
}

// ❌ 避免：直接创建 Resource
@Service
public class BadService {
    public void loadResource() {
        Resource resource = new ClassPathResource("data.json");
    }
}
```

### 9.2 正确处理异常

```java
@Service
public class SafeResourceService {
    
    @Autowired
    private ResourceLoader resourceLoader;
    
    public Optional<String> loadResourceSafely(String path) {
        try {
            Resource resource = resourceLoader.getResource(path);
            
            if (!resource.exists()) {
                return Optional.empty();
            }
            
            String content = new String(resource.getInputStream().readAllBytes());
            return Optional.of(content);
            
        } catch (IOException e) {
            log.error("Failed to load resource: " + path, e);
            return Optional.empty();
        }
    }
}
```

### 9.3 关闭资源

```java
@Service
public class ResourceCleanupService {
    
    public void processResource(Resource resource) {
        try (InputStream inputStream = resource.getInputStream()) {
            // 处理资源
            byte[] data = inputStream.readAllBytes();
            // try-with-resources 自动关闭流
        } catch (IOException e) {
            log.error("Error processing resource", e);
        }
    }
}
```

## 10. 总结

| 类型 | 用途 | 示例 |
|------|------|------|
| ClassPathResource | 类路径资源 | `classpath:config/app.yml` |
| FileSystemResource | 文件系统资源 | `file:/var/data/file.txt` |
| UrlResource | URL 资源 | `https://example.com/data.json` |
| ResourceLoader | 统一资源加载 | 自动注入使用 |
| @Value | 注入资源 | `@Value("classpath:...")` |

---

**关键要点**：

- 使用 Resource 接口统一资源访问
- ResourceLoader 提供灵活的资源加载
- 支持类路径、文件系统、URL 等多种资源类型
- 使用 try-with-resources 自动关闭流
- @Value 可以直接注入资源对象

**下一步**：学习 [Spring 最佳实践](/docs/spring/best-practices)
