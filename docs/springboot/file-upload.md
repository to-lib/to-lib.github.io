---
sidebar_position: 21
---

# 文件上传下载

> [!TIP]
> **文件处理要点**: 合理限制文件大小和类型、使用流式处理大文件、选择合适的存储方案（本地存储 vs 云存储）。

## 文件上传

### 基本配置

```yaml
spring:
  servlet:
    multipart:
      enabled: true
      max-file-size: 10MB      # 单个文件最大大小
      max-request-size: 50MB   # 整个请求最大大小
      file-size-threshold: 2KB # 文件写入磁盘的阈值
```

### 单文件上传

```java
@RestController
@RequestMapping("/api/files")
public class FileUploadController {
    
    @Value("${file.upload.dir:/tmp/uploads}")
    private String uploadDir;
    
    @PostMapping("/upload")
    public ResponseEntity<FileUploadResponse> uploadFile(
            @RequestParam("file") MultipartFile file) {
        
        // 验证文件
        if (file.isEmpty()) {
            throw new IllegalArgumentException("文件不能为空");
        }
        
        // 验证文件类型
        String contentType = file.getContentType();
        if (!isValidFileType(contentType)) {
            throw new IllegalArgumentException("不支持的文件类型");
        }
        
        try {
            // 生成唯一文件名
            String originalFilename = file.getOriginalFilename();
            String extension = FilenameUtils.getExtension(originalFilename);
            String newFilename = UUID.randomUUID().toString() + "." + extension;
            
            // 创建上传目录
            Path uploadPath = Paths.get(uploadDir);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }
            
            // 保存文件
            Path filePath = uploadPath.resolve(newFilename);
            Files.copy(file.getInputStream(), filePath, 
                StandardCopyOption.REPLACE_EXISTING);
            
            // 构建响应
            FileUploadResponse response = FileUploadResponse.builder()
                .filename(newFilename)
                .originalFilename(originalFilename)
                .size(file.getSize())
                .contentType(contentType)
                .uploadTime(LocalDateTime.now())
                .url("/api/files/download/" + newFilename)
                .build();
            
            return ResponseEntity.ok(response);
            
        } catch (IOException e) {
            throw new RuntimeException("文件上传失败", e);
        }
    }
    
    private boolean isValidFileType(String contentType) {
        List<String> allowedTypes = Arrays.asList(
            "image/jpeg", "image/png", "image/gif",
            "application/pdf", "text/plain"
        );
        return allowedTypes.contains(contentType);
    }
}
```

### 多文件上传

```java
@PostMapping("/upload/multiple")
public ResponseEntity<List<FileUploadResponse>> uploadMultipleFiles(
        @RequestParam("files") MultipartFile[] files) {
    
    List<FileUploadResponse> responses = new ArrayList<>();
    
    for (MultipartFile file : files) {
        if (!file.isEmpty()) {
            FileUploadResponse response = saveFile(file);
            responses.add(response);
        }
    }
    
    return ResponseEntity.ok(responses);
}
```

## 文件下载

### 基本下载

```java
@GetMapping("/download/{filename}")
public ResponseEntity<Resource> downloadFile(@PathVariable String filename) {
    try {
        Path filePath = Paths.get(uploadDir).resolve(filename).normalize();
        Resource resource = new UrlResource(filePath.toUri());
        
        if (!resource.exists()) {
            throw new ResourceNotFoundException("文件不存在: " + filename);
        }
        
        // 获取文件的 Content-Type
        String contentType = Files.probeContentType(filePath);
        if (contentType == null) {
            contentType = "application/octet-stream";
        }
        
        return ResponseEntity.ok()
            .contentType(MediaType.parseMediaType(contentType))
            .header(HttpHeaders.CONTENT_DISPOSITION, 
                "attachment; filename=\"" + resource.getFilename() + "\"")
            .body(resource);
            
    } catch (IOException e) {
        throw new RuntimeException("文件下载失败", e);
    }
}
```

### 流式下载大文件

```java
@GetMapping("/download/large/{filename}")
public void downloadLargeFile(@PathVariable String filename, 
                              HttpServletResponse response) {
    try {
        Path filePath = Paths.get(uploadDir).resolve(filename);
        
        response.setContentType("application/octet-stream");
        response.setHeader(HttpHeaders.CONTENT_DISPOSITION,
            "attachment; filename=\"" + filename + "\"");
        response.setContentLengthLong(Files.size(filePath));
        
        // 使用流式传输
        try (InputStream inputStream = Files.newInputStream(filePath);
             OutputStream outputStream = response.getOutputStream()) {
            
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.flush();
        }
        
    } catch (IOException e) {
        throw new RuntimeException("文件下载失败", e);
    }
}
```

## 文件管理服务

```java
@Service
public class FileStorageService {
    
    @Value("${file.upload.dir}")
    private Path uploadPath;
    
    @PostConstruct
    public void init() {
        try {
            Files.createDirectories(uploadPath);
        } catch (IOException e) {
            throw new RuntimeException("无法创建上传目录", e);
        }
    }
    
    public String store(MultipartFile file) {
        String filename = StringUtils.cleanPath(file.getOriginalFilename());
        
        try {
            if (file.isEmpty()) {
                throw new RuntimeException("存储失败：文件为空 " + filename);
            }
            
            if (filename.contains("..")) {
                throw new RuntimeException("非法路径 " + filename);
            }
            
            String storedFilename = UUID.randomUUID().toString() + 
                "_" + filename;
            
            try (InputStream inputStream = file.getInputStream()) {
                Files.copy(inputStream, uploadPath.resolve(storedFilename),
                    StandardCopyOption.REPLACE_EXISTING);
            }
            
            return storedFilename;
            
        } catch (IOException e) {
            throw new RuntimeException("存储文件失败 " + filename, e);
        }
    }
    
    public Resource loadAsResource(String filename) {
        try {
            Path file = uploadPath.resolve(filename);
            Resource resource = new UrlResource(file.toUri());
            
            if (resource.exists() || resource.isReadable()) {
                return resource;
            } else {
                throw new RuntimeException("文件不存在或不可读: " + filename);
            }
        } catch (MalformedURLException e) {
            throw new RuntimeException("文件不存在 " + filename, e);
        }
    }
    
    public void delete(String filename) {
        try {
            Path file = uploadPath.resolve(filename);
            Files.deleteIfExists(file);
        } catch (IOException e) {
            throw new RuntimeException("删除文件失败 " + filename, e);
        }
    }
}
```

## 图片上传与压缩

```java
@Service
public class ImageService {
    
    public String uploadAndCompressImage(MultipartFile file) throws IOException {
        // 读取图片
        BufferedImage originalImage = ImageIO.read(file.getInputStream());
        
        // 压缩图片
        BufferedImage compressedImage = compressImage(originalImage, 0.7f);
        
        // 保存压缩后的图片
        String filename = UUID.randomUUID().toString() + ".jpg";
        Path outputPath = Paths.get(uploadDir, filename);
        ImageIO.write(compressedImage, "jpg", outputPath.toFile());
        
        return filename;
    }
    
    private BufferedImage compressImage(BufferedImage image, float quality) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // 如果图片太大，缩小尺寸
        if (width > 1920 || height > 1080) {
            double scale = Math.min(1920.0 / width, 1080.0 / height);
            width = (int) (width * scale);
            height = (int) (height * scale);
            
            BufferedImage resized = new BufferedImage(
                width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0, 0, width, height, null);
            g.dispose();
            
            return resized;
        }
        
        return image;
    }
}
```

## 云存储集成（阿里云 OSS）

```java
@Service
public class OSSFileService {
    
    @Value("${aliyun.oss.endpoint}")
    private String endpoint;
    
    @Value("${aliyun.oss.accessKeyId}")
    private String accessKeyId;
    
    @Value("${aliyun.oss.accessKeySecret}")
    private String accessKeySecret;
    
    @Value("${aliyun.oss.bucketName}")
    private String bucketName;
    
    private OSS ossClient;
    
    @PostConstruct
    public void init() {
        ossClient = new OSSClientBuilder().build(
            endpoint, accessKeyId, accessKeySecret);
    }
    
    public String upload(MultipartFile file) {
        String filename = UUID.randomUUID().toString() + 
            "_" + file.getOriginalFilename();
        
        try {
            ossClient.putObject(bucketName, filename, 
                file.getInputStream());
            
            return "https://" + bucketName + "." + endpoint + "/" + filename;
        } catch (IOException e) {
            throw new RuntimeException("上传OSS失败", e);
        }
    }
    
    @PreDestroy
    public void destroy() {
        if (ossClient != null) {
            ossClient.shutdown();
        }
    }
}
```

## 最佳实践

> [!IMPORTANT]
> **文件上传安全建议**：
>
> 1. **验证文件类型** - 检查 MIME 类型和文件扩展名
> 2. **限制文件大小** - 防止磁盘撑爆
> 3. **重命名文件** - 使用 UUID 避免冲突和安全问题
> 4. **扫描病毒** - 集成杀毒软件
> 5. **存储隔离** - 不要存储在 Web 根目录
> 6. **访问控制** - 根据需求控制文件访问权限

## 总结

- **MultipartFile** - Spring 提供的文件上传接口
- **Resource** - 文件下载资源抽象
- **流式处理** - 处理大文件避免内存溢出
- **云存储** - 生产环境推荐使用云存储服务

下一步学习 [国际化](./i18n)。
