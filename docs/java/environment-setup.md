---
sidebar_position: 2
title: JDK 1.8 开发环境搭建
---

# JDK 1.8 开发环境搭建

本文档指导你在不同操作系统上安装和配置 JDK 1.8 开发环境。

> [!IMPORTANT]
> 本教程所有代码示例均基于 **JDK 1.8 (Java 8)** 版本，这是目前企业开发中最广泛使用的 Java 版本。

## JDK 版本选择

### 为什么选择 JDK 1.8？

- ✅ **长期支持版本（LTS）** - Oracle 和 OpenJDK 提供长期维护
- ✅ **企业首选** - 大多数企业生产环境使用 JDK 8
- ✅ **生态成熟** - 绝大多数框架和库完美支持
- ✅ **革命性特性** - Lambda 表达式、Stream API、新日期时间 API

### JDK 发行版选择

| 发行版               | 特点                             | 推荐场景       |
| -------------------- | -------------------------------- | -------------- |
| **Oracle JDK**       | Oracle 官方版本，商业使用需授权  | 企业付费用户   |
| **OpenJDK**          | 开源免费，与 Oracle JDK 功能一致 | 通用开发       |
| **Amazon Corretto**  | AWS 维护，免费长期支持           | AWS 云环境     |
| **Adoptium Temurin** | Eclipse 基金会维护，推荐使用     | 个人和商业开发 |

## Windows 安装

### 步骤 1：下载 JDK

从 [Adoptium](https://adoptium.net/temurin/releases/?version=8) 下载 JDK 8：

1. 选择 **Operating System**: Windows
2. 选择 **Architecture**: x64
3. 下载 `.msi` 安装包

### 步骤 2：安装 JDK

1. 双击下载的 `.msi` 文件
2. 按照向导完成安装
3. 记录安装路径（默认：`C:\Program Files\Eclipse Adoptium\jdk-8.xxx`）

### 步骤 3：配置环境变量

1. 右键点击"此电脑" → "属性" → "高级系统设置"
2. 点击"环境变量"

**设置 JAVA_HOME：**

```
变量名: JAVA_HOME
变量值: C:\Program Files\Eclipse Adoptium\jdk-8.0.xxx-hotspot
```

**设置 Path：**
在系统变量 Path 中添加：

```
%JAVA_HOME%\bin
```

### 步骤 4：验证安装

打开命令提示符：

```bash
java -version
```

预期输出：

```
openjdk version "1.8.0_xxx"
OpenJDK Runtime Environment (Temurin)(build 1.8.0_xxx)
OpenJDK 64-Bit Server VM (Temurin)(build 25.xxx, mixed mode)
```

```bash
javac -version
```

预期输出：

```
javac 1.8.0_xxx
```

## macOS 安装

### 方法 1：使用 Homebrew（推荐）

```bash
# 安装 Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 JDK 8
brew install --cask temurin@8
```

### 方法 2：手动安装

1. 从 [Adoptium](https://adoptium.net/temurin/releases/?version=8) 下载 `.pkg` 文件
2. 双击安装
3. 安装路径：`/Library/Java/JavaVirtualMachines/temurin-8.jdk`

### 配置环境变量

编辑 `~/.zshrc` 或 `~/.bash_profile`：

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
export PATH=$JAVA_HOME/bin:$PATH
```

使配置生效：

```bash
source ~/.zshrc
```

### 验证安装

```bash
java -version
javac -version
echo $JAVA_HOME
```

## Linux 安装

### Ubuntu/Debian

```bash
# 更新包列表
sudo apt update

# 安装 OpenJDK 8
sudo apt install openjdk-8-jdk

# 设置 JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### CentOS/RHEL

```bash
# 安装 OpenJDK 8
sudo yum install java-1.8.0-openjdk-devel

# 设置 JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 验证安装

```bash
java -version
javac -version
echo $JAVA_HOME
```

## IDE 配置

### IntelliJ IDEA

1. 打开 **File** → **Project Structure** (⌘/Ctrl + ;)
2. 选择 **Project** → **SDK**
3. 点击 **Add SDK** → **JDK**
4. 选择 JDK 8 安装路径
5. 设置 **Project language level** 为 `8 - Lambdas, type annotations etc.`

### Eclipse

1. 打开 **Window** → **Preferences**
2. 选择 **Java** → **Installed JREs**
3. 点击 **Add** → **Standard VM**
4. 设置 JRE home 为 JDK 8 安装路径
5. 勾选新添加的 JDK 作为默认

### VS Code

1. 安装扩展：**Extension Pack for Java**
2. 打开设置（⌘/Ctrl + ,）
3. 搜索 `java.jdt.ls.java.home`
4. 设置为 JDK 8 安装路径

```json
{
  "java.jdt.ls.java.home": "/path/to/jdk-8"
}
```

## 多版本管理

如果需要在多个 JDK 版本间切换，推荐使用 **SDKMAN!**：

```bash
# 安装 SDKMAN!
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# 安装 JDK 8
sdk install java 8.0.392-tem

# 切换版本
sdk use java 8.0.392-tem

# 设置默认版本
sdk default java 8.0.392-tem

# 查看已安装版本
sdk list java
```

## 第一个 Java 程序

创建 `HelloWorld.java`：

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java 8!");

        // JDK 8 特性：Lambda 表达式
        Runnable runnable = () -> System.out.println("Hello from Lambda!");
        runnable.run();
    }
}
```

编译和运行：

```bash
# 编译
javac HelloWorld.java

# 运行
java HelloWorld
```

输出：

```
Hello, Java 8!
Hello from Lambda!
```

## 常见问题

### Q1: 提示 "java" 不是内部或外部命令？

**原因**：环境变量未正确配置

**解决**：

1. 确认 `JAVA_HOME` 指向正确的 JDK 安装目录
2. 确认 `Path` 包含 `%JAVA_HOME%\bin`
3. 重新打开命令行窗口

### Q2: 多个 JDK 版本冲突？

**解决**：使用 SDKMAN!（Linux/macOS）或设置不同的 `JAVA_HOME` 环境变量

### Q3: IDEA 编译报错 "source level" 不兼容？

**解决**：

1. 检查 **Project Structure** → **Project SDK** 是否为 JDK 8
2. 检查 **Module** → **Language Level** 是否为 8

## 下一步

环境配置完成后，开始学习 [Java 基础语法](/docs/java/basic-syntax)！
