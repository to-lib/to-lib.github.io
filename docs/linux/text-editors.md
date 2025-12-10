---
sidebar_position: 10
title: 文本编辑器
---

# Linux 文本编辑器

文本编辑器是 Linux 系统中不可或缺的工具。掌握至少一种终端文本编辑器对系统管理至关重要。

## 编辑器概览

### 常用编辑器对比

| 编辑器 | 难度       | 特点       | 适用场景         |
| ------ | ---------- | ---------- | ---------------- |
| nano   | ⭐         | 简单易用   | 初学者、快速编辑 |
| vim/vi | ⭐⭐⭐⭐   | 强大、高效 | 高级用户、编程   |
| emacs  | ⭐⭐⭐⭐⭐ | 功能丰富   | 重度使用者       |
| pico   | ⭐         | 类似 nano  | 简单编辑         |
| joe    | ⭐⭐       | 模式可切换 | 中级用户         |

## Vim 编辑器

### Vim 基础

```bash
# 安装 vim
sudo apt install vim    # Debian/Ubuntu
sudo yum install vim    # RHEL/CentOS

# 启动 vim
vim filename
vim +10 filename        # 打开并跳到第10行
vim +/pattern filename  # 打开并搜索模式
```

### Vim 模式

Vim 有四种主要模式：

1. **普通模式（Normal Mode）** - 默认模式，用于导航和命令
2. **插入模式（Insert Mode）** - 编辑文本
3. **可视模式（Visual Mode）** - 选择文本
4. **命令模式（Command Mode）** - 执行命令

```bash
# 模式切换
i       # 进入插入模式（当前位置）
I       # 进入插入模式（行首）
a       # 进入插入模式（下一个字符）
A       # 进入插入模式（行尾）
o       # 新建下一行并进入插入模式
O       # 新建上一行并进入插入模式

Esc     # 返回普通模式

v       # 进入可视模式（字符选择）
V       # 进入可视模式（行选择）
Ctrl+v  # 进入可视块模式

:       # 进入命令模式
```

### 移动光标

```bash
# 基本移动
h       # 左移
j       # 下移
k       # 上移
l       # 右移

# 单词移动
w       # 下一个单词开头
b       # 上一个单词开头
e       # 下一个单词结尾

# 行内移动
0       # 行首
^       # 行首非空字符
$       # 行尾
gg      # 文件开头
G       # 文件结尾
:n      # 跳到第 n 行
nG      # 跳到第 n 行

# 屏幕移动
Ctrl+f  # 向下翻页
Ctrl+b  # 向上翻页
Ctrl+d  # 向下半页
Ctrl+u  # 向上半页
H       # 屏幕顶部
M       # 屏幕中间
L       # 屏幕底部
```

### 编辑操作

```bash
# 删除
x       # 删除当前字符
X       # 删除前一个字符
dw      # 删除单词
dd      # 删除整行
D       # 删除到行尾
d$      # 删除到行尾
d0      # 删除到行首
dG      # 删除到文件尾
dgg     # 删除到文件头

# 复制粘贴
yy      # 复制当前行
yw      # 复制单词
y$      # 复制到行尾
p       # 粘贴到下一行
P       # 粘贴到上一行

# 剪切
dd      # 剪切当前行
d       # 剪切选中内容

# 撤销重做
u       # 撤销
Ctrl+r  # 重做

# 修改
r       # 替换当前字符
R       # 进入替换模式
cw      # 修改单词
cc      # 修改整行
C       # 修改到行尾
s       # 删除字符并进入插入模式
S       # 删除行并进入插入模式
```

### 搜索和替换

```bash
# 搜索
/pattern        # 向下搜索
?pattern        # 向上搜索
n               # 下一个匹配
N               # 上一个匹配
*               # 搜索当前单词
#               # 反向搜索当前单词

# 替换
:s/old/new/         # 替换当前行第一个
:s/old/new/g        # 替换当前行所有
:%s/old/new/g       # 替换全文所有
:%s/old/new/gc      # 替换全文所有（确认）
:10,20s/old/new/g   # 替换10-20行

# 高级搜索
:set ic             # 忽略大小写
:set noic           # 区分大小写
:set hlsearch       # 高亮搜索结果
:noh               # 取消高亮
```

### 文件操作

```bash
# 保存退出
:w              # 保存
:w filename     # 另存为
:wq             # 保存并退出
:x              # 保存并退出
ZZ              # 保存并退出

# 退出
:q              # 退出
:q!             # 强制退出（不保存）
:wq!            # 强制保存并退出

# 多文件
:e filename     # 打开文件
:bn             # 下一个文件
:bp             # 上一个文件
:bd             # 关闭当前文件
:ls             # 列出缓冲区
:b2             # 切换到缓冲区2
```

### 分屏操作

```bash
# 水平分屏
:split          # 分割当前文件
:split file     # 分割打开文件
Ctrl+w s        # 分割当前文件

# 垂直分屏
:vsplit         # 垂直分割
:vsplit file    # 垂直分割打开文件
Ctrl+w v        # 垂直分割

# 窗口切换
Ctrl+w w        # 切换窗口
Ctrl+w h        # 左窗口
Ctrl+w j        # 下窗口
Ctrl+w k        # 上窗口
Ctrl+w l        # 右窗口

# 窗口调整
Ctrl+w =        # 均分窗口
Ctrl+w +        # 增加高度
Ctrl+w -        # 减少高度
Ctrl+w >        # 增加宽度
Ctrl+w <        # 减少宽度
```

### 可视模式

```bash
# 进入可视模式
v               # 字符选择
V               # 行选择
Ctrl+v          # 块选择

# 选中后操作
d               # 删除
y               # 复制
c               # 修改
>               # 缩进
<               # 反缩进
```

### Vim 配置

```bash
# vimrc 配置文件
~/.vimrc        # 用户配置
/etc/vim/vimrc  # 全局配置

# 示例配置
" 基础设置
set number              " 显示行号
set relativenumber      " 相对行号
set cursorline          " 高亮当前行
set showcmd             " 显示命令
set showmatch           " 括号匹配

" 缩进
set tabstop=4           " tab 宽度
set shiftwidth=4        " 缩进宽度
set expandtab           " tab 转空格
set autoindent          " 自动缩进
set smartindent         " 智能缩进

" 搜索
set hlsearch            " 高亮搜索
set incsearch           " 增量搜索
set ignorecase          " 忽略大小写
set smartcase           " 智能大小写

" 其他
syntax on               " 语法高亮
set mouse=a             " 启用鼠标
set encoding=utf-8      " 编码
set clipboard=unnamed   " 系统剪贴板
```

### 常用插件

```bash
# 安装 vim-plug 插件管理器
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# 配置插件（~/.vimrc）
call plug#begin('~/.vim/plugged')

Plug 'preservim/nerdtree'           " 文件树
Plug 'vim-airline/vim-airline'      " 状态栏
Plug 'tpope/vim-fugitive'           " Git 集成
Plug 'junegunn/fzf.vim'             " 模糊搜索

call plug#end()

# 安装插件
:PlugInstall

# 更新插件
:PlugUpdate

# 清理插件
:PlugClean
```

## Nano 编辑器

### Nano 基础

```bash
# 安装 nano
sudo apt install nano

# 启动 nano
nano filename
nano +10 filename       # 跳到第10行
```

### Nano 快捷键

```bash
# 文件操作
Ctrl+O          # 保存
Ctrl+X          # 退出
Ctrl+R          # 读取文件

# 编辑操作
Ctrl+K          # 剪切当前行
Ctrl+U          # 粘贴
Alt+6           # 复制当前行
Ctrl+6          # 标记开始
Ctrl+J          # 对齐段落

# 搜索替换
Ctrl+W          # 搜索
Ctrl+\          # 替换
Alt+W           # 下一个匹配

# 导航
Ctrl+Y          # 上翻页
Ctrl+V          # 下翻页
Ctrl+_          # 跳到指定行
Ctrl+A          # 行首
Ctrl+E          # 行尾

# 其他
Ctrl+G          # 帮助
Ctrl+C          # 显示光标位置
Alt+U           # 撤销
Alt+E           # 重做
```

### Nano 配置

```bash
# nano 配置文件
~/.nanorc
/etc/nanorc

# 示例配置
set autoindent          # 自动缩进
set tabsize 4           # tab 宽度
set mouse               # 启用鼠标
set linenumbers         # 显示行号
set multibuffer         # 多文件
set smooth              # 平滑滚动

# 语法高亮
include "/usr/share/nano/*.nanorc"
```

## Emacs 基础

```bash
# 安装 emacs
sudo apt install emacs

# 启动
emacs filename
emacs -nw       # 终端模式

# 基本快捷键
Ctrl+x Ctrl+f   # 打开文件
Ctrl+x Ctrl+s   # 保存
Ctrl+x Ctrl+c   # 退出

# 编辑
Ctrl+k          # 删除到行尾
Ctrl+y          # 粘贴
Ctrl+space      # 标记

# 搜索
Ctrl+s          # 向前搜索
Ctrl+r          # 向后搜索
```

## 选择建议

### 初学者

```bash
# 推荐使用 nano
# - 界面友好
# - 快捷键显示在底部
# - 学习曲线平缓

nano /etc/hosts
```

### 日常管理

```bash
# 推荐学习 vim 基础
# - 系统默认安装
# - 快速高效
# - 广泛使用

vim /etc/nginx/nginx.conf
```

### 专业开发

```bash
# 深入学习 vim 或 emacs
# - 强大的编辑功能
# - 丰富的插件生态
# - 高度可定制
```

## 最佳实践

### 1. 备份重要文件

```bash
# 编辑前备份
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Vim 自动备份
set backup
set backupdir=~/.vim/backup
```

### 2. 使用版本控制

```bash
# 配置文件使用 Git
cd /etc
sudo git init
sudo git add nginx/
sudo git commit -m "Initial nginx config"
```

### 3. 语法检查

```bash
# 编辑配置文件后检查语法
sudo nginx -t
sudo apache2ctl configtest
```

## 总结

本文介绍了 Linux 文本编辑器：

- ✅ Vim 详细教程（模式、命令、插件）
- ✅ Nano 快速使用
- ✅ Emacs 基础
- ✅ 编辑器选择建议
- ✅ 配置和最佳实践

继续学习 [系统管理](./system-admin) 和 [Shell 脚本](./shell-scripting)。
