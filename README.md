# Git 入门练习

> 本文档用于完成招新任务「Git 入门」。

## 仓库用途

本仓库用于记录我学习 Git 基本操作的过程，包括：

- 初次提交代码到 GitHub
- 练习 Git 的常用命令（init、add、commit、push 等）
- 学习版本控制的基本概念

## 目录结构

```
jotang-recruit-2026-git/
├── README.md          # 本文件
├── notes/
│   └── git-notes.md   # Git 学习笔记
└── examples/
    └── hello.py       # 示例代码
```

## 我的学习收获

通过本仓库的练习，我掌握了以下 Git 基本操作：

1. **创建本地仓库**：使用 `git init` 初始化一个仓库
2. **查看状态**：使用 `git status` 查看文件变更情况
3. **添加文件**：使用 `git add` 将文件加入暂存区
4. **提交更改**：使用 `git commit` 将暂存区的文件保存到本地仓库
5. **连接远程仓库**：使用 `git remote add` 绑定 GitHub 仓库
6. **推送到 GitHub**：使用 `git push` 将本地代码同步到远端
7. **查看历史**：使用 `git log` 查看提交记录

## 注意事项

- 不在公开仓库中上传密码、Token、API Key 等敏感信息
- Commit message 尽量描述清楚本次修改内容
- 设置仓库权限为 public，确保评审者可正常访问