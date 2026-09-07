# AI Agent 学习笔记

> 🤖 AI 使用说明：本笔记基于使用 Claude（由 Anthropic 开发，通过 Claude Agent SDK 提供）完成招新题目的实际体验撰写。题目过程中 Claude 协助了 Git 仓库配置、Linux Shell 环境搭建、Makefile 和 CMakeLists.txt 编写等工作。所有代码修改均由本人通过 `git diff` 检查确认，最终结果由本人负责。

---

## 任务 1：使用记录

### 使用的 Agent

Claude（Anthropic），通过 Claude Agent SDK 使用。它能读取项目文件、执行终端命令、搜索代码、修改文档，基本覆盖了本次招新所需的所有场景。

### 安装和配置过程

不需要单独安装，通过 Claude Agent SDK 提供的集成环境直接使用。配置过程很简单：打开环境后，Agent 会自动识别当前工作目录，可以按需调用工具。没有复杂的 IDE 插件安装步骤，上手门槛比较低。

### 交给它的任务

在本次招新中，我把以下任务交给了 Claude：

1. **Git 入门题**：配置 GitHub 账号、创建仓库、初始化本地仓库、推送代码。Claude 帮我写了 `README.md` 和 Git 学习笔记，还指导了我逐步执行 `git init`、`git add`、`git commit`、`git push` 等命令。遇到分支名 `master` 和 `main` 不一致的报错时，它帮我定位了问题并给出修复命令。

2. **Linux & Shell 题**：9 道题目涵盖了文件搜索、日志统计、管道命令、脚本编写、进程管理。Claude 帮我逐题分析题意、探索仓库结构、编写 `output/` 下的答案文件、完成 `scripts/analyze.sh` 和 `scripts/batch-copy.sh` 的实现，并写了 `answers/02.md` 和 `answers/08.md` 的思考题答案。

3. **Make & CMake 题**：补全了 `make-task/Makefile`（用变量 `$(CC)`、自动变量 `$^`、`$@` 实现增量构建）、`cmake-task/CMakeLists.txt`（用 `add_executable` 和 `target_include_directories` 定义构建目标），并写了 `answers/task1.md`（编译流程笔记）和 `answers/task4.md`（思考题+截图+心得）。遇到 Windows 环境下 MSVC 输出 CRLF 导致 `check.sh` 比对失败的问题时，Claude 帮我诊断了根因，并通过 `set_target_properties` 设置输出目录解决了 exe 路径问题。

### Agent 做了哪些修改

通过 `git diff` 我可以看到 Claude 具体修改了什么。比如在 Linux & Shell 题中：

- 创建了 `output/01_project_id.txt` 等 11 个答案文件
- 修改了 `scripts/analyze.sh`，用 `$1` 接收参数、`grep` 统计 ERROR、`awk` 提取错误码
- 修改了 `scripts/batch-copy.sh`，将 `$@` 改为 `"$@"` 修复带空格文件名的问题
- 补全了 `answers/02.md` 和 `answers/08.md` 的文字答案

在 Make & CMake 题中：

- 补全了 `make-task/Makefile` 的四个 TODO，用变量和自动变量实现简洁的构建规则
- 补全了 `cmake-task/CMakeLists.txt`，添加 `add_executable` 和 `target_include_directories`
- 写了 `answers/task1.md`（编译四阶段说明）和 `answers/task4.md`（三个思考题+心得）

### git diff 中我看到了什么

每次修改后我都会运行 `git diff` 检查。比如修复 `batch-copy.sh` 时，diff 显示：

```
-  for file in $@
+  for file in "$@"
```

一行改动，但正是这个引号解决了带空格文件名的问题。这比 Agent 直接告诉我"改好了"要有用得多——看到 diff 之后，我才能真正理解它做了什么，以及为什么这样做。

另一个例子是 Make & CMake 题的 `CMakeLists.txt`，diff 显示新增了两行核心代码：

```
+ add_executable(calculator src/main.c src/calculator.c)
+ target_include_directories(calculator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

简洁清晰，我能在提交前确认它没有做多余的改动。

### 最终结果是否符合预期

符合预期。9 道 Linux Shell 题目全部通过 `./check.sh` 验证，Make & CMake 的构建产物能正确运行并输出预期结果。Claude 的修改都在合理范围内，没有出现大幅重写或引入无关变更的情况。

---

## 任务 2：理解 Agent

### 1. Agent 为什么能读取文件、修改代码，而普通聊天 AI 通常不能？

普通聊天 AI（比如直接在网页上和 ChatGPT 对话）只能处理用户输入的文字，它的输出也只停留在文字层面。它没有"手"，不能打开文件、不能敲终端、不能访问你的项目。

Agent 不同。Agent 背后的大语言模型被赋予了一组工具（Tools），比如读取文件、写文件、执行终端命令、搜索代码等。当 Agent 判断需要操作文件时，它会主动调用这些工具，工具返回的结果再交给模型继续处理。这就是 Agent 能"动手"的原因——不是模型变强了，而是它有了工具。

### 2. Tool 在 Agent 中起到了什么作用？

Tool 是 Agent 连接"思考"和"行动"的桥梁。大语言模型本身只会生成文本，它不会真正打开一个文件、执行一条命令。Tool 把模型的意图转化为实际的文件系统操作。

工作流程大致是：模型分析当前情况，决定需要做什么 → 调用对应的 Tool（比如"读取 `workspace/.project/metadata` 文件"）→ Tool 返回文件内容 → 模型基于返回内容继续推理 → 必要时再调用其他 Tool。

可以理解为：模型是"大脑"，Tool 是"手脚"。没有 Tool 的 Agent 只能坐在椅子上讨论"应该怎么改"，而无法真正动手。

### 3. 为什么项目需要给 Agent 配置一份"员工手册"？

Agent 第一次进入项目时是"两眼一抹黑"的——它不知道项目是做什么的、文件放在哪里、有哪些开发规范、哪些东西不能动。

不配置规则可能导致的问题：

- 把输出文件写到了错误的位置
- 用了项目不推荐的构建方式（比如项目用 CMake，Agent 却生成 Makefile）
- 修改了不该改的文件（比如 `.gitignore`、CI 配置）
- 把密码、Token 等敏感信息写进代码
- 做了大量"自作聪明"的改动，diff 变得难以审查

配置了规则之后，Agent 的行为会被约束在项目的预期范围内。就像新员工入职时先看员工手册，能少走很多弯路。CLAUDE.md 就是一个典型的例子——它告诉 Agent 这个仓库的用途、目录结构、提交前需要检查什么，Agent 读了之后就会按规则行事。

### 4. Agent 为什么可能"忘记"之前说过的内容？额度怎么计算？

大语言模型有上下文窗口（Context Window）的限制。它每次处理请求时，只能看到有限的文本内容——包括之前的对话、代码、Tool 返回结果、项目规则等，所有这些都会占用这个窗口。

当对话变长、代码文件变多、Tool 调用返回大量数据时，上下文会越来越多。如果超过了模型的窗口限制，早期的信息就会丢失。这就是"忘记"的原因——不是模型偷懒，而是它真的看不到了。

额度的计算通常基于 **Token**（词元），而不是字符数。一个英文单词大约 1-2 个 Token，一个中文字大约 1-2 个 Token。模型每次处理时消耗的 Token 数量取决于上下文长度。不同 Agent 平台的免费额度、付费套餐和调用限制各不相同，额度用完就需要升级或等待重置。

### 5. 如果 Agent 可以随便执行任何终端命令，会有什么风险？

风险很大。终端命令的能力是双刃剑：

- **误删文件**：`rm -rf /` 或一条写错的 `rm` 命令可能删掉重要数据。Agent 如果误判了当前目录，后果可能很严重。
- **泄露敏感信息**：Agent 可能把 `.env` 文件、数据库密码、API Key 的内容读出来并写入提交记录，泄露到公开仓库。
- **执行恶意操作**：如果 Agent 被注入攻击（prompt injection），它可能被诱骗执行非预期的命令，比如上传文件到外部服务器。
- **占用系统资源**：无限循环、大量 `fork` 操作可能拖慢或崩溃系统。
- **修改不该改的系统文件**：没有充分授权时，Agent 可能修改系统配置、Git 配置等。

所以好的 Agent 使用习惯是：在不确定时先让 Agent 给出方案，人工确认后再执行；不轻易授权高权限操作；涉及敏感信息时格外小心。

---

## 任务 3：MCP 和 Skills 的探索

在招新过程中，我了解到 Claude Agent SDK 支持通过 MCP（Model Context Protocol）连接外部数据源和服务，通过 Skills 为 Agent 提供特定领域的最佳实践指导。

### Skills

我了解到 Skills 是一种预定义的指令集，可以为特定任务类型（比如写 PPT、生成 Word 文档、处理 PDF）提供最佳实践。在本次招新中，Claude 自动识别了文档生成类的任务并应用了对应的 Skills 指导，比如生成 Markdown 笔记时自动采用了规范的格式。

### MCP

MCP 是一种协议标准，允许 Agent 连接各种数据源——数据库、Slack、Google Sheets、GitHub API 等。通过 MCP，Agent 可以访问实时数据，而不只是依赖训练时学到的知识。比如连接 GitHub MCP 后，Agent 可以直接查询你的仓库状态、创建 Issue、查看 PR。

### 我的想法

这些工具确实能让 Agent 的能力边界大幅扩展。在后续的学习中，我可以尝试配置 GitHub MCP 来实现自动化代码审查，或者连接 Slack 来设置 CI 失败通知。不过对于当前的招新任务来说，核心的 Agent 使用体验已经足够深入，这些扩展可以留到以后的项目中再实践。