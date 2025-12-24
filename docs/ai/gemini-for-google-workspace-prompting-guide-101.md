---
sidebar_position: 4
title: 📋 Gemini Workspace 提示词指南
---

# Gemini for Google Workspace 提示词指南

## October 2024 edition

## 2024 年 10 月版

## Writing effective prompts

## 编写高效提示词

从一开始，Google Workspace (Google 工作区) 的设计理念就是让你能够与他人进行实时协作。现在，你也可以借助 AI——Gemini for Google Workspace (Google Workspace 版 Gemini) 与 AI 协作，在不牺牲隐私与安全的前提下提升效率与创造力。这些内嵌的生成式 AI 功能可以帮助你写作、整理信息、生成图片、加速工作流、提升会议质量等，同时仍可在你熟悉的应用中使用，例如 Gmail (Gmail 邮箱)、Google Docs (Google 文档)、Google Drive (Google 云端硬盘)、Google Sheets (Google 表格)、Google Meet (Google 会议)、Google Slides (Google 幻灯片)，以及 Gemini Advanced (Gemini 高级版；可在 gemini.google.com 使用、具备企业级安全保障的独立对话体验）。Gemini 能在你工作的原处直接使用——它可以访问你在 Drive (云端硬盘)、Docs (文档)、Gmail (邮箱) 等中的个人知识库——因此你可以在 Workspace (工作区) 各应用之间构建强大的工作流，减少切换标签页带来的打断。

本指南将为你提供在使用 Gemini for Workspace (Workspace 版 Gemini) 时编写高效提示词所需的基础技能。你可以把提示词 (prompt) 理解为与 AI 助手开启对话的“开场白”。随着对话推进，你可能会连续输入多个提示词。虽然可能性几乎无穷无尽，但你今天就可以开始应用一些稳定、通用的最佳实践。

编写高效提示词时需要考虑的四个核心要素是：

- 角色

- 任务

- 上下文

- 输出格式

下面是一个同时包含四个要素的提示词示例，适用于 Gmail (Gmail 邮箱) 和 Google Docs (Google 文档)：

你是一名 [行业] 的项目经理。请基于 [相关项目文档的要点/细节]，为 [目标对象/角色] 起草一封高管摘要邮件。请限制为要点列表 (bullet points)。

你不必在每一次提示词里都用到这四个要素，但至少用其中几个会更有帮助。务必记得：在“任务 (Task)”里包含一个动词或明确指令 (例如“总结”“撰写”“改写”)，这是提示词中最关键的组成部分。

如需开始使用 Gemini for Workspace (Workspace 版 Gemini)，请联系销售团队。

以下是帮助你快速上手的小技巧：

1. 使用自然语言。像在和另一个人说话一样写提示词，用完整句子表达完整想法。

2. 具体明确并持续迭代。告诉 Gemini 你希望它做什么 (总结、撰写、改变语气、创作等)，并尽可能提供充足的上下文信息。

3. 简洁清晰，避免复杂表达。用简短但具体的语言提出请求，尽量避免行话或晦涩术语。

4. 把它当作对话。如果结果不符合预期，或你认为仍可改进，就继续微调提示词；使用追问式提示词，并通过“审阅—改写—再审阅”的迭代过程获得更好的结果。

5. 善用你的文档。将 Google Drive (Google 云端硬盘) 中你自己的文件信息加入提示词，让 Gemini 的输出更贴合你的实际情况。

6. 让 Gemini 帮你“润色提示词”。在使用 Gemini Advanced (Gemini 高级版) 时，你可以用这样的开头："Make this a power prompt: [original prompt text here]." Gemini 会给出如何改进提示词的建议。确认改写后的提示词表达了你的真实需求后，再将其粘贴回 Gemini Advanced (Gemini 高级版) 获取输出。

提示词写作是一项人人都能学习的技能。你不必成为“提示词工程师”才能使用生成式 AI。不过，如果第一次没有得到理想结果，你往往需要尝试几种不同的表达方式。根据我们目前从用户那里得到的经验，效果最好的提示词平均约 21 个词，并包含相关上下文；但人们实际尝试的提示词往往少于 9 个词。

生成式 AI 及其无限可能令人兴奋，但它仍处于较新的阶段。即使模型每天都在进步，提示词有时仍会得到不可预测的回应。

在将 Gemini for Workspace (Workspace 版 Gemini) 的输出付诸行动前，请先审阅以确保表达清晰、内容相关且准确。当然，最重要的一点是：生成式 AI 的目的是辅助人类，但最终产出责任仍在你自己。

本指南中的示例提示词仅用于说明与启发。

## 目录

- 编写高效提示词（第 2 页）

- 引言（第 5 页）

- 行政支持（第 7 页）

- 传播与沟通（第 11 页）

- 客户服务（第 15 页）

- 高管（第 20 页）

- 一线管理（第 28 页）

- 人力资源（第 32 页）

- 市场营销（第 37 页）

- 项目管理（第 46 页）

- 销售（第 50 页）

- 小企业主与创业者（第 58 页）

- 初创企业领导者（第 62 页）

- 提示词进阶（第 67 页）

## 引言

### Google Workspace 版 Gemini：提示词入门 101

Gemini for Workspace (Workspace 版 Gemini) 是一个 AI 助手，已集成到你每天使用的应用中——Gmail (Gmail 邮箱)、Google Docs (Google 文档)、Google Sheets (Google 表格)、Google Meet (Google 会议)、Google Slides (Google 幻灯片)，以及 Gemini Advanced (Gemini 高级版；可在 gemini.google.com 使用、具备企业级安全保障的独立对话体验)。这意味着你熟悉的应用可以无缝协同，让你在工作的原位置就能与 Gemini 协作。你可以减少对专注与工作流的打断，更高效地完成任务，并做到一些你原本可能不知道如何着手的事情。

你可以通过多种方式使用 Gemini for Workspace (Workspace 版 Gemini) 的功能。在 Workspace (工作区) 应用中与 Gemini 交互，可以让你基于自己的文件与文档生成高度个性化的内容——即使这些文件并不是 Google Docs (Google 文档)。你可以在几秒钟内引用自己的 Docs (文档) 提取相关上下文生成个性化邮件，也可以根据你的简报或报告中的信息直接生成 Slides (幻灯片)，以及更多。

理解什么是“有效的提示词”，并学会随时随地快速写出提示词，可以显著提升你的效率与创造力。Gemini for Workspace (Workspace 版 Gemini) 可以帮助你：

- 改进写作

- 整理数据

- 创作原创图片

- 总结信息并提炼洞见

- 通过自动记录要点提升会议质量

- 更轻松地研究陌生主题

- 发现趋势、综合信息并识别商业机会

For 25 years, Google has built helpful, secure products that give users choice and control over their data. It’s a bedrock principle for us. This was the case back when we first launched Gmail in 2004, and it remains true in the era of generative AI. This means your data is your data and does not belong to Google. Your data stays in your Workspace environment. Your privacy is protected. Your content is never used for targeting ads or to train or improve Gemini or any other generative AI models.

25 年来，Google 一直在打造既好用又安全的产品，让用户能够对自己的数据拥有选择权与控制权。这是我们的基本原则。从 2004 年首次推出 Gmail (Gmail 邮箱) 时如此，进入生成式 AI 时代仍然如此。这意味着你的数据属于你自己，而不属于 Google；你的数据会留在 Workspace (工作区) 环境中；你的隐私受到保护；你的内容不会被用于广告定向，也不会被用于训练或改进 Gemini 或其他任何生成式 AI 模型。

## How to use this prompt guide

## 如何使用本提示词指南

This guide introduces you to prompting with Gemini for Workspace. It includes strong prompt design examples to help you get started. Additionally, it covers scenarios for different personas, use cases, and potential prompts.

本指南将为你提供在使用 Gemini for Workspace (Workspace 版 Gemini) 时编写高效提示词所需的基础技能。其中包含高质量的提示词设计示例，帮助你快速上手；同时也覆盖不同角色 (personas)、不同使用场景 (use cases) 以及可参考的提示词范例。

You will notice a variety of prompt styles. Some prompts have brackets, which indicate where you would fill in specific details or tag your own personal files by typing @file name. Other prompts are presented without variables highlighted to show you what a full prompt could look like. All of the prompts in this guide are meant to inspire you, but ultimately they will need to be customized to help you with your specific work.

你会看到多种提示词写法。有些提示词带有方括号，表示你需要在这些位置填入具体信息，或通过输入 `@文件名` 来引用你的个人文件。另一些提示词则不突出变量，用于展示一条完整提示词可能是什么样子。本指南中的所有提示词都旨在启发你，但最终仍需要根据你的具体工作进行定制。

To get started, use the role-specific suggested prompts as inspiration to help you unlock a new and powerful way of working.

开始使用时，你可以先参考与自身角色相关的建议提示词，以此为灵感，解锁一种更强大、更高效的工作方式。

Next, learn how you can get started with different features by visiting g.co/gemini/features.

接下来，你可以访问 g.co/gemini/features，了解如何开始使用不同功能。

## Administrative support

## 行政支持

As an administrative support professional, you are responsible for keeping teams on track. You’re required to stay organized and efficient — even under pressure — while juggling many priority tasks.
作为行政支持 (Administrative support) 从业者，你需要让团队保持在正确轨道上推进工作。即使在压力之下，你也必须保持有条理与高效率，同时处理多项高优先级任务。

This section provides you with simple ways to integrate prompts in your daily tasks.

本节将提供一些简单的方法，帮助你把提示词融入日常工作。
本节将提供一些简单的方法，帮助你把提示词更自然地融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.

首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101 (提示词入门 101) 部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace (Google Workspace 版 Gemini). The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.

下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace (Google Workspace 版 Gemini) 协作。“提示词迭代示例 (Prompt iteration example)”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example

提示词迭代示例 (Prompt iteration example)
NEW Use case: Plan agendas (offsite, meetings, and more)

新用例：规划议程（团建、会议等）

You’re planning a three-day offsite meeting. To build an agenda, you brainstorm with Gemini Advanced. You type:

你正在规划一个为期三天的线下团建/异地会议（offsite）。为了制定议程，你与 Gemini Advanced（Gemini 高级版）进行头脑风暴。你输入：

I am an executive administrator to a team director. Our newly formed team now consists of content marketers, digital marketers, and product marketers. We are gathering for the first time at a three-day offsite in Washington, DC. Plan activities for each day that include team bonding activities and time for
deeper strategic work. Create a sample agenda for me. (Gemini Advanced)

我是某团队负责人的行政助理。我们新组建的团队由内容营销人员、数字营销人员和产品营销人员组成。我们将首次在华盛顿特区（Washington, DC）进行为期三天的线下团建/异地会议。请为每天规划活动，既包含团队联结/团建活动，也包含更深入的战略工作时间。请为我创建一个示例议程。（Gemini Advanced／Gemini 高级版）

- Persona • Task • Context • Format

- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini Advanced

## Gemini Advanced（Gemini 高级版）

This is a helpful start to your planning. You need to generate specific ideas for the team bonding activities. You type:

这为你的规划提供了一个不错的起点。你还需要生成更具体的团建活动点子。你输入：

Suggest three different icebreaker activities that encourage people to learn about their teammates’ preferred working styles, strengths, and goals. Make sure the icebreaker ideas are engaging and can be
completed by a group of 25 people in 30 minutes or less. (Gemini Advanced)

请建议三种不同的破冰活动（icebreaker），帮助大家了解队友偏好的工作风格、优势与目标。请确保这些破冰活动有趣且吸引人，并且适合 25 人团队在 30 分钟内完成。（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

## Gemini Advanced（Gemini 高级版）

You are happy with the agenda as a starting point. You now want to reformat Gemini’s response into a table. You type:

你对这份议程作为起点感到满意。现在你希望把 Gemini 的回复重新整理成表格。你输入：

Organize this agenda in a table format. Include one of your suggested icebreakers for each day.
(Gemini Advanced)

请把这份议程整理成表格形式。每天都包含你建议的一项破冰活动。（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

## Gemini Advanced（Gemini 高级版）

You select Export to Docs. You open the newly created Doc. Now, you want to bring in detailed summaries for the strategy sessions using your existing files in Google Drive to provide more context for what will be discussed. You prompt Gemini in Docs and tag your relevant files by typing @file name.

你选择导出到 Docs（Google Docs／Google 文档），并打开新创建的文档。现在，你希望通过 Google Drive（Google 云端硬盘）中的现有文件，为战略讨论环节补充更详细的摘要，以提供更多讨论背景。你在 Docs（Google 文档）的侧边栏中提示 Gemini，并通过输入 `@文件名` 来引用相关文件。

Use @[2024 H2 Team Vision] to generate a summary for the opening remarks on Day 1 of this agenda.
(Gemini in Docs)

（Gemini in Docs／Google 文档中的 Gemini）
请使用 @[2024 H2 Team Vision] 为本议程第 1 天的开场致辞生成一段摘要。（Gemini in Docs／Google 文档中的 Gemini）

## Example use cases

## 示例用例

Executive administrators and executive business partners
高管行政助理与高管业务伙伴
NEW Use case: Manage multiple email inboxes
新用例：管理多个邮箱收件箱

After returning from vacation, you have many unread, unsorted emails. You prompt Gemini in the Gmail side panel. You type:

假期结束后，你有大量未读、未整理的邮件。你在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini。你输入：

Summarize emails from [manager] from the last week. (Gemini in Gmail)

请汇总过去一周来自 [经理] 的邮件。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Gemini returns short summaries of each message. To directly access a message, you click on Sources and see tiles that bring you to specific emails. You select the most important one. Once the email thread opens, you see that many messages were exchanged. You prompt Gemini in Gmail:

Gemini 会为每封邮件返回简短摘要。若要直接打开某封邮件，你可以点击 Sources（来源），看到能跳转到具体邮件的卡片。你选择了最重要的一封。打开邮件会话线程后，你发现其中来往消息很多，于是你在 Gmail（Gmail 邮箱）中继续提示 Gemini：

Summarize this email thread and list all action items and deadlines. (Gemini in Gmail)

请总结这条邮件线程，并列出所有待办事项与截止日期。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

You owe a response to a question, which you believe is best answered by a document in your Drive. You prompt Gemini in the Gmail side panel. You type:

你需要回复对方的一个问题，而你认为最好的回答依据在 Drive（Google Drive／Google 云端硬盘）中的某份文档里。你在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini。你输入：

Generate a response to this email and use @[file name] to describe how the [initiative] can complement
the workstream outlined in [colleague’s name]’s message. (Gemini in Gmail)

请生成对这封邮件的回复，并使用 @[file name] 说明 [initiative] 如何与 [同事姓名] 邮件中描述的工作流/工作流线 (workstream) 互补。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Gemini in Gmail returns a suggested email that pulls directly from your own Doc. After reading it over, you select the Copy icon in the side panel and paste it directly into your message.

Gmail（Gmail 邮箱）中的 Gemini 会返回一封建议邮件，并直接引用你自己的 Doc（Google Docs／Google 文档）内容。你阅读确认后，点击侧边栏的复制图标，将内容直接粘贴到你的邮件中。

## NEW Use case: Plan business travel

## 新用例：规划商务差旅

Your manager has an upcoming meeting that is out of town. You are responsible for booking travel arrangements and creating a personalized itinerary. You need to research places to eat. You brainstorm with Gemini Advanced. You type:
你的经理即将参加一次外地会议。你负责预订出行安排并制定个性化行程；同时需要调研用餐地点。你与 Gemini Advanced（Gemini 高级版）进行头脑风暴。你输入：

I am an executive assistant. I need to create an itinerary for a two-day business trip in [location] during [dates]. My manager is staying at [hotel]. Suggest different options for breakfast and dinner within a 10-minute walk of the hotel, and find one entertainment option such as a movie theater, a local art show,
or a popular tourist attraction. Put it in a table for me. (Gemini Advanced)

我是行政助理。我需要为 [日期] 在 [地点] 的两天商务旅行制定行程安排。我的经理将住在 [酒店]。请推荐酒店 10 分钟步行范围内不同的早餐与晚餐选项，并提供一个娱乐选项，例如电影院、本地艺术展，或热门景点。请用表格形式呈现。（Gemini Advanced／Gemini 高级版）

You continue your conversation until you are happy with the itinerary. Before you make reservations, you want to share the draft with your manager. You select Share & export and select Draft in Gmail. Once the drafted email is created, you put the final touches on the message and send.

你继续对话，直到对行程满意为止。在预订之前，你想把草稿分享给经理。你选择 Share & export（分享与导出），并选择 Draft in Gmail（在 Gmail 中生成草稿）。草稿邮件生成后，你做最后润色并发送。

## NEW Use case: Track travel and entertainment budget

## 新用例：追踪差旅与招待预算

You want to create a spreadsheet to keep track of all of the travel expenses incurred. You open a new Google Sheet and prompt Gemini in the Sheets side panel. You type:
你想创建一张电子表格来追踪全部差旅费用。你打开一个新的 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Create a budget tracker for business travel. It should include columns for: date, expense type (meal,
entertainment, transportation), vendor name, and a description. (Gemini in Sheets)

请创建一个商务差旅预算追踪表。需要包含以下列：日期、费用类型（餐饮/招待/交通）、供应商名称，以及费用说明。（Gemini in Sheets／Google 表格中的 Gemini）

Gemini returns a tracker that is now ready for you to enter data.

Gemini 会返回一张可直接使用的追踪表，方便你立刻开始录入数据。

## Communications

## 传播与沟通

As a communications professional, you are responsible for ensuring your business is well understood by the public. You have to stay up to date with the trends, communicate clearly and effectively with many stakeholders, and build compelling narratives.
作为传播与沟通（Communications）从业者，你负责确保公众能够准确理解你的业务。你需要紧跟趋势，与众多利益相关方清晰、高效地沟通，并构建有说服力的叙事与故事线。

This section provides you with simple ways to integrate prompts in your daily tasks.

本节将提供一些简单的方法，帮助你把提示词融入日常工作。
本节将提供一些简单方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.

下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example

提示词迭代示例 (Prompt iteration example)
NEW Use case: Create a press release

新用例：撰写新闻稿

You are in charge of public relations at a company in the personal care industry. The company you work for has just acquired a smaller brand, and you need to craft a press release. You’ve completed interviews with your company’s CEO, CFO, and the acquired company’s CEO. You’ve stored all of the most important quotes in one Doc. You also have a Doc with all of the information about the acquired brand, its vision, how it got started, and stats. You open a new Doc and prompt Gemini in the Docs side panel and type @file name to reference your relevant files. You type:

你是个人护理行业公司的公关负责人。你的公司刚刚收购了一家小型品牌，你需要撰写一份新闻稿。你已经完成了与公司 CEO、CFO 以及被收购公司 CEO 的访谈，并将所有重要的引用语存储在一个文档中。你还有一份文档，包含被收购品牌的信息、愿景、创立过程以及统计数据。你打开一个新文档，并在 Docs（Google Docs／Google 文档）的侧边栏中提示 Gemini，输入 `@文件名` 来引用相关文件。你输入：

I’m a PR manager. I need to create a press release with a catchy title. Include quotes from
@[VIP Quotes Acquisition]. (Gemini in Docs)

我是一名公关经理。我需要创建一份新闻稿，标题要吸引人。请包含 @[VIP Quotes Acquisition] 中的引用语。（Gemini in Docs／Google 文档中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

## [Gemini returns a response]

## [Gemini 返回回复]

Now you have a starting place for the press release, but you want to include more details about the brand that is being acquired and its founder. This information is stored in your Drive in another file. In the press release Doc, you prompt Gemini in the Docs side panel. You type:

现在你已经有了新闻稿的初稿起点，但你希望加入更多关于被收购品牌及其创始人的细节。这些信息存放在 Drive（Google Drive／Google 云端硬盘）的另一个文件中。在新闻稿文档里，你在 Docs（Google Docs／Google 文档）侧边栏提示 Gemini。你输入：

Use @[Biography and Mission Statement] to add more information about the company that is being
acquired, its mission, and how it got started. (Gemini in Docs)

请使用 @[Biography and Mission Statement] 补充关于被收购公司的信息、使命，以及它是如何创立的。（Gemini in Docs／Google 文档中的 Gemini）

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

The generated paragraphs are a good starting place, so you select Insert to add them into your draft, and you begin making edits to the press release.

生成的段落是一个不错的起点，因此你选择 Insert（插入）将其加入草稿，并开始对新闻稿进行编辑完善。

## Example use cases

## 示例用例

Analyst and public relations

分析师与公共关系
NEW Use case: Prepare for analyst or press briefings

新用例：为分析师或媒体简报做准备

You need to create a brief to prepare a spokesperson for an upcoming meeting with analysts and the media for a new product launch. You open a new Doc and prompt Gemini in the Docs side panel. You type:

你需要创建一份简报，用于帮助发言人为即将到来的分析师与媒体会议（围绕新产品发布）做准备。你打开一个 Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Generate a brief template to prepare [spokesperson] for an upcoming media and analyst briefing for @[Product Launch]. Include space for a synopsis, key messages, and supporting data. (Gemini in Docs)

请生成一个简报模板，用于让 [spokesperson] 为即将到来的媒体与分析师简报（@[Product Launch]）做准备。模板中需要留出：概要 (synopsis)、关键信息 (key messages) 以及支撑数据 (supporting data) 的空间。（Gemini in Docs／Google 文档中的 Gemini）

This gives you a starting point to pull in additional information from your files. You prompt Gemini in the Docs side panel and tag your relevant files by typing @file name. You type:

这为你从文件中补充更多信息提供了起点。你在 Docs（Google Docs／Google 文档）侧边栏提示 Gemini，并通过输入 `@文件名` 来引用相关文件。你输入：

Craft a synopsis of the product launch in three main points using @[Product Launch - Notes].
(Gemini in Docs)

（Gemini in Docs／Google 文档中的 Gemini）
请使用 @[Product Launch - Notes] 将这次产品发布的概要提炼为 3 个要点。（Gemini in Docs／Google 文档中的 Gemini）

You click Insert before repeating the process to fill out the rest of the briefing document. Next, you need to create a spreadsheet of media and analyst contacts. You open a new Google Sheet and prompt Gemini in the Sheets side panel. You type:

你点击 Insert（插入），然后重复该流程以补齐简报文档的其余部分。接下来，你需要创建一张媒体与分析师联系人表格。你打开一个新的 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

Organize my media and analyst contacts from @[Analyst and Journalist Contact Notes] for a new product briefing. I need to keep track of their names, type of contact (analyst or journalist), focus area, the name of the outlet, agency or firm that they work for, and a place where I can indicate the priority level of their
attendance at this briefing (low, medium, high). (Gemini in Sheets)

请将 @[Analyst and Journalist Contact Notes] 中的媒体与分析师联系人整理成适用于新产品简报的表格。我需要追踪：姓名、联系人类型（分析师或记者）、关注领域、媒体/机构名称、其所在的代理公司或机构/公司，以及一个位置用于标注他们参加本次简报的优先级（低/中/高）。（Gemini in Sheets／Google 表格中的 Gemini）

Gemini in Sheets returns a spreadsheet, and you can go through and indicate priority level for each contact. Next, you want to create a slideshow to use during the briefing. You open a new Google Slide and prompt Gemini in the Slides side panel. You tag relevant files by typing @file name in the prompt. You type:

Gemini in Sheets（Google 表格中的 Gemini）会返回一张电子表格，你可以逐一为每位联系人标注优先级。接下来，你想创建在简报中使用的幻灯片。你打开一个新的 Google Slide（Google Slides／Google 幻灯片），并在 Slides（Google 幻灯片）侧边栏提示 Gemini。在提示词中通过输入 `@文件名` 引用相关文件。你输入：

Create a slide describing what [product] is from @[Product Launch - Notes]. Make sure it is short and
easily understood by a broad audience. (Gemini in Slides)

请根据 @[Product Launch - Notes] 创建一张幻灯片，说明 [product] 是什么。确保内容简短，并能被广泛受众轻松理解。（Gemini in Slides／Google 幻灯片中的 Gemini）

Gemini returns a Slide. You continue to build your presentation by using this method to generate additional Slides.

Gemini 会返回一张幻灯片。你继续用这种方法生成更多幻灯片，从而逐步搭建完整演示文稿。

## NEW Use case: Create mock interview questions to prepare spokespeople

## 新用例：创建模拟面试题以帮助发言人准备

Now, you need to prepare your company’s spokesperson for interviews that will follow the briefing. To generate a list of mock interview questions, you decide to chat with Gemini Advanced. You type:

现在，你需要为公司发言人准备简报之后即将到来的采访。为了生成一份模拟面试问题清单，你决定与 Gemini Advanced（Gemini 高级版）对话。你输入：

I am a [PR/AR] manager at [company name]. We just launched [product] and had a briefing where we discussed [key messages]. I am preparing [spokesperson and role/title] for interviews. Generate a list of mock interview questions to help [spokesperson] prepare. Include a mixture of easy and hard questions, with some asking about the basics of [product] and some asking about the long-term vision of [product].
(Gemini Advanced)

我是 [company name] 的一名 [PR/AR] 经理。我们刚刚发布了 [product]，并举办了一场简报会，讨论了 [key messages]。我正在为 [spokesperson and role/title] 的采访做准备。请生成一份模拟面试问题清单，帮助 [spokesperson] 做准备。问题要有难易搭配：既包含关于 [product] 基础信息的问题，也包含关于 [product] 长期愿景的问题。（Gemini Advanced／Gemini 高级版）

Gemini returns a list of questions that can help you prepare your company’s spokesperson. You refine the suggested questions by continuing the conversation with Gemini. Then you select Share & export and Export to Docs. You open the newly created Doc, prompt Gemini in the Docs side panel, and tag relevant files by typing @file name. You type:

Gemini 会返回一份问题清单，帮助你为公司发言人做准备。你继续与 Gemini 对话以优化这些建议问题。随后你选择 Share & export（分享与导出）并 Export to Docs（导出到 Google Docs／Google 文档）。打开新创建的文档后，你在 Docs（Google 文档）侧边栏提示 Gemini，并通过输入 `@文件名` 引用相关文件。你输入：

Use @[Product Launch Notes] to write suggested answers for these questions. Write the talking points
as if you are [title of spokesperson] at [company]. (Gemini in Docs)

请使用 @[Product Launch Notes] 为这些问题撰写建议答案。请以你是 [company] 的 [title of spokesperson] 的口吻来写要点式话术（talking points）。（Gemini in Docs／Google 文档中的 Gemini）

Gemini in Docs returns suggested talking points, and you select Insert to add them into your draft. Now you’re ready to continue tweaking the interview prep for your spokesperson.

Gemini in Docs（Google 文档中的 Gemini）会返回建议的话术要点，你选择 Insert（插入）将其加入草稿。现在你可以继续打磨发言人的采访准备材料。

Communications manager

传播与沟通经理
NEW Use case: Craft internal communications

新用例：撰写内部沟通内容

Your company has redesigned its intranet to be more user friendly. You’re in charge of internal communications for the launch. You want help drafting this message. You open a new Google Doc and prompt Gemini in the Docs side panel. You type:

你们公司重新设计了内网（intranet），让其更易用。你负责此次上线的内部沟通工作，希望获得起草消息的帮助。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

I need to draft a company-wide memo unveiling our relaunched intranet. The [new page] addresses [common feedback we heard from employees] and aims to create a more user friendly experience. Draft an upbeat memo announcing [the new site] using @[Intranet Launch Plan Notes]. (Gemini in Docs)

我需要起草一份全公司范围的备忘录，宣布我们重新上线的内网。新的页面解决了我们从员工那里听到的 [common feedback we heard from employees]，目标是提供更友好的使用体验。请使用 @[Intranet Launch Plan Notes] 起草一封语气积极、令人振奋的备忘录来宣布 [the new site]。（Gemini in Docs／Google 文档中的 Gemini）

Gemini in Docs returns a drafted memo. You refine and edit the text to be exactly as you need it.

Gemini in Docs（Google 文档中的 Gemini）会返回一份备忘录草稿。你继续润色和编辑，使其完全符合你的需求。

## Customer service

## 客户服务

As a customer service professional, you strive to deliver service that’s effortlessly efficient, consistently delightful, and powered by a proactive, helpful team. This section provides you with simple ways to integrate prompts in your daily tasks.

作为客户服务（Customer service）从业者，你致力于提供“高效且省心、稳定且令人愉悦”的服务体验，并依靠一支积极主动、乐于助人的团队来实现。本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.

下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Draft customer communications

用例：撰写客户沟通内容

You’re a customer service representative, and you’re responsible for responding directly to customer inquiries and concerns. You just received an email from a customer who received damaged goods. You open a new Google Doc and click on Help me write to prompt Gemini in Docs. Type the following:

你是一名客户服务代表，负责直接回复客户咨询与关切。你刚收到一封邮件：客户表示收到的商品有损坏。你打开一个新的 Google Doc（Google Docs／Google 文档），点击 Help me write（帮我写）来在 Docs（Google 文档）中提示 Gemini。输入如下内容：

Help me craft an empathetic email response. I am a customer service representative, and I need to create a response to a customer complaint. The customer ordered a pair of headphones that arrived damaged. They’ve already contacted us via email and provided pictures of the damage. I’ve offered a replacement, but they’re requesting an expedited shipping option that isn’t typically included with their order. Include a paragraph that acknowledges their frustration and three bullet points with potential

请帮我起草一封富有同理心的邮件回复。我是一名客户服务代表，需要回复一位客户的投诉。该客户订购了一副耳机，但到货时已损坏。他们已通过邮件联系并提供了损坏照片。我已经提出补发，但他们要求加急配送（通常不包含在订单中）。请包含一段文字来认可他们的挫败感，并给出三个可能的解决方案要点：
resolutions. (Gemini in Docs)

（Gemini in Docs／Google 文档中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Docs: [Drafts email copy]

## Gemini in Docs（Google 文档中的 Gemini）：[起草邮件文案]

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

You like the email that Gemini in Docs created so you select Insert. But you want to brainstorm ways to resolve the issue without offering expedited shipping. You prompt by selecting Help me write. You type:

你很喜欢 Gemini in Docs（Google 文档中的 Gemini）生成的邮件，于是选择 Insert（插入）。但你还想头脑风暴一些不提供加急配送也能解决问题的方式。你点击 Help me write（帮我写）继续提示。你输入：

Suggest 10 alternative options in place of expedited shipping to resolve the customer’s frustration about
receiving the damaged package. (Gemini in Docs)

请提供 10 个替代方案，用来替代加急配送，以缓解客户对收到损坏包裹的失望与不满。（Gemini in Docs／Google 文档中的 Gemini）

## Gemini in Docs: [List of alternative solutions]

## Gemini in Docs（Google 文档中的 Gemini）：[替代方案列表]

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

These 10 suggestions are helpful. You click Insert to add the text into your draft.

这 10 条建议很有帮助。你点击 Insert（插入），将文本加入草稿。

## Example use cases

## 示例用例

Customer Service Manager or Representative

客户服务经理或客户服务代表
NEW Use case: Respond to complex customer issues using FAQ documents

新用例：借助 FAQ 文档回应复杂客户问题

A customer has reached out with a multi-part, complex question. You need to find and use information that is spread across multiple documents in order to respond accurately. You prompt Gemini in the Drive side panel. You type:

一位客户提出了一个包含多部分、较为复杂的问题。为了准确回复，你需要查找并使用分散在多份文档中的信息。你在 Drive（Google Drive／Google 云端硬盘）侧边栏提示 Gemini。你输入：

Summarize information about [product name] including the product’s specific [return policy],
[ingredients], and [certifications]. (Gemini in Drive)

请汇总关于 [product name] 的信息，包括该产品的具体 [return policy]（退货政策）、[ingredients]（成分）以及 [certifications]（认证）。（Gemini in Drive／Google 云端硬盘中的 Gemini）

Gemini returns a summary and links to relevant files, which you can directly click into from the side panel. You read the information before returning to your email to generate a response to the customer. You open the message and prompt Gemini in the Gmail side panel and tag relevant files by typing @file name. You type:

Gemini 会返回摘要及相关文件链接，你可以在侧边栏直接点击打开。你阅读相关信息后回到邮件，准备给客户生成回复。你打开邮件，在 Gmail（Gmail 邮箱）侧边栏提示 Gemini，并通过输入 `@文件名` 引用相关文件。你输入：

Generate a response to the customer question about our [return policy] and [product certifications]
based on @[Customer FAQ Document]. Use a helpful and professional tone. (Gemini in Gmail)

请基于 @[Customer FAQ Document] 生成对客户问题的回复，内容涉及我们的 [return policy]（退货政策）与 [product certifications]（产品认证）。语气要专业且乐于助人。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Standardize communication frameworks

## 用例：标准化沟通框架

You’re a customer service team manager. You need to create scalable resources to standardize your team’s communications. You open a new Google Doc. You brainstorm by prompting Gemini in the Docs side panel. You type:

你是客户服务团队经理，需要创建可规模化的资源来标准化团队沟通方式。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini 进行头脑风暴。你输入：

Draft templates for three different types of customer communication. Create templates for apology emails, order confirmation messages, and thank you notes for loyal customers. Keep each template to
one paragraph and use a friendly tone. (Gemini in Docs)

请为三种不同类型的客户沟通撰写模板：道歉邮件、订单确认消息、以及给忠诚客户的感谢便条。每个模板请控制为一段，并使用友好语气。（Gemini in Docs／Google 文档中的 Gemini）

The suggested templates offer a starting point for you to begin editing and personalizing with elements consistent with your company’s brand and policies. Now you want to outline your team’s communication best practices for onboarding. You open a new Doc and prompt Gemini in Docs. You type:

这些建议模板为你提供了编辑和个性化的起点，你可以结合公司品牌与政策进行调整。现在你想整理一份可用于新人入职培训的团队沟通最佳实践。你打开一个新的 Doc（Google Docs／Google 文档），并在 Docs（Google 文档）中提示 Gemini。你输入：

Craft a list of customer communication best practices that can be used to train new team members. Outline three sections, including how to handle happy customer inquiries, neutral customer inquiries,
and dissatisfied customer inquiries. (Gemini in Docs)

请整理一份客户沟通最佳实践清单，用于培训新团队成员。请划分三部分：如何处理满意客户的咨询、如何处理中立客户的咨询、以及如何处理不满意客户的咨询。（Gemini in Docs／Google 文档中的 Gemini）

You also want to support the team with standardized language that they can use when interacting with customers on phone calls. You prompt Gemini Advanced:

你还希望为团队提供一套标准化话术，以便他们在电话沟通中与客户互动时使用。你提示 Gemini Advanced（Gemini 高级版）：

I am a [customer service manager]. I am trying to create standardized language that the team can use when interacting with customers on phone calls. Generate templates for common call openings, greetings, and closures for a customer service representative at a retail store. These templates should allow for personalization with customer details. The goal is to ensure consistency and professionalism

我是 [customer service manager]（客户服务经理）。我想创建一套标准化话术，供团队在与客户电话沟通时使用。请为零售门店的客户服务代表生成常见的电话开场、问候、以及结束语模板。这些模板应允许根据客户细节进行个性化。目标是在保持一致性与专业性的同时，
while allowing for differentiation with specific customer information. (Gemini Advanced)

并能够结合具体客户信息做出差异化表达。（Gemini Advanced／Gemini 高级版）

## Use case: Improve customer service

## 用例：提升客户服务

You’ve noticed an uptick in customer complaints. You need to collaborate across departments to address recurring issues. You prompt Gemini in Gmail. You type:

你注意到客户投诉有所上升。你需要跨部门协作来解决反复出现的问题。你在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email to my colleagues proposing a meeting to discuss customer experience improvement initiatives. Request that marketing, sales, and product stakeholders meet in the next week to get a clear
sense of roles and responsibilities. (Gemini in Gmail)

请起草一封邮件给我的同事，提议召开一次会议讨论提升客户体验的举措。请邀请市场、销售与产品相关方在下周内开会，以明确角色与职责分工。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

You edit the email and send it to your colleagues. Now you want to create a spreadsheet that you can use to track progress on this cross-departmental initiative. You open a Google Sheet and prompt Gemini in the Sheets side panel. You type:

你编辑邮件并发送给同事。现在你想创建一张电子表格来追踪这项跨部门计划的进展。你打开一个 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

Create a table to track the progress and impact of different customer experience improvement tactics using relevant metrics, including support ticket volume and priority level (high, medium, low).

请创建一张表格，用相关指标来追踪不同客户体验改进策略的进展与影响，指标包括支持工单数量，以及优先级（高/中/低）。
(Gemini in Sheets)
（Gemini in Sheets／Google 表格中的 Gemini）

Customer Support Specialist
客户支持专员
NEW Use case: Analyze customer feedback

新用例：分析客户反馈

You have a spreadsheet that tracks customer feedback. You want to analyze it and brainstorm potential reasons for the trends. You chat with Gemini Advanced. You upload the file and type:

你有一张追踪客户反馈的电子表格。你想分析数据并头脑风暴这些趋势可能的原因。你与 Gemini Advanced（Gemini 高级版）对话，上传文件并输入：

I am a customer support specialist. Using the attached spreadsheet, identify trends and patterns in our [customer feedback] by [category] over [time period]. Identify areas where [customer outreach] has
increased significantly and investigate potential reasons. (Gemini Advanced)

我是客户支持专员。请使用附件电子表格，按 [category]（类别）在 [time period]（时间范围）内识别我们 [customer feedback]（客户反馈）的趋势与模式。请找出 [customer outreach]（客户触达/外联）显著增加的领域，并分析可能原因。（Gemini Advanced／Gemini 高级版）

## Use case: Enable customer self-service

## 用例：启用客户自助服务

Customer feedback has consistently said your return policy guidelines are unclear. You open a Doc that states the return, refund, and store credit policies. You prompt Gemini in Docs by selecting Help me write. You type:

客户反馈长期指出你们的退货政策指引不够清晰。你打开一份 Doc（Google Docs／Google 文档），其中包含退货、退款与门店购物金（store credit）政策。你点击 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Summarize this content to write a clear and concise product return policy and outline 5 steps for
customers to take in sequential order. (Gemini Docs)

请总结这段内容，写出一份清晰简明的产品退货政策，并以顺序方式列出客户需要采取的 5 个步骤。（Gemini in Docs／Google 文档中的 Gemini）

You like how simple the steps are. You repeat the process for your refund policy and store credit policy. Now, you want to use the newly simplified content to create a blog post for customers. Using your Google Doc with the newly written guidance, you prompt Gemini in Google Docs. You type:

你很喜欢这些步骤的简洁明了。你对退款政策与门店购物金政策也重复同样的流程。现在，你想用这些新简化的内容为客户创建一篇博客文章。你使用包含新指导内容的 Google Doc（Google Docs／Google 文档），并在 Google Docs（Google 文档）中提示 Gemini。你输入：

Take this content and turn it into a short blog with the title “Resolve Common Issues Without Agent Assistance.” Have separate sections for our return policy, our refund policy, and our store credit policy.

请将这段内容改写成一篇短博客，标题为“Resolve Common Issues Without Agent Assistance.”（无需客服人员介入即可解决常见问题）。请分别为我们的退货政策、退款政策、以及门店购物金政策设置独立小节。
(Gemini in Docs)
（Gemini in Docs／Google 文档中的 Gemini）

Now you want to create an email template that the team can use when they receive customer questions around these three areas. You open a new Google Doc and prompt Gemini in Docs using Help me write. You type:

现在你想创建一个邮件模板，供团队在收到围绕这三个主题的客户问题时使用。你打开一个新的 Google Doc（Google Docs／Google 文档），使用 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft an email template to a customer that highlights self-service resources referencing [blog link] for [support issues]. Thank the customer for their business and assure them of our commitment to meeting
their needs. (Gemini in Docs)

请起草一封给客户的邮件模板，突出自助资源，并针对 [support issues]（支持问题）引用 [blog link]（博客链接）。感谢客户的支持，并向其保证我们会致力于满足他们的需求。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Conduct voice of the customer research

## 用例：开展 VOC（客户之声）研究

You want to email a dissatisfied customer to attempt to make things right. You open an email that includes a customer complaint. You prompt Gemini in Gmail by selecting Help me write. You type:

你想给一位不满意的客户发邮件，尝试补救并解决问题。你打开一封包含客户投诉的邮件。你点击 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Request a follow-up conversation on [date] at [time] with this customer who provided negative feedback to understand their concern and offer resolutions. Include example solutions. (Gemini in Gmail)

请向这位提供了负面反馈的客户请求在 [date]（日期）[time]（时间）进行一次跟进沟通，以了解其关切并提供解决方案。请包含示例解决方案。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

The drafted response is a nice start, but you want to refine the language. You iterate by prompting Gemini in Gmail using Refine and Elaborate. Next, you want to create a short survey that you can send after each follow-up customer call. You open a new Google Doc and prompt Gemini in Docs. You type:

这份草拟回复是个不错的开始，但你想进一步润色措辞。你在 Gmail（Gmail 邮箱）中使用 Refine（优化）与 Elaborate（扩写）继续迭代。接下来，你想创建一份简短问卷，在每次跟进客户电话后发送。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）中提示 Gemini。你输入：

Create five different questions to customers who have just spoken to an agent on the phone. Questions should gauge how effective the call was, if the customer’s concern was addressed, and if they would
recommend our business to others. (Gemini in Docs)

请为刚刚与客服人员通话过的客户创建 5 个不同的问题。问题应衡量：通话是否有效、客户关切是否得到解决，以及他们是否愿意向他人推荐我们的业务。（Gemini in Docs／Google 文档中的 Gemini）

## Executives

## 高管

As an executive, your time is incredibly constrained. Every decision you make can impact growth, innovation, and the trajectory of your business. Understanding your market and making informed, strategic decisions is paramount, and so is getting urgent tasks done while you’re on the go.

作为高管 (Executives)，你的时间非常有限。你做出的每一个决定都可能影响增长、创新与企业发展轨迹。理解市场并做出有依据的战略决策至关重要；同样重要的是，在出行途中也能高效完成紧急任务。

This section provides you with simple ways to integrate prompts in your daily tasks.
本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Communicate on the go

用例：在路上进行沟通

You are an executive about to board a long flight, and you just received an invitation for the next board meeting with an agenda. You have a couple of comments, and you want to propose adding a few topics to the agenda. You open Gmail, and you prompt Gemini in Gmail. You type:

你即将登上一段长途航班，刚收到下一次董事会会议的邀请（含议程）。你有一些意见，希望提议在议程中加入几个话题。你打开 Gmail（Gmail 邮箱），并在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email confirming that I will be at the board meeting. Ask if we can adjust the agenda to give 15
minutes to [urgent topics]. (Gemini in Gmail)

请起草一封邮件确认我会参加董事会会议，并询问我们是否可以调整议程，为 [urgent topics] 预留 15 分钟时间。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Gmail: [Drafts an email]

## Gemini in Gmail（Gmail 邮箱中的 Gemini）：[起草邮件]

## Gemini in Gmail

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

The email looks good, but you want to make sure the tone is as formal as possible. You select Refine and Formalize.

邮件看起来不错，但你希望语气尽可能正式。你选择 Refine（优化）与 Formalize（正式化）。

## Gemini in Gmail: [Formalizes tone]

## Gemini in Gmail（Gmail 邮箱中的 Gemini）：[正式化语气]

## Gemini in Gmail

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

You read the email and select Insert. Before sending it, you make a light edit to thank the team for keeping you on track while traveling.

你阅读邮件并选择 Insert（插入）。在发送之前，你做了少量编辑，感谢团队在你出行期间帮助你保持进度。

## Example use cases

## 示例用例

Chief Executive Officer

首席执行官 (Chief Executive Officer)
NEW Use case: Enhance personal productivity and time management

新用例：提升个人效率与时间管理

You have important email threads that have numerous responses. You need to quickly catch up. You open the message in Gmail and read the automatically generated summary from Gemini in the Gmail side panel. To respond, you prompt Gemini in the Gmail side panel and tag relevant files by typing @file name. You type:

你有一些重要的邮件线程，回复很多，需要快速跟进。你在 Gmail（Gmail 邮箱）中打开邮件，阅读 Gmail 侧边栏里由 Gemini 自动生成的摘要。要回复邮件，你在 Gmail 侧边栏提示 Gemini，并通过输入 `@文件名` 引用相关文件。你输入：

Generate a response to [person] about [topic]. Include details on [deliverable] and [timeline] using
@[Project A Status Report]. (Gemini in Gmail)

请就 [topic] 给 [person] 生成一封回复，并使用 @[Project A Status Report] 补充 [deliverable] 与 [timeline] 的细节。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## NEW Use case: Create outlines of presentations in seconds

## 新用例：快速生成演示文稿大纲

Your team will pull together a presentation for you, and you want to provide an outline to get them started. You want to generate an outline using Gemini Advanced. You select the microphone icon and use your voice to prompt. You say:

你的团队将为你制作一份演示文稿，你希望先提供一个大纲帮助他们快速开始。你想使用 Gemini Advanced（Gemini 高级版）生成大纲。你点击麦克风图标，用语音进行提示。你说：

I’m the CEO giving a presentation to [audience] at [event], and I want to create a detailed outline for my team to get started. I want to include a few important topics, including [areas of focus] and how our company is innovating with [company initiatives]. I’m envisioning time for a customer Q&A to end the presentation. Include suggested questions we could ask of a customer from the [industry] industry about
how they are using our [product] to achieve [business outcome]. (Gemini Advanced)

我是 CEO，将在 [event] 面向 [audience] 做演讲。我想为团队创建一份详细大纲，帮助他们开始制作。我希望包含一些重要主题，包括 [areas of focus]，以及我们公司如何通过 [company initiatives] 推动创新。我设想在演讲最后安排一段客户问答 (Q&A)。请提供一些建议问题，我们可以向来自 [industry] 行业的客户提问，了解他们如何使用我们的 [product] 来实现 [business outcome]。（Gemini Advanced／Gemini 高级版）

Chief Operating Officer

首席运营官 (Chief Operating Officer)
Use case: Prepare challenging employee communications

用例：准备棘手的员工沟通内容

You’re hosting a quarterly town hall meeting with the entire company. You want to write uplifting remarks to open the meeting. You open a new Doc and prompt Gemini in the Docs side panel. You type:

你要为全公司主持一次季度全员大会（town hall）。你希望写一些鼓舞人心的开场致辞。你打开一个新的 Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Write two uplifting paragraphs for employees who have just finished a challenging quarter. Acknowledge [difficulties] and emphasize [positives] for the upcoming quarter. Use a tone that is motivating, optimistic,
and fosters a sense of unity and collaboration. (Gemini in Docs)

请为刚经历了一个艰难季度的员工写两段鼓舞人心的文字。认可 [difficulties]（困难），并强调下个季度的 [positives]（积极因素）。语气要激励、乐观，并营造团结与协作的氛围。（Gemini in Docs／Google 文档中的 Gemini）

You want to brainstorm and practice how you will respond empathetically to potentially tough questions. You go to Gemini Advanced and type:

你希望头脑风暴并练习：当员工提出可能比较尖锐/棘手的问题时，你该如何以同理心进行回应。你打开 Gemini Advanced（Gemini 高级版）并输入：

I’m the COO of a mid-sized company. I am hosting a quarterly town hall meeting with the entire company. I want to brainstorm and practice how I will respond to potentially tough questions. Help me write challenging questions that employees may ask at the upcoming town hall about [URL of company announcement]. Generate potential answers for each question that use a confident but firm tone. The responses should acknowledge the concern and let the employees know that we are striving to do our
best for the entire company. (Gemini Advanced)

我是某家中型公司的 COO。我要为全公司主持一次季度全员大会（town hall）。我想头脑风暴并练习如何回应可能比较棘手的问题。请帮我写出员工在即将到来的全员大会上，可能会围绕 [URL of company announcement] 提出的尖锐问题，并为每个问题生成一个可能的回答，语气要自信但坚定。回答应认可员工的担忧，并让员工知道我们正在努力为整个公司做到最好。（Gemini Advanced／Gemini 高级版）

## Use case: Streamline responses on the go

## 用例：在路上快速整理与发送回复

Your plans have changed, and you can’t attend a meeting. You need to provide the team with answers on a few key items. You open Gmail and use a voice command to prompt Gemini in Gmail. You say:

你的行程有变，无法参加某个会议，但你需要就几个关键事项给团队反馈。你打开 Gmail（Gmail 邮箱），并用语音指令在 Gmail（Gmail 邮箱）中提示 Gemini。你说：

Draft an email to [project lead] letting them know I will not be in the meeting due to an urgent matter. Ask them to take detailed notes and to ensure the team arrives at a decision on [key topic] in addition
to assigning ownership of the postmortem report to [colleague]. (Gemini in Gmail)

请起草一封邮件给 [project lead]，告知我因紧急事项无法参会。请对方做详细会议记录，并确保团队就 [key topic] 做出决策，同时把复盘报告（postmortem report）的负责人与所有权分配给 [colleague]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Chief Marketing Officer

首席市场官 (Chief Marketing Officer)
NEW Use case: Perform market research and campaign planning

新用例：进行市场调研与活动规划

You’re starting annual planning. You want to conduct research on your target audience. You chat with Gemini Advanced. You type:

你正在进行年度规划，希望对目标受众进行调研。你与 Gemini Advanced（Gemini 高级版）对话。你输入：

I’m a marketing leader conducting analysis in preparation for next year’s [launch]. Define my target audiences [audiences], for my new line of [product]. Include interests, relevant marketing channels, and
top trends that drive their consideration and purchase behavior. (Gemini Advanced)

我是一名市场负责人，正在为明年的 [launch] 做分析准备。请为我的新产品线 [product] 定义目标受众 [audiences]：包括他们的兴趣点、相关营销渠道，以及推动他们产生考虑与购买行为的关键趋势。（Gemini Advanced／Gemini 高级版）

Next, you export your findings to a Doc by selecting Share & export and Export to Docs. Now, you want to pull in relevant data from your own files by typing @file name. You prompt Gemini in the Docs side panel. You type:

接下来，你选择 Share & export（分享与导出）并 Export to Docs（导出到 Google Docs／Google 文档），把调研结果导出到文档中。现在，你希望通过输入 `@文件名` 引用自己文件中的相关数据。你在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Brainstorm value props for my [target audiences] based on features from @[Product Requirements Document]. Include a section on campaign learnings from @[Campaign Performance]. (Gemini in Docs)

请基于 @[Product Requirements Document] 中的功能，为我的 [target audiences] 头脑风暴价值主张（value props）。并加入一节内容，总结来自 @[Campaign Performance] 的活动经验与洞察。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Brainstorm content and thought leadership

## 用例：头脑风暴内容与思想领导力（thought leadership）

You finished a meeting with your social media team leads. You took notes in a Doc about what resonates with your audience, trending topics, target audience data, and keywords that are effective in driving engagement with your brand. You want to brainstorm potential thought leadership pieces using these insights. You prompt Gemini in the Docs side panel. You type:

你刚与社交媒体团队负责人开完会，并在一份 Doc（Google Docs／Google 文档）里记录了：哪些内容能引发受众共鸣、热门话题、目标受众数据，以及能有效提升品牌互动的关键词。你希望基于这些洞察头脑风暴一些思想领导力 (thought leadership) 内容选题。你在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Generate a list of four relevant and engaging thought leadership blog post ideas for [company] based on
trending topics, target audience analysis, and brand keywords. (Gemini in Docs)

请基于热门话题、目标受众分析和品牌关键词，为 [company] 生成 4 个相关且有吸引力的思想领导力博客选题。（Gemini in Docs／Google 文档中的 Gemini）

During the same conversation, the team discussed launching a new brand campaign. You know that your customers value your reliable and unique services, and your company has a long history of delivering for customers. You need help getting started with ideas on a new campaign tagline. You open a new Google Doc and select Help me write. You type:

在同一次讨论中，团队还聊到了要推出新的品牌活动。你知道客户看重你们服务的可靠性与独特性，你们公司也有长期为客户交付成果的历史。你需要一些新活动标语 (tagline) 的灵感。你打开一个新的 Google Doc（Google Docs／Google 文档），选择 Help me write（帮我写）。你输入：

Generate three options for a new slogan emphasizing reliability, innovation, and a long history of
popularity for [company]. (Gemini in Docs)

请生成 3 个新的口号（slogan）选项，强调可靠性、创新性，以及 [company] 长期以来的受欢迎程度。（Gemini in Docs／Google 文档中的 Gemini）

The slogans help you get started with the creative process. You have upcoming events that could be the perfect place to test elements of a new campaign. You want to mock up ideas for booth graphics for your events team. You open a new presentation in Google Slides and select Create image with Gemini. You type:

这些口号能帮助你开启创意流程。你即将参加一些活动，这正是测试新活动元素的好机会。你想为活动团队制作展位视觉 (booth graphics) 的概念草图。你在 Google Slides（Google 幻灯片）中打开一个新的演示文稿，选择 Create image with Gemini（用 Gemini 创建图片）。你输入：

Create an image of a trade show booth using orange and blue colors. The booth should be modern and
showcase interactive computer stations. (Gemini in Slides)

请创建一张使用橙色与蓝色配色的展会展位图片。展位风格应现代，并展示可交互的电脑工作站。（Gemini in Slides／Google 幻灯片中的 Gemini）

## Use case: Conduct competitive analysis

## 用例：进行竞争分析

Your team is considering expanding into a new line of business. To research, you go to Gemini Advanced, and you type:

你的团队正在考虑拓展到一条新的业务线。为进行调研，你打开 Gemini Advanced（Gemini 高级版），并输入：

I am a CMO conducting a competitive analysis. My company is considering expanding into [a new line of business]. Generate a list of the top five competitors in the [industry] industry and include their pricing,
strengths, weaknesses, and target audience. (Gemini Advanced)

我是 CMO，正在进行竞争分析。我的公司正在考虑拓展到 [a new line of business]。请列出 [industry] 行业的 5 个主要竞争对手，并包含他们的定价、优势、劣势以及目标受众。（Gemini Advanced／Gemini 高级版）

After going deeper in your research, you decide to create a five-year strategy to see what this could look like for the company. You type:

在进一步深入调研后，你决定制定一份五年战略，看看这项业务拓展对公司意味着什么。你输入：

Okay, I am going to try to convince my CEO that we should expand into [line of business]. Draft a concise, competitive strategy outline for the next five years for the [industry] industry across North America
markets with potential goals, strategies, and tactics. (Gemini Advanced)

好的，我要尝试说服 CEO 我们应该拓展到 [line of business]。请为北美市场的 [industry] 行业起草一份未来五年的简洁竞争战略大纲，包含可能的目标、战略与战术。（Gemini Advanced／Gemini 高级版）

After iterating to generate an appropriate outline, you fill in additional details and thoughts to make the document your own.

在迭代生成合适的大纲后，你再补充更多细节与想法，使其成为你自己的文档。

Chief Technology Officer

首席技术官 (Chief Technology Officer)
Use case: Summarize emerging technology trends

用例：总结新兴技术趋势

You need to catch up on emerging technology trends as the landscape is shifting quickly. You open Gemini Advanced, and you type:

技术格局变化很快，你需要快速了解新兴技术趋势。你打开 Gemini Advanced（Gemini 高级版），并输入：

I am the CTO of [company] in [industry]. I want to understand emerging technology trends. Summarize the top five emerging technologies with the most significant potential impact on [industry]. For each technology, list its potential benefits and challenges, and suggest how it could impact [company] in the
next two to three years. (Gemini Advanced)

我是 [industry] 行业 [company] 的 CTO。我想了解新兴技术趋势。请总结对 [industry] 可能产生最重大影响的 5 项新兴技术。对于每项技术，请列出潜在收益与挑战，并建议它在未来 2-3 年可能如何影响 [company]。（Gemini Advanced／Gemini 高级版）

You want to dig deeper on specific topics, so you continue the conversation by typing:

你想就某些具体主题进一步深入，于是继续对话并输入：

Recommend three areas where [my company] can take proactive steps to stay ahead of the curve on
[specific areas]. (Gemini Advanced)

请推荐 3 个领域，让 [my company] 能采取主动行动，在以下方面保持领先：[specific areas]。（Gemini Advanced／Gemini 高级版）

Chief Information Officer

首席信息官 (Chief Information Officer)
NEW Use case: Communicate technical topics to non-technical audiences

新用例：向非技术受众解释技术主题

You’re making the case to digitally transform your company by adopting generative AI solutions. You need to present to the CEO and other leadership. You want help in communicating technical topics to non-technical audiences. You chat with Gemini Advanced. You type:

你正在论证：通过采用生成式 AI 解决方案来推动公司数字化转型。你需要向 CEO 与其他管理层做汇报。你希望获得帮助，把技术话题讲给非技术受众听。你与 Gemini Advanced（Gemini 高级版）对话。你输入：

I am the CIO at [company], and I am trying to build the case to [adopt generative AI solutions]. I need to explain the technical concept of generative AI to a non-technical audience (the CEO and board). Help me write talking points that will help me convey what generative AI is, ways it could help us digitally transform, and why it’s important to our growth as a company. Include details about how it could potentially refocus our technical talent on more strategic work, help enhance our workforce’s productivity, and help us better
serve our global workforce and customers. (Gemini Advanced)

我是 [company] 的 CIO，我正在为 [adopt generative AI solutions] 建立论证。我需要向非技术受众（CEO 与董事会）解释“生成式 AI”的技术概念。请帮我写一组要点式话术（talking points），说明：什么是生成式 AI、它如何帮助我们进行数字化转型，以及它为何对公司增长很重要。请包含细节：它如何让技术人才从日常事务转向更具战略性的工作、如何提升员工生产力，以及如何更好地服务全球员工与客户。（Gemini Advanced／Gemini 高级版）

Gemini provides suggested ways to discuss the topic. You continue your brainstorm and then export your conversation by clicking Share & export and Export to Docs. Then, to build a presentation, you open a new Google Slide and prompt Gemini in the Slides side panel and tag relevant files by typing @file name. You type:

Gemini 会给出一些建议的讨论方式。你继续头脑风暴，并点击 Share & export（分享与导出）以及 Export to Docs（导出到 Google Docs／Google 文档）导出对话。随后，为了制作演示文稿，你打开一个新的 Google Slide（Google Slides／Google 幻灯片），在 Slides（Google 幻灯片）侧边栏提示 Gemini，并通过输入 `@文件名` 引用相关文件。你输入：

I need to build a presentation to explain a technical topic to a non-technical audience. Generate an [introduction slide] that [describes what generative AI is] using @[Gen AI Explanation Notes].
(Gemini in Slides)

我需要制作一份演示文稿，用于向非技术受众解释技术主题。请使用 @[Gen AI Explanation Notes] 生成一张 [introduction slide]（引言页），用来[描述什么是生成式 AI]。（Gemini in Slides／Google 幻灯片中的 Gemini）

You continue to use the same prompt, adjusting the topic to generate more slides for your presentation based on your notes.

你继续使用同一个提示词，只是调整主题，从而基于你的笔记为演示文稿生成更多幻灯片。

## NEW Use case: Research vendor products, services, and features

## 新用例：调研供应商产品、服务与功能

You’re working on a report to make a vendor recommendation. You visit Gemini Advanced and type:

你正在撰写一份报告，用于给出供应商推荐。你打开 Gemini Advanced（Gemini 高级版）并输入：

I am the CIO at [company]. We are currently evaluating vendor options to [replatform our intranet]. Right now, we use [vendor], but we are looking to switch because [we are unhappy with limited functionality and account support]. Suggest additional vendor options to consider and include descriptions of their product
and services and key features. (Gemini Advanced)

我是 [company] 的 CIO。我们正在评估供应商选项以 [replatform our intranet]（重构/迁移我们的内网平台）。目前我们使用 [vendor]，但我们考虑更换，因为 [we are unhappy with limited functionality and account support]（对功能受限与客户支持不满意）。请建议可考虑的其他供应商，并包含其产品与服务描述，以及关键功能。（Gemini Advanced／Gemini 高级版）

## Use case: Develop technical summaries

## 用例：撰写技术摘要

Your team just provided a lengthy technical report. You need to summarize it for your CEO. You open the Google Doc with the full report, and you prompt Gemini in the Docs side panel. You type:

你的团队刚提供了一份很长的技术报告。你需要为 CEO 总结其中内容。你打开包含完整报告的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Summarize the key findings and implications of this report for [audience]. Focus on the main [vulnerabilities] identified and the recommended actions to address them. Use a formal tone.
(Gemini in Docs)

请总结这份报告对 [audience] 的关键发现与影响。聚焦报告中识别出的主要 [vulnerabilities]（漏洞/薄弱点）以及建议采取的应对措施。语气要正式。（Gemini in Docs／Google 文档中的 Gemini）

You make light edits to the summary and include it as an executive summary.

你对摘要做少量编辑，并将其作为高管摘要 (executive summary) 加入文档。

## Use case: Track IT assets

## 用例：追踪 IT 资产

Your company needs a quick way to track software access for new hires. You open a new Google Sheet and prompt Gemini in the Sheets side panel. You type:

你的公司需要一种快速方式来追踪新员工的软件访问权限。你打开一个新的 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

Create a tracker of software licenses for employees and include columns for license types, usage rights,
and renewal dates. (Gemini in Sheets)

请创建一张员工软件许可证追踪表，并包含以下列：许可证类型、使用权限，以及续订日期。（Gemini in Sheets／Google 表格中的 Gemini）

Chief Human Resources Officer

首席人力资源官 (Chief Human Resources Officer)
Use case: Demonstrate employee appreciation

用例：表达对员工的认可与感谢

You want to set up a new program to help everyone feel included, appreciated, and acknowledged across the organization. To brainstorm, you open a new Google Doc and prompt Gemini in the Docs side panel. You type:

你想建立一个新项目，让组织内每个人都感到被接纳、被重视并得到认可。为了头脑风暴，你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Brainstorm 10 employee appreciation ideas based on diverse employee interests such as cooking,
gardening, sports, reading, and traveling. (Gemini in Docs)

请基于员工多元兴趣（如烹饪、园艺、运动、阅读和旅行）头脑风暴 10 个员工关怀/表彰创意。（Gemini in Docs／Google 文档中的 Gemini）

Gemini in Docs kick-starts your creativity, and now you have ideas for employee interest clubs and events. You also want to ensure your leadership team is regularly encouraging managers to recognize talent on their teams, so you create email templates they can use as inspiration. You prompt Gemini in Docs by selecting Help me write, and you type:

Gemini in Docs（Google 文档中的 Gemini）为你打开思路，你现在有了关于员工兴趣社团与活动的想法。你还希望确保领导团队能定期鼓励管理者认可团队人才，因此你创建一些可供参考的邮件模板。你在 Docs（Google 文档）中选择 Help me write（帮我写）来提示 Gemini，并输入：

Draft an email template that thanks [employee] for their hard work and [recent accomplishments]. Offer them an extra perk for their dedication, such as [a coffee gift card]. Use an upbeat and professional tone.
(Gemini in Docs)

请起草一封邮件模板，感谢 [employee] 的辛勤工作与 [recent accomplishments]（近期成果）。为其付出提供一项额外福利，例如 [a coffee gift card]（咖啡礼品卡）。语气要积极、专业。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Assess employee satisfaction

## 用例：评估员工满意度

You want to draft an anonymous survey that allows people to openly and honestly assess how they are feeling. To draft questions, you open a new Google Doc and prompt Gemini in the Docs side panel. You type:

你想起草一份匿名问卷，让大家能够开放、坦诚地评估自己的感受。为了撰写问题，你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Draft an anonymous employee satisfaction survey with questions and answer options that touch upon key areas like workload, work-life balance, compensation, and career growth opportunities. Ensure the
questions are clear, concise, and avoid leading answers. (Gemini in Docs)

请起草一份匿名员工满意度问卷，包含问题及答案选项，覆盖工作量、工作与生活平衡、薪酬、以及职业成长机会等关键领域。确保问题清晰、简洁，并避免诱导性答案。（Gemini in Docs／Google 文档中的 Gemini）

You received feedback from 15 senior leaders, and you’ve gathered all of the anonymous results in a Doc. You want to create a summary that you can use in your next call. You prompt Gemini in the Docs side panel. You type:

你收到了 15 位高级领导的反馈，并已将所有匿名结果汇总到一个 Doc（Google Docs／Google 文档）中。你想创建一份总结用于下一次会议。你在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Summarize the results of the employee feedback to identify key themes. (Gemini in Docs)

请总结员工反馈结果，并识别关键主题。（Gemini in Docs／Google 文档中的 Gemini）

## Frontline management

## 一线管理

As a frontline worker manager, your team’s work is indispensable to your organization — your team may not primarily complete its day’s work on a computer, but communication and collaboration remains key.

作为一线岗位管理者 (Frontline management)，你的团队对组织至关重要——他们可能并不主要在电脑上完成日常工作，但沟通与协作依然是关键。

This section provides you with simple ways to integrate prompts in your daily tasks.
本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
NEW Use case: Find accurate information quickly

新用例：快速找到准确信息

A customer just approached you with a question about an ongoing sale. You could use help navigating the numerous files you have access to so that you find the right information quickly. You prompt Gemini in the Drive side panel. You type:

一位顾客刚向你咨询正在进行的促销活动。你可以借助帮助来快速浏览你有权限访问的大量文件，以便迅速找到正确信息。你在 Drive（Google Drive／Google 云端硬盘）侧边栏提示 Gemini。你输入：

Find the document that details the [company name]’s [holiday] sale details. (Gemini in Drive)

请找到包含 [company name] 的 [holiday] 促销活动细则的文档。（Gemini in Drive／Google 云端硬盘中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Drive

## Gemini in Drive（Google 云端硬盘中的 Gemini）

Gemini in Drive returns suggested relevant files. From the side panel, you can directly summarize the files or you can click into a specific document. You open a suggested Doc to help answer the question. You prompt Gemini in the Docs side panel. You type:

Gemini in Drive（Google 云端硬盘中的 Gemini）会返回建议的相关文件。你可以在侧边栏直接总结这些文件，也可以点击打开某个具体文档。你打开一份建议的 Doc（Google Docs／Google 文档）来回答问题，然后在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

How much can customers save on [product type] during this sale? (Gemini in Docs)

在这次促销中，顾客购买 [product type] 最多能节省多少？（Gemini in Docs／Google 文档中的 Gemini）

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

Gemini returns a response, which helps you answer your customer’s question in a timely manner.

Gemini 会返回回复，帮助你及时回答顾客的问题。

## Example use cases

## 示例用例

Retail associate

零售店员 (Retail associate)
NEW Use case: Improve team collaboration by finding and sharing information easily

新用例：通过查找与分享信息轻松提升团队协作

Your store recently updated its return and exchange policies. To find the information, you prompt Gemini in the Drive side panel. You type:

你的门店最近更新了退换货政策。为了找到相关信息，你在 Drive（Google Drive／Google 云端硬盘）侧边栏提示 Gemini。你输入：

Find the document that discusses our new return and exchange policies. (Gemini in Drive)

请找到讨论我们最新退换货政策的文档。（Gemini in Drive／Google 云端硬盘中的 Gemini）

Gemini returns suggested files that are related to the new policies. You directly click into the relevant file. Now, you want to send an email summarizing the document for your colleagues’ future reference. You open your email and prompt Gemini in the Gmail side panel. You type:

Gemini 会返回与新政策相关的建议文件，你直接点击打开相关文件。接下来，你想给同事发一封邮件，总结该文档以便日后查阅。你打开邮件，在 Gmail（Gmail 邮箱）侧边栏提示 Gemini。你输入：

Write an email to my new colleagues summarizing @[Updated Return and Exchange Policy H2 2024].
(Gemini in Gmail)

请给我的新同事写一封邮件，总结 @[Updated Return and Exchange Policy H2 2024]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

You select Insert and further personalize the message before sending it.

你选择 Insert（插入），并在发送前进一步个性化这封邮件。

## NEW Use case: Streamline task management

## 新用例：简化任务管理

You have a list of opening and closing duties that you must perform depending on what shift you are working. You want to keep yourself organized, so you create a tracker using the duties listed in your onboarding Doc. You open a new Google Sheet and prompt Gemini in the Sheets side panel and tag relevant files by typing @file name. You type:

你有一份开店/闭店职责清单，会因你当班的班次而变化。为了保持井然有序，你想基于入职培训文档里的职责列表创建一个追踪表。你打开一个 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏提示 Gemini，通过输入 `@文件名` 引用相关文件。你输入：

I am a retail manager and I need to create a checklist for my opening and closing duties. Create a template with columns for [opening and closing duties] from @[Onboarding New Hire Information].
(Gemini in Sheets)

我是零售经理，我需要为开店和闭店职责创建一份清单。请基于 @[Onboarding New Hire Information] 中的 [opening and closing duties] 创建一个模板，并包含相应列。（Gemini in Sheets／Google 表格中的 Gemini）

Gemini creates a spreadsheet. As you go through your day, you mark different tasks as complete. You have to leave your shift early, but you first need to communicate to the rest of the team what still needs to be done. You open your Gmail and prompt Gemini in the Gmail side panel and tag the spreadsheet you just created. You type:

Gemini 会生成一张电子表格。你在一天工作过程中将不同任务标记为已完成。你需要提前离开班次，但在离开前必须告知团队还有哪些事项未完成。你打开 Gmail（Gmail 邮箱），在 Gmail 侧边栏提示 Gemini，并通过输入 `@文件名` 引用你刚创建的表格。你输入：

Write an email to the team telling them what still needs to be done from the AM shift from @[Opening and
Closing Duties Tracker]. (Gemini in Gmail)

请给团队写一封邮件，说明 AM 班次中还有哪些事项尚未完成，信息来自 @[Opening and Closing Duties Tracker]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Warehouse worker

仓库员工 (Warehouse worker)
NEW Use case: Manage inventory

新用例：管理库存

A customer wants to place a bulk order. You need to check the store’s inventory to see if you have enough to fulfill it. You open your inventory spreadsheet that tracks this information and prompt Gemini in the Sheets side panel. You type:

一位顾客想下一个大额批量订单。你需要检查门店库存，确认是否足够履约。你打开用于追踪库存信息的表格，并在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

How many [units] of [product] do we have left in our inventory? (Gemini in Sheets)

我们的库存里还剩多少 [product] 的 [units]？（Gemini in Sheets／Google 表格中的 Gemini）

## NEW Use case: Manage audits

## 新用例：管理盘点审计

Your warehouse is undergoing an inventory audit, and you’re in charge of verifying any numbers that are misaligned between your inventory tracker product total and what was counted during the audit. You prompt Gemini in the Sheets side panel. You type:

你的仓库正在进行库存盘点审计。你负责核对：库存追踪表中的产品总量与审计盘点结果之间不一致的数字。你在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

Create a formula that helps me calculate the difference between two columns. Which items have a discrepancy in [the total number counted] versus [the quantity on hand]? (Gemini in Sheets)

请创建一个公式，帮助我计算两列数据的差值。哪些商品在 [the total number counted]（盘点数量）与 [the quantity on hand]（账面/现有数量）之间存在差异？（Gemini in Sheets／Google 表格中的 Gemini）

You verify Gemini’s response that there are only a few items whose count did not align to your inventory tracker’s total. You need to write a message to your supervisor telling them that you’re looking into the issue. You open your Gmail and prompt Gemini in the Gmail side panel. You type:

你核对了 Gemini 的回复，确认只有少数商品的数量与库存追踪表总量不一致。你需要给主管写一条消息，说明你正在调查该问题。你打开 Gmail（Gmail 邮箱），并在 Gmail 侧边栏提示 Gemini。你输入：

I’m a warehouse worker managing an audit. Write a message to my supervisor to let them know that I am
looking into the products whose counts are incorrect. (Gemini in Gmail)

我是负责审计的仓库员工。请给我的主管写一条消息，告知我正在调查那些数量不正确的商品。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

The drafted email looks good to go, so you hit send after reviewing.

草拟的邮件看起来没问题，你审阅后点击发送。

## Human resources

## 人力资源

As an HR professional, you’re the backbone of your organization, and you deal with a large volume of confidential and sensitive information. You shape company culture, find and nurture talent, and ensure a positive employee experience. These are no small feats.

作为人力资源 (Human resources) 从业者，你是组织的中坚力量，需要处理大量机密和敏感信息。你塑造公司文化，发掘并培养人才，并确保员工获得积极体验——这些都并非易事。

This section provides you with simple ways to integrate prompts in your day-to-day tasks.
本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Welcome new employees

用例：欢迎新员工

You’re an HR manager working on a presentation script. You have a Google Doc full of notes, bullet points, and topics that you would like to cover. You begin by opening your Google Doc with notes, and you prompt Gemini in Docs.

你是一名 HR 经理，正在准备一份面向新员工的演示稿脚本。你有一份 Google Doc（Google Docs／Google 文档），里面包含笔记、要点和你希望覆盖的主题。你先打开这份笔记文档，然后在 Docs（Google 文档）中提示 Gemini。

I am an HR manager, and I am developing a script for my presentation for new hires. I need to create the script for an onboarding presentation about our company’s commitment to employee development and well-being. Help me draft talking points that showcase why employee mentorship and development are
core values for our company using @[Mission Statement and Core Values]. (Gemini in Docs)

我是 HR 经理，正在为新员工演示制作一份讲稿。我需要为入职培训演示创建脚本，主题是我们公司对员工发展与福祉（well-being）的承诺。请使用 @[Mission Statement and Core Values] 帮我起草要点式话术，说明为什么员工导师制（mentorship）与员工发展是我们公司的核心价值。（Gemini in Docs／Google 文档中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Docs: [Drafts talking points]

## Gemini in Docs（Google 文档中的 Gemini）：[起草要点]

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

You select Insert. Now, you want to add more targeted talking points. You type:
你选择 Insert（插入）。接下来你想补充更有针对性的要点。你输入：

Add four talking points for a new section of the presentation script that explains how we support our employees’ development. Mention our training and certification programs and mentorship opportunities using @[Learning and Development Paths], and write a strong closing statement about our expectation that everyone contributes to a respectful and welcoming workplace. Use a professional tone.
(Gemini in Docs)

请为演示脚本新增一节内容，补充 4 条要点，解释我们如何支持员工发展。请使用 @[Learning and Development Paths] 提及我们的培训与认证项目以及导师机会，并写一句有力的结尾陈述，强调我们期望每个人都为尊重且友好的工作环境做出贡献。语气要专业。（Gemini in Docs／Google 文档中的 Gemini）
（Gemini in Docs／Google 文档中的 Gemini）

## Gemini in Docs: [Adds talking points]

## Gemini in Docs（Google 文档中的 Gemini）：[补充要点]

You add in more details and then you’re ready to create a draft of the Google Slides that will accompany your talking points.
你补充了更多细节，然后准备创建一份与这些要点配套的 Google Slides（Google 幻灯片）草稿。

## Example use cases

## 示例用例

Recruiter
招聘专员（Recruiter）
NEW Use case: Report on recruitment metrics

新用例：汇报招聘指标

The business is growing, and you have a large hiring effort underway. You want to see a holistic view of how your hiring efforts are going. You open your Google Sheet and prompt Gemini in the Sheets side panel. You type:

业务正在增长，你正在推进一项大规模招聘工作。你想从整体上了解招聘进展。你打开 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏提示 Gemini。你输入：

Help me create a formula to calculate the total total number of [hires] by [department].
(Gemini in Sheets)

请帮我创建一个公式，用来按 [department] 统计 [hires] 的总人数。（Gemini in Sheets／Google 表格中的 Gemini）
（Gemini in Sheets／Google 表格中的 Gemini）

You continue your conversation by prompting additional questions. You type:
你继续对话并提出更多问题。你输入：

In what month did we hire the most people? (Gemini in Sheets)

我们在哪个月招聘人数最多？（Gemini in Sheets／Google 表格中的 Gemini）

You continue with your line of questions until you feel ready to write your report.
你沿着这个问题链继续追问，直到你觉得已经准备好撰写报告。

## Use case: Manage the recruiting process

## 用例：管理招聘流程

You want to brainstorm potential ways the company can better manage the recruiting process. You open the team’s Google Doc with recruiting strategies. You prompt Gemini in the Docs side panel. You type:

你想头脑风暴：公司可以如何更好地管理招聘流程。你打开团队关于招聘策略的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏提示 Gemini。你输入：

Create a list of strategies our recruiters can use to improve our existing recruiting process and identify
potential job candidates. (Gemini in Docs)

请列出一份策略清单，帮助招聘人员改进现有招聘流程并识别潜在候选人。（Gemini in Docs／Google 文档中的 Gemini）

After creating a short recommendation for leadership on how the team will improve existing recruiting processes, the team receives guidance for a job opening for a content marketing manager. You open a new Doc and prompt Gemini in Docs. You type:

在为管理层写了一份简短建议（说明团队将如何改进现有招聘流程）之后，团队收到了一份内容营销经理岗位的招聘需求指导。你打开一个新的 Doc（Google Docs／Google 文档），并在 Docs（Google 文档）中提示 Gemini。你输入：

I am opening a new job position on the marketing team. Write a compelling role description for a content marketing manager. Highlight key responsibilities [insert] and requirements, including B2B and B2C content creation, a minimum of five years experience, and a portfolio of writing examples.
(Gemini in Docs)

我正在为市场团队新增一个岗位。请为“内容营销经理（content marketing manager）”撰写一份有吸引力的岗位描述，突出关键职责 [insert] 与要求，包括 B2B 和 B2C 内容创作、至少五年工作经验，以及写作作品集。（Gemini in Docs／Google 文档中的 Gemini）
（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Manage the interview process

## 用例：管理面试流程

You want to prepare questions for phone screen interviews. You decide to prepare by using Gemini Advanced. You upload the relevant file and type:

你想为电话初筛面试准备问题。你决定使用 Gemini Advanced（Gemini 高级版）来准备。你上传相关文件并输入：

I am a recruiter, and I am preparing for candidate interviews. Using the job description in the file I’m uploading, write a list of 20 open-ended interview questions that I can use to screen candidates.

我是一名招聘专员，正在准备候选人面试。请基于我上传文件中的岗位描述，生成 20 个开放式面试问题，用于筛选候选人。（Gemini Advanced／Gemini 高级版）
(Gemini Advanced)

## Use case: Communicate with candidates

## 用例：与候选人沟通

The team has made its hiring decisions. You open the Google Doc with notes on each candidate. You prompt Gemini in Docs by selecting Help me write. You type:

I am writing an email to a job candidate who just finished the interview process. Create a template for an offer letter for the [selected candidate] for the [position] with a request to schedule a call to discuss
benefits, compensation, and start date. (Gemini in Docs)

我正在给刚完成面试流程的候选人写邮件。请为 [selected candidate] 的 [position] 创建一份 offer letter（录用通知）模板，并请求对方安排一次电话沟通，以讨论福利、薪酬与入职日期。（Gemini in Docs／Google 文档中的 Gemini）

Now, you want to generate personalized, empathetic email copy to send to the job candidates who will not receive an offer. You prompt Gemini in Docs by selecting Help me write. You type:

I am writing an email to job candidates who finished the interview process, but who were not selected. Help me write a rejection letter for [candidate] for the [position]. Use an empathetic tone.
(Gemini in Docs)

（Gemini in Docs／Google 文档中的 Gemini）

HR Manager
人力资源经理
NEW Use case: Conduct employee engagement and satisfaction surveys

新用例：开展员工敬业度与满意度调查

You are in charge of building a survey that will go out to all employees. You want to brainstorm ideas on questions to ask. You visit Gemini Advanced and type:

你负责创建一份面向全体员工的问卷。你想头脑风暴一下可以问哪些问题。你访问 Gemini Advanced（Gemini 高级版）并输入：

I am an HR manager in charge of running our enterprise-wide survey at [company] to gauge employee engagement and satisfaction. Generate a list of questions I can use to build the survey.
(Gemini Advanced)

我是 HR 经理，负责在 [company] 开展全公司范围的调查，以评估员工敬业度与满意度。请生成一份我可以用来构建问卷的问题清单。（Gemini Advanced／Gemini 高级版）

Your company has completed its annual employee engagement and satisfaction survey. Now, you want to clean up the data before you analyze it. You go to Gemini Advanced, upload the relevant file, and type:

贵公司已完成年度员工敬业度与满意度调查。现在，你想在分析数据之前清理数据。你打开 Gemini Advanced（Gemini 高级版），上传相关文件并输入：

Help me clean my employee survey spreadsheet. Specifically, fill any blank values in the name column with “Anonymous,” and if the region column shows Headquarters, replace that with HQ. Finally, remove any rows where the satisfaction column is blank. Please generate a new file for me with my cleaned data.
(Gemini Advanced)

请帮我清理员工调查电子表格。具体来说，如果有姓名栏为空，请填入“Anonymous”（匿名）；如果地区栏显示“Headquarters”，请替换为“HQ”。最后，删除满意度栏为空的所有行。请为我生成一个包含清理后数据的新文件。（Gemini Advanced／Gemini 高级版）

## NEW Use case: Create individualized learning and development plans

## 新用例：制定个性化学习与发展计划

You have all of your company’s learning resources stored in your Google Drive. For each new hire, you want to create a tailored learning and development plan. To do this, you prompt Gemini in the Drive side panel. You type:

你已将公司所有学习资源存储在 Google Drive（Google 云端硬盘）中。你想为每位新员工制定量身定制的学习与发展计划。为此，你在 Drive（Google 云端硬盘）侧边栏中提示 Gemini。你输入：

Create a personalized learning and development plan for a new hire who needs to learn about [topic].
Organize it by day and suggest relevant files. (Gemini in Drive)

请为需要学习 [topic] 的新员工创建一个个性化的学习与发展计划。按天安排，并建议相关文件。（Gemini in Drive／Google 云端硬盘中的 Gemini）

## Use case: Onboard employees

## 用例：员工入职

The recruiters have just filled the company’s two open roles. Now, you’re in charge of ensuring the candidates have a smooth onboarding experience. You need help in structuring information for the new hires, so you open a Google Sheet and prompt Gemini in the Sheets side panel. You type:

招聘人员刚刚填补了公司的两个空缺职位。现在，你负责确保候选人拥有顺畅的入职体验。你需要帮助构建新员工信息，因此你打开 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Create a table that outlines a new employee’s first-week schedule, including key meetings, training sessions, and introductions. Provide a column for key contacts and priority level (low, medium, high) for
each activity. (Gemini in Sheets)

请创建一张表格，列出新员工第一周的日程安排，包括关键会议、培训课程和介绍会。为每个活动提供关键联系人和优先级（低、中、高）列。（Gemini in Sheets／Google 表格中的 Gemini）

Gemini in Sheets returns a formatted Google Sheet that you can now fill in with key contacts, meetings, and activities. The conditional formatting makes it easy for you to sort tasks by priority level with color-coded visual cues. Next, you need to create ways for the team to bond. You open a new Google Doc and prompt Gemini in the Docs side panel. You type:

Gemini in Sheets（Google 表格中的 Gemini）返回一个格式化的 Google Sheet，你现在可以在其中填入关键联系人、会议和活动。条件格式使你可以轻松地通过颜色编码的视觉提示按优先级对任务进行排序。接下来，你需要创造团队联结的方式。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini。你输入：

Design a team-bonding activity, such as an office scavenger hunt, to have team members work together
during their team meeting. (Gemini in Docs)

设计一个团队联结活动，例如办公室寻宝游戏，让团队成员在团队会议期间协同工作。（Gemini in Docs／Google 文档中的 Gemini）

Gemini in Docs provides suggestions that help you brainstorm about the scavenger hunt. You tweak the outputs and get the idea approved by the team lead. Now, you need to communicate with the new hires about their first day when they will meet the team. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:

Gemini in Docs（Google 文档中的 Gemini）提供的建议帮助你头脑风暴寻宝游戏。你调整了输出内容，并获得了团队负责人的批准。现在，你需要与新员工沟通他们第一天与团队见面的事宜。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email to the new employees on the [team] to meet the rest of their team and explain the team-
building purposes of the meeting. (Gemini in Gmail)

起草一封发给 [team] 新员工的邮件，以此会见团队其他成员，并解释会议的团队建设目的。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Communicate key findings and draft follow-up surveys

## 用例：沟通关键发现并起草后续调查

Now that you’ve finished onboarding new employees, you need to focus on ensuring that the latest company research data is easily understood by leadership. You’re committed to creating a welcoming environment for all employees where they can develop their skills. You open the Google Doc with the finalized report. You prompt Gemini in Docs by selecting Help me write. You type:

既然完成了新员工入职，你需要专注于确保领导层能够轻松理解最新的公司研究数据。你致力于为所有员工创造一个受欢迎的环境，让他们能够发展技能。你打开包含最终报告的 Google Doc（Google Docs／Google 文档）。你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft an email to senior leadership that summarizes the key findings from our [report]. Include a short
introductory paragraph with bullet points on the most important findings. (Gemini in Docs)

起草一封发给高层领导的邮件，总结我们 [report] 中的关键发现。包含简短的介绍段落，并用要点列出最重要的发现。（Gemini in Docs／Google 文档中的 Gemini）

Gemini in Docs returns a summary with bullet points. You edit it and then use it to email the leadership team. As a follow-up action, you want to understand how changes made to company policies impact the employee experience. You open Gemini in Docs to begin drafting a survey. You select Help me write and type:

Gemini in Docs（Google 文档中的 Gemini）返回带有要点的摘要。你对其进行编辑，然后将其通过邮件发送给领导团队。作为后续行动，你想了解公司政策的变更如何影响员工体验。你打开 Gemini in Docs（Google 文档中的 Gemini）开始起草调查。你选择 Help me write（帮我写）并输入：

Draft an anonymous employee survey with questions and answer options to monitor company progress on
[topics]. (Gemini in Docs)

起草一份包含问题和答案选项的匿名员工调查，以监控公司在 [topics] 上的进展。（Gemini in Docs／Google 文档中的 Gemini）

## Marketing

## 市场营销

As a marketing professional, you’re the creative force behind captivating campaigns, brand experiences, lead generation, and more. You understand the power of data-driven insights, compelling messaging, and connecting with your audience on a deeper level.

作为营销专业人员，你是引人入胜的营销活动、品牌体验、潜在客户开发等背后的创意力量。你了解数据驱动洞察、令人信服的信息以及与受众建立更深层联系的力量。

This section provides you with simple ways to integrate prompts in your day-to-day tasks. For chief marketing officer (CMO) use cases, visit the Executives section of the guide.

本节提供了一些简单的方法，帮助你将提示词整合到日常任务中。有关首席营销官 (CMO) 的用例，请访问指南的高管部分。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.

首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.

下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example

提示词迭代示例 (Prompt iteration example)
Use case: Develop a visual identity

You own your own consulting business and are often hired to function as a brand manager for your clients. You help businesses in a variety of industries. Your customer is getting ready to launch a coffee shop and video game cafe, and you need to kick-start the creative process by developing a visual identity. You want to ideate and provide early thoughts to the rest of the team. You decide to chat with Gemini Advanced. You type:

你拥有一家自己的咨询公司，经常被客户聘请担任品牌经理。你帮助各行各业的企业。你的客户正准备开设一家咖啡店和视频游戏网咖，你需要通过开发视觉识别系统来启动创意流程。你想构思并将初步想法提供给团队的其他成员。你决定与 Gemini Advanced（Gemini 高级版）聊天。你输入：

Generate ideas for a creative and eye-catching logo for my new business, a coffee shop combined with a video game cafe. Generate a logo considering the following:

请为我的新业务——一家结合了咖啡店与视频游戏网咖的店铺，构思一个富有创意且引人注目的 Logo。生成 Logo 时请考虑以下几点：

Dual Concept: The logo needs to clearly signal both the coffee and gaming aspects of the business without being too cluttered.

双重概念：Logo 需要清晰地传达咖啡和游戏这两个业务方面，同时不能显得过于杂乱。

Target Audience: Appeal to a wide range of gamers (casual and enthusiast), as well as coffee lovers seeking a unique hangout spot.

Style Options: I’m open to these approaches — let’s get a few examples in each of these three styles to compare: Modern and Playful: Bold colors, fun graphics, maybe a pixel art aesthetic. Retro-Cool: Think classic arcade style — chunky lettering, neon color inspiration. Sleek and Minimalist: Clean lines,
geometric shapes, a more subtle nod to both themes. (Gemini Advanced)

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini Advanced

You like the retro-cool options. You continue your conversation and you type:
你喜欢复古酷炫的选项。你继续对话并输入：

I like the retro-cool options. Can you provide three more in that same style? (Gemini Advanced)
我喜欢复古酷炫的选项。你能再提供三个同样风格的吗？（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

Now that you have a sense of what the logo could look like, you want to brainstorm names. You type:
既然你对 Logo 的样子有了概念，你想头脑风暴一下名字。你输入：

Write a tagline and 10 potential names for the business to go with these logos. (Gemini Advanced)
请写一条标语和 10 个潜在的业务名称，以搭配这些 Logo。（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

## Example use cases

## 示例用例

Brand Manager
品牌经理
NEW Use case: Develop brand strategy
新用例：制定品牌战略

You are working to refresh your company’s brand architecture and messaging frameworks. To get started, you visit Gemini Advanced and type:
你正致力于更新公司的品牌架构和信息传递框架。为了开始，你访问 Gemini Advanced（Gemini 高级版）并输入：

I am a brand manager at [company]. Help me define a clear and effective brand architecture for [company], considering its diverse products and service portfolio. For additional context, here is our current brand portfolio: [List all existing brands, products, and services]. Here is our company mission and vision: [Provide a brief overview of the company’s mission and vision]. And these are our target audience(s): [describe target audience(s)]. Our desired brand positioning is [explain how the company
wants to be perceived in the market]. (Gemini Advanced)
我是 [company] 的品牌经理。请帮助我为 [company] 定义清晰有效的品牌架构，并考虑到其多元化的产品和服务组合。作为补充背景，这是我们目前的品牌组合：[列出所有现有品牌、产品和服务]。这是我们的公司使命和愿景：[简要概述公司使命和愿景]。这是我们的目标受众：[描述目标受众]。我们期望的品牌定位是 [解释公司希望在市场上被如何感知]。（Gemini Advanced／Gemini 高级版）

## NEW Use case: Brainstorm brand partnerships

## 新用例：头脑风暴品牌合作伙伴关系

You are working on a new brand campaign. You want to identify influencers or complementary brands you could partner with as part of the social amplification plan. You visit Gemini Advanced and type:
你正在策划一个新的品牌活动。你想识别可以合作的网红或互补品牌，作为社交传播计划的一部分。你访问 Gemini Advanced（Gemini 高级版）并输入：

I am a [brand manager] at [company] working to launch a new campaign focused on [topic]. Identify potential types of influencers and complementary brands that [company] could partner with to amplify the [campaign] on social media channels. The goal is to reach a wide audience of [audiences], while
building credibility and driving engagement. (Gemini Advanced)
我是 [company] 的 [brand manager]，正在致力于推出一个专注于 [topic] 的新活动。请识别 [company] 可以合作的潜在网红类型和互补品牌，以便在社交媒体渠道上扩大 [campaign] 的影响力。目标是触达 [audiences] 的广泛受众，同时建立信誉并推动互动。（Gemini Advanced／Gemini 高级版）

## Use case: Conduct market research and identify trends

## 用例：开展市场调查并识别趋势

The landscape in your industry is rapidly changing, and you need to conduct market research to better identify and understand emerging trends. You go to Gemini Advanced, and you type:
你所在行业的格局变化迅速，你需要进行市场调查，以更好地识别和理解新兴趋势。你打开 Gemini Advanced（Gemini 高级版），并输入：

I need to do market research on [industry] industry to identify new trends. Use [URLs] to uncover
emerging trends and shifting consumer preferences. (Gemini Advanced)
我需要对 [industry] 行业进行市场调查以识别新趋势。请使用 [URLs] 来揭示新兴趋势和消费者偏好的转变。（Gemini Advanced／Gemini 高级版）

After completing your research, you and the team have new messaging that you want to A/B test. You want to generate multiple variations of ad copy using Gemini Advanced. You type:
完成调查后，你和团队有了想要进行 A/B 测试的新消息。你想使用 Gemini Advanced（Gemini 高级版）生成多个广告文案变体。你输入：

I need to A/B test new messaging. Here is our messaging: [messaging]. Generate three different variations
of ad copy. (Gemini Advanced)
我需要对新消息进行 A/B 测试。这是我们的消息：[messaging]。请生成三个不同的广告文案变体。（Gemini Advanced／Gemini 高级版）

## Use case: Create and manage content and distribution

## 用例：创建与管理内容及分发

A customer has exciting organizational changes underway. You need to create content to shape the brand narrative of the company as it enters its next era. You open a Google Doc to get started on a blog draft. You prompt Gemini in Docs by selecting Help me write. You type:
一位客户正在进行激动人心的组织变革。由于公司进入了一个新时代，你需要通过创建内容来塑造公司的品牌叙事。你打开 Google Doc（Google Docs／Google 文档），开始起草博客草稿。你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Create a blog draft announcing that [name] is joining [company] as [position]. [Share two or three details from their bio, such as their previous position and company, their involvement in professional
organizations, etc.]. (Gemini in Docs)
起草一篇博客草稿，宣布 [name] 加入 [company] 担任 [position]。[分享两三个简历细节，例如之前的职位和公司，参与的专业组织等]。（Gemini in Docs／Google 文档中的 Gemini）

You also want a way to efficiently track how and where this content is amplified, so you open a Google Sheet. You prompt Gemini in the Sheets side panel. You type:
你还想要一种高效追踪内容传播方式和位置的方法，因此你打开 Google Sheet（Google Sheets／Google 表格）。你在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Create a project tracker for content amplification and include columns for channel, owner, URL,
and priority level (low, medium, high). (Gemini in Sheets)
创建一个用于内容传播的项目追踪表，并包含渠道、负责人、URL 和优先级（低、中、高）列。（Gemini in Sheets／Google 表格中的 Gemini）

Marketing Specialist
营销专员
NEW Use case: Improve collaboration with customers, agencies, and teams
新用例：改善与客户、代理商和团队的协作

You are hosting a meeting discussing an upcoming project with multiple teams and an agency that will complete the project’s design work. You use Gemini in Google Meet and select Take notes with Gemini so that all participants can give their undivided attention to the conversation. After the meeting, Gemini provides a summary of the discussion and pulls out action items to keep the team on track. (Gemini in Meet)
你正在主持一个会议，与多个团队和负责设计工作的代理商讨论即将开展的项目。你在 Google Meet 中使用 Gemini，并选择 Take notes with Gemini（用 Gemini 做笔记），以便所有参与者都能全神贯注于对话。会议结束后，Gemini 提供了讨论摘要并提取了行动项，以确保团队按计划进行。（Gemini in Meet／Google Meet 中的 Gemini）

From the generated Doc with call notes, you want to create a spreadsheet to help keep the team on track. You open a new Google Sheet and prompt Gemini in the Sheets side panel and tag relevant files by typing @file name. You type:
你想根据生成的通话笔记文档创建一个电子表格，以帮助团队保持进度。你打开一个新的 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini，通过输入 `@文件名` 标记相关文件。你输入：

Generate a project tracker using the action items from @[Meeting Notes from Gemini].
(Gemini in Sheets)
（Gemini in Sheets／Google 表格中的 Gemini）

NEW Use case: Analyze social media trends and other data to reduce
time to market
新用例：分析社交媒体趋势和其他数据以缩短上市时间

You want to analyze different data sources and collate findings to help you reduce your time to market. You open Gemini Advanced and type:
你想分析不同的数据源并整理发现，以帮助你缩短上市时间。你打开 Gemini Advanced（Gemini 高级版）并输入：

I am a [marketing specialist] at [company]. We are working on our [go to market] plans for [type of product]. Help me research social media trends around [topics]. Be specific about trending keywords,
top influencer voices, and common themes in popular content. (Gemini Advanced)
我是 [company] 的 [marketing specialist]。我们正在制定 [type of product] 的 [go to market] 计划。请帮我研究围绕 [topics] 的社交媒体趋势。具体说明热门关键词、主要网红声音以及热门内容的共同主题。（Gemini Advanced／Gemini 高级版）

You verify Gemini’s response by selecting the Double-check response option beneath Gemini’s response.
你通过选择 Gemini 回复下方的 Double-check response（核对回复）选项来验证 Gemini 的回复。

Now, you want to review a report you’ve commissioned that surveyed customers from different industries. You continue your conversation with Gemini. You upload the relevant file and type:
现在，你想审查一份委托进行的报告，该报告调查了不同行业的客户。你继续与 Gemini 对话。你上传相关文件并输入：

Analyze the findings in this [report]. I am especially interested in any common themes about [topic] that stand out to you that will help me better position [marketing materials] for [product] for [target audience].
(Gemini Advanced)
分析这份 [report] 中的发现。我特别感兴趣的是围绕 [topic] 的任何共同主题，这些主题能够帮助我更好地为 [target audience] 定位 [product] 的 [marketing materials]。（Gemini Advanced／Gemini 高级版）

## NEW Use case: Perform audience research and develop personas

## 新用例：开展受众研究并开发人物画像

You need to refresh your audience research and persona development as the team updates webpage copy, pitch decks, and other marketing assets. You brainstorm and research using Gemini Advanced. You type:
随着团队更新网页文案、推介演示文稿和其他营销资产，你需要更新受众研究和人物画像开发。你使用 Gemini Advanced（Gemini 高级版）进行头脑风暴和研究。你输入：

I am a marketing specialist focused on [area] at [company]. I need to conduct in-depth audience research so that I can develop convincing marketing artifacts for [personas]. To start, help me generate a comprehensive profile of [target audience]. Include core demographics and psychographics, online platforms they frequent, key pain points [product] could solve, and language and messaging that
resonates with them. (Gemini Advanced)
我是在 [company] 专注于 [area] 的营销专员。我需要进行深入的受众研究，以便能够为 [personas] 开发具有说服力的营销材料。首先，请帮我生成 [target audience] 的综合档案。包括核心人口统计特征和心理特征、他们经常访问的在线平台、[product] 可以解决的关键痛点，以及能引起他们共鸣的语言和信息。（Gemini Advanced／Gemini 高级版）

Digital Marketing Manager
数字营销经理
NEW Use case: Create and optimize copy for search engine marketing (SEM)
新用例：为搜索引擎营销 (SEM) 创建和优化文案

You want to create a robust list of keywords and long-tail keywords and phrases to uncover new opportunities for SEM targeting. You go to Gemini Advanced and type:
你想创建一个强大的关键词以及长尾关键词和短语列表，以发掘 SEM 目标定位的新机会。你打开 Gemini Advanced（Gemini 高级版）并输入：

I am a digital marketing manager at [company]. I am working on SEM ads for [product]. Here are my seed keywords: [list keywords]. Help me generate a list of additional keywords and long-tail keywords and
phrases that can help me maximize ad performance. (Gemini Advanced)
我是 [company] 的数字营销经理。我正在为 [product] 制作 SEM 广告。这是我的种子关键词：[list keywords]。请帮我生成一份包含额外关键词和长尾关键词及短语的列表，以帮助我最大化广告效果。（Gemini Advanced／Gemini 高级版）

After you finish brainstorming your keywords list, you want to generate a few variations of ad copy. You type:
在完成关键词列表的头脑风暴后，你想生成几个广告文案的变体。你输入：

For my SEM campaign, use these keywords as inspiration to generate multiple ad copy variations with different headlines, descriptions, and calls to action for [product]. Use a [tone] tone in the copy.
(Gemini Advanced)
对于我的 SEM 活动，请使用这些关键词作为灵感，为 [product] 生成多个具有不同标题、描述和号召性用语的广告文案变体。文案请使用 [tone] 语气。（Gemini Advanced／Gemini 高级版）

You want to further refine the text according to different audiences, so you type:
你想根据不同的受众进一步完善文本，所以你输入：

Do the same thing, except write new options for [audience], adjust the tone to be [tone] and focus
the copy on highlighting [feature] of [product]. (Gemini Advanced)
做同样的事情，但要为 [audience] 编写新的选项，调整语气为 [tone]，并把文案重点放在突出 [product] 的 [feature] 上。（Gemini Advanced／Gemini 高级版）

## Use case: Draft customer acquisition communications

## 用例：起草客户获取沟通

Email is one of your company’s main channels of direct communication with prospects and customers. You want help getting started with copy for a new email campaign. You open a new Google Doc, and you prompt Gemini in Docs by selecting Help me write. You type:
邮件是贵公司与潜在客户和客户直接沟通的主要渠道之一。你需要帮助来开始一个新的邮件活动的文案。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Write three different email subject lines that reference [audience segments] and our [product]. Make them
catchy but professional. (Gemini in Docs)
写三个不同的邮件主题行，引用 [audience segments] 和我们的 [product]。要是它们既引人注目又专业。（Gemini in Docs／Google 文档中的 Gemini）

Now you want to share the proposed email subject lines with the copywriting team. You open Gmail, and you select Help me write. You type:
现在，你想与文案团队分享提议的邮件主题行。你打开 Gmail（Gmail 邮箱），选择 Help me write（帮我写）。你输入：

Write an email proposing [suggested email subject lines] to the copywriting team. Keep the email short and simple and request feedback by the end of week. Thank them for their help. (Gemini in Gmail)
写一封邮件向文案团队提议 [suggested email subject lines]。邮件要简短明了，并请求在本周末前反馈。感谢他们的帮助。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Generate inbound marketing campaigns

## 用例：生成集客营销活动

The team created a new ebook on best practices for executives using our new solution. You’re creating a landing page to house the gated asset, and you need engaging copy. You open a new Google Doc and select Help me write. You type:
团队制作了一本关于高管使用我们新解决方案的最佳实践的新电子书。你正在创建一个着陆页来存放这个受限资产，你需要引人入胜的文案。你打开一个新的 Google Doc（Google Docs／Google 文档）并选择 Help me write（帮我写）。你输入：

Create compelling copy for a landing page promoting a new [ebook/webinar/free trial and details] designed for an executive target audience. Highlight key benefits and encourage conversions with
persuasive calls to action. (Gemini in Docs)
为针对高管目标受众设计的推广新 [ebook/webinar/free trial and details] 的着陆页创建引人注目的文案。突出关键利益，并用有说服力的号召性用语鼓励转化。（Gemini in Docs／Google 文档中的 Gemini）

The webpage launched, and you’re now running an inbound marketing campaign. You need to nurture the leads that downloaded your latest ebook. You open a new Google Doc, and you prompt Gemini in Docs by selecting Help me write. You type:
网页已上线，你正在进行集客营销活动。你需要培育下载了最新电子书的潜在客户。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Generate copy for a sequence of five automated emails to nurture leads after they download the ebook on [topic]. Personalize emails and encourage further engagement [with other valuable resources or
offers]. (Gemini in Docs)
生成包含五封自动邮件的序列文案，用于在你下载了关于 [topic] 的电子书后培育潜在客户。个性化邮件并鼓励进一步互动 [with other valuable resources or offers]。（Gemini in Docs／Google 文档中的 Gemini）

Content Marketing Manager
内容营销经理
NEW Use case: Deliver personalized content to customers at scale
新用例：大规模向客户交付个性化内容

You want to create copy for a five-step email nurture cadence for your new product. You open a new Google Doc and prompt Gemini in the Docs side panel and tag relevant files by typing @file name. You type:
你想为你的新产品创建一个五步邮件培育节奏的文案。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini，通过输入 `@文件名` 标记相关文件。你输入：

Create a 5-step nurture email cadence to [prospective customers] who have signed up for [our newsletter], with the goal of getting them to [purchase] [product] using @[Product Specific Notes]
and @[Product FAQ]. (Gemini in Docs)
使用 @[Product Specific Notes] 和 @[Product FAQ]，为注册了 [our newsletter] 的 [prospective customers] 创建一个五步邮件培育节奏，目标是让他们 [purchase] [product]。（Gemini in Docs／Google 文档中的 Gemini）

## NEW Use case: Create visuals for ad campaigns

## 新用例：为广告活动创建视觉素材

You want to create visuals to help your creative agency better understand the team’s direction for an upcoming campaign. You open a new Google Slide and prompt Gemini in Slides. You type:
你想创建视觉素材，以帮助你的创意代理商更好地理解团队对于即将开展的活动的方针。你打开一个新的 Google Slide（Google Slides／Google 幻灯片），并在 Slides（Google 幻灯片）中提示 Gemini。你输入：

Help me create inspirational images for a marketing campaign for [type of product]. Images should use [colors] and [natural elements, such as clouds]. Use a [photorealistic] style. (Gemini in Slides)
帮我为 [type of product] 的营销活动创建灵感图片。图片应使用 [colors] 和 [natural elements, such as clouds]。使用 [photorealistic] 风格。（Gemini in Slides／Google 幻灯片中的 Gemini）

## Use case: Generate inspiration for your blog

## 用例：为你的博客产生灵感

You work for a travel company as the content marketing manager for the company’s blog channel. You need to kick-start the brainstorming process for a new blog post. You decide to gather ideas by collaborating with Gemini Advanced. You type:
你在一家旅游公司担任公司博客渠道的内容营销经理。你需要启动新博客文章的头脑风暴流程。你决定通过与 Gemini Advanced（Gemini 高级版）协作来收集想法。你输入：

Suggest blog post topics that would be interesting for people passionate about travel and the tourism industry. Here’s what I want you to focus on: Make the topics unique. There are lots of tourism blogs out there — let’s come up with fresh angles that would stand out. Keep the topics relevant. Tap into current trends or recent challenges/innovations within the tourism industry when brainstorming. I’d like each topic to include:

Target audience: Who would this topic specifically appeal to?

Content outline: A few bullet points with the main ideas the blog post would discuss.

Call to action: Suggest one way to engage the reader at the end of the post. (Gemini Advanced)

You love the initial ideas you were able to create. You also need to focus on generating creative imagery to accompany the copy in the blog. You type:
你喜欢你能够创造出的初步想法。你还需要专注于生成创意图像来配合博客中的文案。你输入：

Create an image of a plane flying above the clouds over mountains and rivers during sunrise that I can use
in the marketing campaign to promote my travel company. (Gemini Advanced)
创建一张飞机在日出时飞越云层、山川和河流的图片，我可以将其用于营销活动中以推广我的旅游公司。（Gemini Advanced／Gemini 高级版）

## Use case: Create social media posts

## 用例：创建社交媒体帖子

You’re focused on creating content that is optimized for social media channels. You need to gather ideas for content targeted to distinct audiences. You open a new Google Doc and prompt Gemini in Docs by selecting Help me write. You type:
你专注于创建针对社交媒体渠道优化的内容。你需要收集针对不同受众的内容想法。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Write three engaging social media posts about [product/service/topic] that would appeal to [target audience]. Keep each social media post to two sentences and include a call to action to visit [our website].
(Gemini in Docs)
写三篇关于 [product/service/topic] 的引人入胜的社交媒体帖子，以吸引 [target audience]。每篇社交媒体帖子限制在两句话以内，并包含访问 [our website] 的号召性用语。（Gemini in Docs／Google 文档中的 Gemini）
（Gemini in Docs／Google 文档中的 Gemini）

You also need to craft social media posts to drive registration for an upcoming event targeting recent grads. You open a new Google Doc and you prompt Gemini in Docs by selecting Help me write. You type:
你还需要制作社交媒体帖子，以推动即将举办的针对应届毕业生的活动的注册。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Create a social media post promoting our upcoming [event name]. Include attention-grabbing language
and relevant hashtags for [audience]. (Gemini in Docs)
创建一篇社交媒体帖子，推广我们即将举行的 [event name]。包含引人注目的语言和针对 [audience] 的相关话题标签。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Create a strategic marketing plan

## 用例：制定战略营销计划

Your company is launching a new app. You need a robust marketing plan, but you want ideas to get started. You chat with Gemini Advanced. You type:
你的公司正在推出一款新应用。你需要一个可靠的营销计划，但你需要想法来起步。你与 Gemini Advanced（Gemini 高级版）聊天。你输入：

I’m developing a marketing plan for a new app that provides [functionality]. My target audience is [audience]. Help me create a plan with a focus on [marketing channels]. Here’s what I’d like you to cover: competitor analysis, ideal marketing channel mix with rationale, budget recommendations, key messaging
ideas, and proposed campaign timeline with KPIs. (Gemini Advanced)
我正在为一款提供 [functionality] 的新应用制定营销计划。我的目标受众是 [audience]。请帮我制定一个专注于 [marketing channels] 的计划。以下是我希望你涵盖的内容：竞争对手分析、理想的营销渠道组合及理由、预算建议、关键信息创意以及建议的活动时间表与 KPI。（Gemini Advanced／Gemini 高级版）

The responses from your chat are helpful in shaping your marketing plan. You need to get the high-level details to your chief marketing officer (CMO). You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
聊天中的回复对制定你的营销计划很有帮助。你需要将高层细节提供给首席营销官 (CMO)。你打开 Gmail（Gmail 邮箱），通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email to the CMO telling them that I will provide a one-pager with a strategic marketing plan for the new app launch project by [date], and it will include an executive summary, overview of the competitive landscape, top marketing channels, and the target demographic for all South American markets.
(Gemini in Gmail)
起草一封发给 CMO 的邮件，告知他们我将在 [date] 之前提供一份关于新应用发布项目战略营销计划的一页纸摘要，其中将包括执行摘要、竞争格局概览、主要营销渠道以及所有南美市场的目标人群。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Project management As the conductor of complex, ever-evolving projects, your mission is to navigate timelines, coordinate teams, and ensure your programs deliver the intended impact.
项目管理 作为复杂、不断演进项目的指挥家，你的使命是掌控时间表、协调团队，并确保项目产生预期的影响。

This section provides you with simple ways to integrate prompts in your daily tasks.
本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Generate user acceptance tests
用例：生成用户验收测试

Your team completed the registration form for a new website, and now you need to generate user acceptance tests (UATs). To start, you visit Gemini Advanced and type:
你的团队完成了新网站的注册表单，现在你需要生成用户验收测试 (UAT)。为了开始，你访问 Gemini Advanced（Gemini 高级版）并输入：

Create a table with 10 user acceptance tests (UAT) for the website registration form. (Gemini Advanced)
请为网站注册表单创建一个包含 10 个用户验收测试 (UAT) 的表格。（Gemini Advanced／Gemini 高级版）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini Advanced

You think the results are a helpful starting point, so you copy the results to a Google Sheet before drafting an email to your colleague who is running the UATs. You want to explain what they need to do. You continue your conversation with Gemini Advanced. You type:
你认为结果是一个有帮助的起点，所以你将结果复制到 Google Sheet 中，然后起草一封邮件发给负责 UAT 的同事。你想解释他们需要做什么。你继续与 Gemini Advanced（Gemini 高级版）对话。你输入：

Draft an email to [my colleague] who is running this UAT and explain what they need to do next.
(Gemini Advanced)
起草一封邮件给负责此 UAT 的 [my colleague]，并解释他们接下来需要做什么。（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

The drafted email provides a helpful starting point, so you export the results to Gmail, and you make edits directly before sending the message to your colleague.
起草的邮件提供了一个有帮助的起点，所以你将结果导出到 Gmail，并在发送给同事之前直接进行编辑。

## Example use cases

## 示例用例

Project Manager
项目经理
Use case: Report on project status
用例：报告项目状态

You just had a lengthy call with all of your project stakeholders, and now you want to summarize what was discussed and follow up with assigned action items. In the Google Doc with the meeting transcript, you prompt Gemini in Docs. You type:
你刚刚与所有项目相关方进行了一次长时间的通话，现在你想总结讨论内容，并未分配的行动项进行后续跟进。在包含会议逐字稿的 Google Doc 中，你提示 Gemini。你输入：

Summarize this call transcript in a short paragraph. In bullet points, highlight the action items, decisions
made, and owners for each item based off of [call transcript]. (Gemini in Docs)
用一个短段落总结这份通话逐字稿。根据 [call transcript]，用要点突出行动项、做出的决定以及每个项目的负责人。（Gemini in Docs／Google 文档中的 Gemini）

You need to update your manager based on the activity from the last call. You want to templatize how your project status updates are delivered. You open a new Google Doc, and you prompt Gemini in Docs by selecting Help me write. You type:
你需要根据上次通话的活动向经理汇报最新情况。你想将项目状态更新的交付方式模板化。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft a project status update email template to send to my manager. Include sections for a summary of key accomplishments this week, any challenges faced, and the top three priorities for next week.
(Gemini in Docs)
起草一份发送给经理的项目状态更新邮件模板。包括本周关键成就总结、面临的挑战以及下周的三大优先事项等部分。（Gemini in Docs／Google 文档中的 Gemini）
（Gemini in Docs／Google 文档中的 Gemini）

The team just hit its key milestones an entire week early. It’s been a challenging project, so you want to gather everyone to celebrate together. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
团队刚刚提前整整一周达成关键里程碑。这是一个充满挑战的项目，所以你想召集大家一起庆祝。你打开 Gmail（Gmail 邮箱），通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Write an invitation for a team lunch to celebrate the progress made on a project and include [date, time, and location]. Thank them for all of their hard work and acknowledge that this has been a challenging
project. (Gemini in Gmail)
写一份团队午餐邀请函，以庆祝项目取得的进展，并包括 [date, time, and location]。感谢他们的辛勤工作，并承认这是一个充满挑战的项目。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Create a project retrospective

## 用例：创建项目回顾

You’ve just wrapped the project, and your senior leadership team needs a project retrospective. To kick-start the process of gathering feedback, you open a Google Doc and prompt Gemini in Docs by selecting Help me write. You type:
你刚刚结束项目，你的高层领导团队需要一份项目回顾。为了启动收集反馈的流程，你打开一个 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

I need to write a report detailing the successes, failures, and lessons learned from [project]. Draft a list of 20 questions to guide a cross-team process investigation. Include questions to uncover what worked, what didn’t, specific process breakdowns, technical issues, communication gaps, or any other potential
contributing factors to the problem or success of the project. (Gemini in Docs)
我需要写一份报告，详细说明 [project] 的成功、失败和吸取的教训。起草一份包含 20 个问题的清单，以指导跨团队流程调查。问题应涵盖有效做法、无效做法、具体的流程故障、技术问题、沟通差距，或任何其他导致项目问题或成功的潜在因素。（Gemini in Docs／Google 文档中的 Gemini）

The questions give you a great starting place. You edit them before sharing with the team for their input. After you gather everyone’s feedback, you want help structuring the report. You prompt Gemini in Docs by selecting Help me write. You type:
这些问题为你提供了一个很好的起点。你在与团队分享以获取输入之前对它们进行编辑。在收集了大家的反馈后，你需要帮助来构建报告结构。你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Summarize this document in two paragraphs. Include high-level information about the project’s goals, the main contributors, the outcomes of the project, and any key successes or failures. (Gemini in Docs)
用两个段落总结这份文档。包括关于项目目标、主要贡献者、项目成果以及任何关键成功或失败的高层信息。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Develop an issue tracker and related communications

## 用例：开发问题追踪器及相关沟通

You need to create a project issue tracker to keep track of risks and solve them in a timely manner. You want to create a template quickly, so you open a new Google Sheet and prompt Gemini in the Sheets side panel. You type:
你需要创建一个项目问题追踪器，以跟踪风险并及时解决它们。你想快速创建一个模板，所以你打开一个新的 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Create a spreadsheet to track project issues, including descriptions, status, assigned owner, and action
items for resolution. (Gemini in Sheets)
创建一个电子表格来跟踪项目问题，包括描述、状态、分配的负责人以及解决问题的行动项。（Gemini in Sheets／Google 表格中的 Gemini）

Before the project fully kicks off, you want to have standardized communication templates at your disposal. For example, you want an email that can be used if an issue arises. You open a new Google Doc and prompt Gemini in Docs by selecting Help me write. You type:
在项目完全启动之前，你希望有标准化的沟通模板可供使用。例如，你希望有一封当问题出现时可以使用的邮件。你打开一个新的 Google Doc（Google Docs／Google 文档），通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft an email template to announce when an issue arises and include causes, solutions, and timelines
to resolve it. (Gemini in Docs)
起草一份邮件模板，用于在问题出现时发布公告，并包括原因、解决方案和解决时间表。（Gemini in Docs／Google 文档中的 Gemini）

You like the template that Gemini in Docs creates, and you want to create an additional, slightly different email template. In the same Google Doc, you prompt Gemini in Docs by selecting Help me write. You type:
你喜欢 Gemini in Docs 创建的模板，你想再创建一个稍微不同的邮件模板。在同一个 Google Doc 中，你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft an email template to a stakeholder to escalate a critical project issue, outlining the impact and
proposed solution. (Gemini in Docs)
起草一封发给相关方的邮件模板，以升级关键项目问题，概述影响和建议的解决方案。（Gemini in Docs／Google 文档中的 Gemini）

Technical Project Manager
技术项目经理
NEW Use case: Create a workback schedule
新用例：创建倒推时间表

You are the technical project manager for a software release. You already have the scope of the project documented. Now, you want to get started on building a workstream tracker and workback schedule. You go to Gemini Advanced and type:
你是软件发布的技术项目经理。你已经记录了项目范围。现在，你想开始构建工作流追踪器和倒推时间表。你访问 Gemini Advanced（Gemini 高级版）并输入：

I am a [technical project manager] at [company] overseeing [project and brief project description]. The project has the following scope: [scope]. Our project goals are: [project goals]. Our project deliverables are: [project deliverables]. Our budget is [budget], and our delivery date is [delivery date]. Help me create a workback schedule to keep the team on track. Include dates for key milestones and
demos. (Gemini Advanced)
我是 [company] 的 [technical project manager]，负责监督 [project and brief project description]。项目范围如下：[scope]。我们的项目目标是：[project goals]。我们的项目交付成果是：[project deliverables]。我们的预算是 [budget]，交付日期是 [delivery date]。请帮我创建一个倒推时间表，以确保团队按计划进行。包括关键里程碑和演示的日期。（Gemini Advanced／Gemini 高级版）

Sales
销售

Understanding your customers inside and out is your ticket to success. You’re in charge of maintaining critical relationships, deciphering buying signals, crafting tailored solutions, driving revenue for the business, and more.
深入了解客户是你成功的关键。你负责维护关键关系、解读购买信号、制定量身定制的解决方案、为企业推动收入增长等等。

This section provides you with simple ways to integrate prompts in your daily tasks.
本节将提供一些简单的方法，帮助你把提示词融入日常工作。
本节将提供一些简单的方法，帮助你把提示词融入日常工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Google Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Google Workspace（Google Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Conduct customer research
用例：开展客户调查

You’re an account executive, and you’ve just been assigned to a new customer. You need a research assistant. You will need to get to know key contacts at the account to begin building trust between your teams, but first, you want to send an introductory email, so you open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
你是一名客户主管，刚刚被分配了一个新客户。你需要一位调查助手。你需要了解客户的关键联系人，以便开始在团队之间建立信任，但首先，你想发送一封介绍信，所以你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Write an email to [name], the new [title] at [company]. Congratulate them on their new role. Introduce me as their contact point at [company name]. Invite them to lunch next week and check if they prefer Monday
or Tuesday. (Gemini in Gmail)
给 [company] 的新任 [title] [name] 写一封邮件。祝贺他们担任新职务。介绍我作为他们在 [company name] 的联系人。邀请他们下周共进午餐，并询问他们是喜欢周一还是周二。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Gmail: [Drafts email]

## Gemini in Gmail

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

This provides a helpful starting point, but you want to try getting an even better response. You click Refine and Formalize.
这提供了一个有帮助的起点，但你想尝试获得更好的回复。你点击 Refine（优化）和 Formalize（正式化）。

## Gemini in Gmail: [Generates refined email suggestions]

## Gemini in Gmail

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

You’re happy with the email, so you click Insert. You read the message one last time, make final light edits directly, and then send the message. Now, you want to learn more about the customer and how it markets itself. To research, you visit Gemini Advanced and type:
你对这封邮件很满意，于是点击 Insert（插入）。你最后读了一遍信息，直接进行最后的简单编辑，然后发送信息。现在，你想更多地了解该客户及其营销方式。为了通过调查，你访问 Gemini Advanced（Gemini 高级版）并输入：

I am an account executive in charge of a new account, [customer name]. I need to do initial research.
What is the market strategy of [customer]? (Gemini Advanced)
我是一名负责新客户 [customer name] 的客户主管。我需要进行初步调查。[customer] 的市场策略是什么？（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

Gemini provides you with useful information to get started in your research. You continue your research by first focusing on news announcements. You gather a list of URLs, and you paste them into your conversation with Gemini Advanced. You type:
Gemini 为你提供了有用的信息，助你开始调查。你通过首先关注新闻公告来继续你的调查。你收集了一份 URL 列表，并将它们粘贴到你与 Gemini Advanced（Gemini 高级版）的对话中。你输入：

[URLs] Summarize these articles. Provide key insights and contextualize why these announcements are
important. (Gemini Advanced)
[URLs] 总结这些文章。提供关键见解，并结合背景说明为什么这些公告很重要。（Gemini Advanced／Gemini 高级版）

Now you have a clear summary of what was announced, why the news is important, and additional insights. Next, you want to better understand the executive who will be your main point of contact. You find a recorded interview featuring the executive. You paste the YouTube URL into your conversation with Gemini Advanced and type:
现在，你对公告内容、新闻重要性以及额外见解有了清晰的总结。接下来，你想更好地了解将成为你主要联系人的高管。你找到了一段该高管的采访录音。你将 YouTube URL 粘贴到你与 Gemini Advanced（Gemini 高级版）的对话中，并输入：

[YouTube URL] Summarize this interview and tell me more about [executive name]. What does [executive]
care about? (Gemini Advanced)
[YouTube URL] 总结这次采访，并告诉我更多关于 [executive name] 的信息。[executive] 关心什么？（Gemini Advanced／Gemini 高级版）

You continue the conversation with additional lines of questioning to build familiarity with your key contact and the account. You prompt:
你通过额外的问题继续对话，以建立对关键联系人和客户的熟悉度。你提示：

Tell me how [company] can help [customer company] with achieving their goals. (Gemini Advanced)
告诉我 [company] 如何帮助 [customer company] 实现他们的目标。（Gemini Advanced／Gemini 高级版）

Once you wrap up your conversation, you export your results into a Google Doc. You open the Google Doc and prompt Gemini in Docs. You type:
结束对话后，你将结果导出到 Google Doc。你打开 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）中提示 Gemini。你输入：

Create an email draft for [customer] explaining why [your company] is the perfect partner for them to
achieve their market goals. (Gemini in Docs)
为 [customer] 创建一封电子邮件草稿，解释为什么 [your company] 是帮助他们实现市场目标的完美合作伙伴。（Gemini in Docs／Google 文档中的 Gemini）

## Example use cases

## 示例用例

Customer Success Manager
客户成功经理
NEW Use case: Map customer journeys
新用例：绘制客户旅程

It’s your first time onboarding a new customer, and you realize you could benefit from creating custom-tailored assets. You open a new Doc and prompt Gemini in the Docs side panel and tag relevant files by typing @file name. You type:
这是你第一次为新客户办理入职手续，你意识到创建量身定制的资产会让你受益匪浅。你打开一个新的 Doc（Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini，通过输入 `@文件名` 标记相关文件。你输入：

Create personalized onboarding materials for [customer]. Use @[Standard Onboarding Documents] and @
[New Customer Migration Notes] to personalize the assets. (Gemini in Docs)
为 [customer] 创建个性化的入职材料。使用 @[Standard Onboarding Documents] 和 @[New Customer Migration Notes] 来个性化资产。（Gemini in Docs／Google 文档中的 Gemini）

Sales Manager
销售经理
NEW Use case: Manage the request for proposal (RFP) process
新用例：管理提案请求 (RFP) 流程

You’ve just received an RFP, and you want to quickly ingest the request as part of your information gathering process. First, you want to do basic research on the company that issued the request. You visit Gemini Advanced and you type:
你刚刚收到了一份 RFP，你想快速摄取该请求，作为你信息收集过程的一部分。首先，你想对发出请求的公司进行基本调查。你访问 Gemini Advanced（Gemini 高级版）并输入：

I just received an RFP from [company]. Before I dive into the RFP, I want your help in conducting research. Give me a business profile of the company including all of the basics (where they are located, what they provide for customers, who their target audience is, any recent news from the company). Be as detailed
as possible as I want to see a full view of [the company]. (Gemini Advanced)
我刚刚收到了 [company] 的 RFP。在深入研究 RFP 之前，我希望你帮我进行调查。给我一份该公司的商业简介，包括所有基本信息（他们位于哪里、他们为客户提供什么、他们的目标受众是谁、公司的任何近期新闻）。请尽可能详细，因为我想看到 [the company] 的全貌。（Gemini Advanced／Gemini 高级版）

Once you finish your research on the company, you want to summarize the RFP. You continue your conversation with Gemini. You type:
完成公司调查后，通过总结 RFP。你继续与 Gemini 对话。你输入：

[URL or uploaded file] I am a sales manager at [company], and this is the RFP we’ve received from [company]. Summarize this content in a few paragraphs. What is the customer seeking, what is the
budget, and when is a response due by? (Gemini Advanced)
[URL or uploaded file] 我是 [company] 的销售经理，这是我们收到的 [company] 的 RFP。用几个段落总结此内容。客户在寻求什么，预算是多少，回复的截止日期是什么时候？（Gemini Advanced／Gemini 高级版）

## NEW Use case: Access information and tools on your phone while on the go

## 新用例：随时随地通过手机访问信息和工具

You are working remotely from your phone. From the mobile app, you open a thread in Gmail and select the Gemini chip to Summarize this email. Gemini quickly provides you with a summary of the back and forth so
that you can focus on the most important points. (Gemini in Gmail)
你正通过手机远程办公。在移动应用中，你打开 Gmail 中的一个会话，并选择 Gemini 芯片来 Summarize this email（总结这封邮件）。Gemini 会快速为你提供往来邮件的摘要，以便你关注最重要的点。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Now, you want to generate a response acknowledging the latest developments. You prompt Gemini in Gmail. You type:
现在，你想生成一个回复，确认最新的进展。你在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Write a response to this email letting [them] know that I’ve received the message and will take [action]
by [Friday]. (Gemini in Gmail)
回复这封邮件，告知 [them] 我已收到信息，并将在 [Friday] 前采取 [action]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Develop customer relationships

## 用例：发展客户关系

Your annual conference is coming up, and your most important prospects will be there. You want to personally invite them to a happy hour. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
你的年度会议即将召开，你最重要的潜在客户将会出席。你想亲自邀请他们参加欢乐时光 (Happy Hour)。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Write an email inviting people interested in [focus area] to our happy hour taking place on [date, time] at
[trade show event]. Include that we specialize in [focus area]. (Gemini in Gmail)
写一封邮件，邀请对 [focus area] 感兴趣的人参加我们在 [trade show event] 举办的欢乐时光，时间是 [date, time]。包括我们要说明我们专注于 [focus area]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Now that the event is over, you want to follow up with customers who came to the happy hour. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
现在活动结束了，你想跟进参加欢乐时光的客户。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email thanking customers for their time at the happy hour on [date, time, location]. End with an invitation to continue the conversations in the next few weeks. Use a friendly tone. (Gemini in Gmail)
起草一封邮件，感谢客户在 [date, time, location] 参加欢乐时光。最后邀请他们在接下来的几周内继续对话。使用友好的语气。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

You want to check in with the customers who attended workshops at the conference because their early feedback is important. You prompt Gemini in Docs. You type:
你想跟进参加会议研讨会的客户，因为他们的早期反馈很重要。你在 Docs（Google 文档）中提示 Gemini。你输入：

Draft 10 questions that I can use to survey customers about their recent experience with our [product/ service]. Include questions to gauge how useful [the product] is, what they liked, and what they thought
could use improvement. (Gemini in Docs)
起草 10 个问题，用于调查客户对我们 [product/ service] 的近期体验。包括评估 [the product] 有多大用处、他们喜欢什么以及他们认为哪些方面可以改进的问题。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Support the sales team

## 用例：支持销售团队

You need to contact all of your team leads in the Southeast region to provide immediate guidance on how to proactively reach out to customers about an ongoing issue. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
你需要联系东南大区的所有团队负责人，就如何主动联系客户解决当前问题提供即时指导。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email to all Southeast region sales leads. Inform them of [issues]. Advise them to communicate with their teams to contact their customers and offer a 20% discount on a future order as an apology.
(Gemini in Gmail)
起草一封发给所有东南大区销售负责人的邮件。通知他们 [issues]。建议他们与其团队沟通，联系客户并提供 20% 的未来订单折扣作为致歉。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Now, you need to email all of the regional team members. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
现在，你需要给所有区域团队成员发邮件。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Draft an email to the regional sales representatives about an urgent meeting that needs to take place next week about the [issues]. Ask them to provide availability on Monday or Tuesday. (Gemini in Gmail)
起草一封发给区域销售代表的邮件，关于下周需要就 [issues] 召开的紧急会议。请他们提供周一或周二的空闲时间。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Coach and train the sales team

## 用例：辅导和培训销售团队

You’ve heard from many team members that they want more learning opportunities. You’re organizing a half-day learning program to support this request. You need to create a schedule, so you open a new Google Doc and
prompt Gemini in Docs by selecting Help me write. You type:
你从许多团队成员那里听说他们想要更多的学习机会。你正在组织一个半天的学习项目来支持这一请求。你需要创建一个时间表，所以你打开一个新的 Google Doc（Google Docs／Google 文档），并通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Create a half-day agenda for an educational session on our latest technology [products] for sales teams. Include time for the product development team to present and include time for lunch. (Gemini in Docs)
为销售团队创建一个关于我们最新技术 [products] 的半天教育课程议程。包括产品开发团队演示的时间和午餐时间。（Gemini in Docs／Google 文档中的 Gemini）

As a follow up to the team meeting, you want to highlight different learning opportunities available. You open Google Sheets and prompt Gemini in the Sheets side panel. You type:
作为团队会议的后续，你想重点介绍可用的不同学习机会。你打开 Google Sheets（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Create a spreadsheet that tracks online courses for sellers. Include columns for the course’s main topic,
price, duration, and priority level. (Gemini in Sheets)
创建一个跟踪销售人员在线课程的电子表格。包括课程主题、价格、时长和优先级列。（Gemini in Sheets／Google 表格中的 Gemini）

Account Manager and Account Executive
客户经理和客户主管
NEW Use case: Improve collaboration and execution by customizing

## sales materials

新用例：通过定制销售材料来改善协作和执行

You are having an important meeting with a customer. From Google Meet, you turn on Transcription and activate Gemini in Meet by selecting Take notes with Gemini. The transcript provides an unedited Doc of what was said. The Take notes with Gemini file will generate notes recapping the meeting, important topics discussed, and action items. Now, you can fully engage with the customer conversation. (Gemini in Meet)
你正在与客户进行一次重要会议。在 Google Meet 中，你开启转录功能，并通过选择 Take notes with Gemini（用 Gemini 做笔记）激活 Meet 中的 Gemini。转录提供了所说内容的未编辑文档。Take notes with Gemini（用 Gemini 做笔记）文件将生成回顾会议、讨论的重要主题和行动项的笔记。现在，你可以完全投入到与客户的对话中。（Gemini in Meet／Google Meet 中的 Gemini）

After the call, you want to send a recap message to the customer. You open a new message and prompt Gemini in the Gmail side panel and tag relevant files by typing @file name. You type:
通话结束后，你想给客户发送一条回顾信息。你打开一条新信息，并在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini，通过输入 `@文件名` 标记相关文件。你输入：

Write a message to [customer] thanking them for their time at our last [meeting]. Provide a quick summary of the meeting and acknowledge any pain points discussed. Ask for additional time to
discuss our [solution] using @[Customer Meeting Gemini Notes]. (Gemini in Gmail)
给 [customer] 写一条信息，感谢他们参加上次的 [meeting]。提供会议的快速总结，并确认讨论的任何痛点。使用 @[Customer Meeting Gemini Notes] 请求额外的时间来讨论我们的 [solution]。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

In preparation for your next meeting, you want to use the transcript and your existing sales materials to generate a customized asset that showcases how your company’s product solves the customer’s pain points mentioned during the call. To do this, you open a new Doc and prompt Gemini in the Docs side panel and tag relevant files by typing @file name. You type:
为了准备下次会议，你想利用转录和你现有的销售材料生成一份定制资产，展示你公司的产品如何解决客户在通话中提到的痛点。为此，你打开一个新的 Doc（Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini，通过输入 `@文件名` 标记相关文件。你输入：

I am an [account manager] and I just finished a call with [customer]. I want to summarize the [pain points] mentioned by [customer] during our last meeting. Provide a list of direct quotes from @[Customer Call
Transcript] where [customer] discusses what they are trying to solve. (Gemini in Docs)
我是 [account manager]，刚结束与 [customer] 的通话。我想总结一下 [customer] 在上次会议中提到的 [pain points]。请提供 @[Customer Call Transcript] 中 [customer] 讨论他们试图解决的问题的直接引语列表。（Gemini in Docs／Google 文档中的 Gemini）

You read through the summary of pain points and see that they capture what was discussed. You click Insert from the side panel. Then, you want to use your existing files to generate custom responses to each of their pain points. You prompt Gemini in the Docs side panel again and tag relevant files. You type:
你通读痛点总结，发现它们捕捉到了讨论的内容。你点击侧边栏中的 Insert（插入）。然后，你想利用现有文件针对每个痛点生成定制的回复。你再次在 Docs（Google 文档）侧边栏中提示 Gemini 并标记相关文件。你输入：

I need to create convincing reasons why [customer] should adopt [product] to solve for [their pain points]. Write specific reasons why [product] from [company] could help them achieve their [business goals] using
@[Product Sales Kit Full Assets]. (Gemini in Docs)
我需要创建令人信服的理由，说明为什么 [customer] 应该采用 [product] 来解决 [their pain points]。使用 @[Product Sales Kit Full Assets] 写出具体的理由，说明 [company] 的 [product] 如何帮助他们实现 [business goals]。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Build customer relationships

## 用例：建立客户关系

You just had a great call with a customer and now you want to use the notes you took from the meeting in Google Docs to draft an email to the customer. In the Google Doc with your notes, you prompt Gemini in Docs by selecting Help me write. You type:
你刚刚与客户进行了一次愉快的通话，现在你想用在 Google Docs（Google 文档）中做的会议笔记起草一封给客户的邮件。在包含笔记的 Google Doc（Google 文档）中，你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Compose a personalized follow-up email to [client] following an initial conversation. Summarize the key
points we discussed and address any outstanding questions. (Gemini in Docs)
在初步交谈后，给 [client] 写一封个性化的后续邮件。总结我们讨论的关键点，并回答任何遗留问题。（Gemini in Docs／Google 文档中的 Gemini）

The account has just adopted one of the company’s service offerings and you need to ensure that they feel supported during the onboarding process. You want to make sure you check in on how things are progressing once a week, but you want to explore what the emails could look like. You open a new Google Doc and prompt Gemini in Docs by selecting Help me write. You type:
该客户刚刚采用了公司的某项服务，你需要确保他们在入职过程中感到受到支持。你想确保每周检查一次进展情况，但你想探索邮件应该是什么样子的。你打开一个新的 Google Doc（Google Docs／Google 文档），并通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Draft four email templates to check in on my customer weekly now that they have purchased our new [service]. Use one value proposition (cost, ease of use, security, availability, and customization) as the
main topic for each email, and include [call to action] in each message. (Gemini in Docs)
既然客户已经购买了我们的新 [service]，起草四封邮件模板每周跟进。每一封邮件以一个价值主张（成本、易用性、安全性、可用性和定制化）为主题，并在每条信息中包含 [call to action]。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Prepare for new customer calls

## 用例：准备新客户通话

You have an upcoming call with a prospect. This is a brand new use case for you, and you need help preparing for the call. You visit Gemini Advanced, and you type:
你即将与一位潜在客户通话。这对你来说是一个全新的用例，你需要帮助准备这次通话。你访问 Gemini Advanced（Gemini 高级版），然后输入：

Draft a customized script for me to follow during my sales call with a prospect. The call will happen over a video call and is set to last 30 minutes. Make sure to add the following in the script: how [company products/solutions] can help address potential customer’s pain points, how [company]’s delivery system guarantees seamless and timely delivery, competitive pricing and volume-discount table, and space for
a customer reference in the [customer’s industry] industry. (Gemini Advanced)
为我起草一份在与潜在客户销售通话时使用的定制脚本。通话将通过视频进行，预计持续 30 分钟。确保在脚本中添加以下内容：[company products/solutions] 如何帮助解决潜在客户的痛点，[company] 的交付系统如何保证无缝和及时交付，具有竞争力的定价和批量折扣表，以及在 [customer’s industry] 行业中的客户推荐空间。（Gemini Advanced／Gemini 高级版）

Now that you’ve done initial research, you export your findings to a new Google Doc. You open the Google Doc and continue working. Now, you want to create a tailored pitch. Using the Google Doc with all of your research notes, you prompt Gemini in Docs by selecting Help me write. You type:
完成初步调查后，你将发现导出到新的 Google Doc。你打开 Google Doc（Google Docs／Google 文档）继续工作。现在，你想创建一个定制的推介。使用包含所有调查笔记的 Google Doc（Google Docs／Google 文档），你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Generate an elevator pitch for [product name] and include key benefits, competitive points of
differentiation, and the pain points that [product name] solves for. (Gemini in Docs)
为 [product name] 生成一个电梯游说（elevator pitch），包括关键利益、竞争差异化点，以及 [product name] 解决的痛点。（Gemini in Docs／Google 文档中的 Gemini）

You have a great start to your elevator pitch and short talking points. You want to use this to further anticipate how the customer call might go. You resume your meeting preparation by returning to Gemini Advanced. You type:
你的电梯游说和简短谈话要点有了一个很好的开端。你想用它来进一步预测客户通话的走向。你回到 Gemini Advanced（Gemini 高级版）继续准备会议。你输入：

I have an upcoming call with a prospect. [Use case] is a new use case for me, and I need help preparing for the call. List the most likely objections [customer] might have for me during a sales call, with suggestions on how to respond to them. I work in [industry], and I am trying to sell [product]. Also provide ideas on how
to handle objections and suggest ways to respond. (Gemini Advanced)
我即将与一位潜在客户通话。[Use case] 对我来说是一个新用例，我需要帮助准备这次通话。列出 [customer] 在销售通话中可能提出的最可能的异议，并提供回应建议。我在 [industry] 工作，正试图销售 [product]。还请提供处理异议的想法和建议的回应方式。（Gemini Advanced／Gemini 高级版）

Business Development Manager Use case: Nurture relationships, personalized outreach, and
thought leadership
业务发展经理用例：培养关系、个性化外展和思想领导力

You’re hoping to build deeper relationships with prospective customers that you recently met. You want to draft a template that you can customize for multiple contacts. You open a new Google Doc, and you prompt Gemini in Docs. You type:
你希望与最近结识的潜在客户建立更深层次的关系。你想起草一个可以针对多个联系人进行定制的模板。你打开一个新的 Google Doc（Google Docs／Google 文档），并在 Docs（Google 文档）中提示 Gemini。你输入：

Draft an outreach email template to industry influencers. Express gratitude that we connected at [event],
and propose collaboration opportunities such as [opportunities]. (Gemini in Docs)
起草一封发给行业影响者的外展邮件模板。对我们在 [event] 上建立联系表示感谢，并提出合作机会，如 [opportunities]。（Gemini in Docs／Google 文档中的 Gemini）

After having a successful call with prospective customers, you want to follow up with thought leadership content from your founder that they may find interesting. You open the Google Doc with the blog post, and prompt Gemini in Docs by selecting Help me write. You type:
在与潜在客户成功通话后，你想跟进你的创始人撰写的思想领导力内容，他们可能会感兴趣。你打开包含博客文章的 Google Doc（Google Docs／Google 文档），并通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Summarize this blog content in bullet points and generate three ideas for follow-up questions I can ask my
customers about their thoughts. (Gemini in Docs)
用要点总结这篇博客内容，并生成三个后续问题的想法，我可以询问客户他们的想法。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Generate personalized customer appreciation materials

## 用例：生成个性化客户答谢材料

You want to personally thank your customers and check in. You open Gmail and prompt Gemini in Gmail by selecting Help me write. You type:
你想亲自感谢客户并进行问候。你打开 Gmail（Gmail 邮箱），并通过选择 Help me write（帮我写）在 Gmail（Gmail 邮箱）中提示 Gemini。你输入：

Generate a personalized email for [customer] on their one-month anniversary working with [company]. Thank them for being a customer. Ask them if they have any questions. Include information about [other
product]. (Gemini in Gmail)
为 [customer] 与 [company] 合作一个月纪念日生成一封个性化邮件。感谢他们成为客户。询问他们是否有任何问题。包含关于 [other product] 的信息。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

You also want to send these customers a gift to thank them. You open a Google Sheet and prompt Gemini in the Sheets side panel. You type:
你还想送这些客户一份礼物以表示感谢。你打开 Google Sheet（Google Sheets／Google 表格），并在 Sheets（Google 表格）侧边栏中提示 Gemini。你输入：

Give me a list of gifts to send new clients that are under $200 and can be shipped to offices.
(Gemini in Sheets)
给我一份送给新客户的礼物清单，价格在 200 美元以下，并且可以运送到办公室。（Gemini in Sheets／Google 表格中的 Gemini）
（Gemini in Sheets／Google 表格中的 Gemini）

Small business owners and entrepreneurs As the owner of a business, getting the most out of your working hours is critical when you’re juggling multiple roles and responsibilities. Understanding your market, delivering for your customers, and staying on top of many competing priorities is critical.
小企业主和企业家 作为企业主，当你身兼数职、责任重大时，充分利用工作时间至关重要。了解市场、服务客户以及处理众多相互竞争的优先事项是关键。

This section introduces you to AI prompts designed to simplify complex choices with AI data analysis, streamline your email inbox, and help you stand out with creative marketing tactics. Discover how Gemini for Google Workspace can help you unlock deep insights, foster collaboration, and help propel your company to new heights.
本节向你介绍旨在通过 AI 数据分析简化复杂选择、理顺电子邮件收件箱，并帮助你通过创意营销策略脱颖而出的 AI 提示词。探索 Gemini for Google Workspace（Google Workspace 版 Gemini）如何帮助你解锁深刻见解、促进协作，并助力你的公司达到新高度。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.
首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.
下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Workspace（Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example
提示词迭代示例 (Prompt iteration example)
Use case: Create pricing comparison
用例：创建价格比较

You are the owner of a local spa. You are evaluating offers you’ve received from two different cleaning companies. You want to find a company with the right price, flexibility, and level of service. You open a new Doc and prompt Gemini in the Docs side panel and tag relevant files by typing @file name in your prompt. You type:
你是一家当地水疗中心的老板。你正在评估收到的两家不同清洁公司的报价。你想找到一家价格合适、灵活且服务水平高的公司。你打开一个新的 Doc（Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini，通过在提示词中输入 `@文件名` 来标记相关文件。你输入：

I’m a business owner and I’m trying to determine the right cleaning vendor using @[Company A Proposal] and @[Company B Proposal]. I need someone to come twice a week, and I want them to vacuum, mop, dust, clean the windows, and wipe down all surfaces. If available, include information about the booking and cancellation policy. Create a comparison table between
the two companies’ proposals. (Gemini in Docs)
我是企业主，我正试图利用 @[Company A Proposal] 和 @[Company B Proposal] 确定合适的清洁供应商。我需要有人一周来两次，我希望他们吸尘、拖地、除尘、擦窗户并擦拭所有表面。如果有的话，请包括有关预订和取消政策的信息。创建两家公司提案的比较表。（Gemini in Docs／Google 文档中的 Gemini）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini in Docs

## Gemini in Docs（Google 文档中的 Gemini）

Gemini returns a formatted table comparing the two proposals. After you make your decision, you go to your email and prompt Gemini in the Gmail side panel. You type:
Gemini 返回一个比较两个提案的格式化表格。做出决定后，你进入邮箱并在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini。你输入：

Write an email to Company A thanking them for their time and their proposal. Ask for a few times
to meet to schedule cleanings. (Gemini in Gmail)
给 A 公司写一封邮件，感谢他们的时间和提案。询问几个会面时间以安排清洁服务。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Gemini in Gmail

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

## Example use cases

## 示例用例

Owner
拥有者
Use case: Enhance personal productivity
用例：提高个人生产力

You have many important email messages to catch up on. You open your email and select an important thread. You open Gemini in the Gmail side panel, and it automatically summarizes the content. (Gemini in Gmail)
你有许多重要的邮件需要处理。你打开邮箱并选择一个重要的会话。你打开 Gmail（Gmail 邮箱）侧边栏中的 Gemini，它会自动总结内容。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Use case: Brainstorm and generate marketing content

## 用例：头脑风暴并生成营销内容

As the business owner, you are also responsible for marketing your services via your social channels, your email-based newsletter, and email marketing. You aren’t sure where to start, so you chat with Gemini Advanced. You type:
作为企业主，你还负责通过社交渠道、基于电子邮件的时事通讯和电子邮件营销来推广你的服务。你不确定从哪里开始，所以你与 Gemini Advanced（Gemini 高级版）聊天。你输入：

I own a [type of business] in [location]. I am working on marketing materials to advertise [event/sale] on [services]. I want to focus on using this sale to bring in repeat customers who haven’t purchased in a while and new customers alike. I want the social posts to feel [inspirational] and [fun]. Suggest some social copy I can use on [social platform] with relevant hashtags, suggested newsletter copy, and two email drafts
(one for existing customers and one for new customers). (Gemini Advanced)
我在 [location] 拥有一家 [type of business]。我正在制作营销材料来宣传 [services] 上的 [event/sale]。我想专注于利用这次促销吸引有一段时间没有购买的回头客以及新客户。我希望社交帖子感觉 [inspirational] 和 [fun]。建议一些我可以在 [social platform] 上使用的社交文案及相关话题标签、建议的时事通讯文案，以及两封邮件草稿（一封给现有客户，一封给新客户）。（Gemini Advanced／Gemini 高级版）

You like the suggestions Gemini provided, so you select Share & export and Export to Docs. You want to continue your brainstorm, so you ask Gemini:
你喜欢 Gemini 提供的建议，所以你选择 Share & export（分享并导出）和 Export to Docs（导出到 Google 文档）。你想继续头脑风暴，所以你问 Gemini：

What are some other effective [event/sale] tactics I can use to bring in new customers? I don’t always want
to offer discounts. Are there other incentives I am overlooking? (Gemini Advanced)

我还能使用哪些其他有效的 [event/sale] 策略来吸引新客户？我不想总是提供折扣。我是否忽略了其他激励措施？（Gemini Advanced／Gemini 高级版）

You continue your conversation with Gemini and are able to create a solid list of marketing tactics to try.

你继续与 Gemini 对话，并能够创建一个可靠的营销策略列表以供尝试。

## Use case: Develop a competitive analysis

## 用例：开展竞争分析

You started a company, and your online business is gaining traction. You have always dreamed of opening a brick-and-mortar store, and now might be the perfect time. You want a thought partner to help you better understand the current landscape. You open Gemini Advanced, and you type:

你创办了一家公司，你的在线业务正获得关注。你一直梦想开一家实体店，现在可能是最佳时机。你需要一个思维伙伴来帮助你更好地了解当前的市场格局。你打开 Gemini Advanced（Gemini 高级版），然后输入：

I am an online business owner. I am considering opening a brick-and-mortar store. Conduct an analysis into the competitive landscape focusing on [focus area]. Provide the strengths and weaknesses of [key competitors] in this area, including their specific strategies, tactics, and results. Identify actionable insights and recommendations for how [my company] can improve its approach and gain a competitive
advantage. (Gemini Advanced)

我是一名在线企业主。我正在考虑开一家实体店。针对竞争格局进行分析，重点关注 [focus area]。提供 [key competitors] 在该领域的优势和劣势，包括他们的具体战略、策略和结果。找出可操作的见解和建议，说明 [my company] 如何改进其方法并获得竞争优势。（Gemini Advanced／Gemini 高级版）

You gathered useful information from your discussion with Gemini Advanced. You want to go deeper in your brainstorming around two competitors in particular. You type:

你从与 Gemini Advanced 的讨论中收集了有用的信息。你想特别针对两个竞争对手进行更深入的头脑风暴。你输入：

Generate a competitive analysis of [company] versus [competitor] within the current market landscape.
(Gemini Advanced)

在当前市场格局下，生成 [company] 与 [competitor] 的竞争分析。（Gemini Advanced／Gemini 高级版）

You select Share & export and Export to Docs.
你选择 Share & export（分享并导出）和 Export to Docs（导出到 Google 文档）。

## Use case: Conduct fundraising and investor relations

## 用例：进行各种筹款和投资者关系活动

You’re ready to reach out to potential investors to make your brick-and-mortar store a reality. You want help getting started on an email to investors, so in the same Google Doc with your competitive analysis research, you prompt Gemini in Docs. You type:

你准备联系潜在投资者，让你实体店的梦想成为现实。你需要帮助开始撰写给投资者的邮件，因此在包含竞争分析调查的同一个 Google Doc（Google Docs／Google 文档）中，你在 Docs（Google 文档）中提示 Gemini。你输入：

Draft a personalized email template to potential investors, highlighting [company’s] unique value proposition and recent progress on [initiatives]. Request a time to meet to discuss opportunities to
collaborate in the next month. (Gemini in Docs)

起草一封发给潜在投资者的个性化邮件模板，重点介绍 [company’s] 独特的价值主张和 [initiatives] 的近期进展。请求在下个月见面讨论合作机会的时间。（Gemini in Docs／Google 文档中的 Gemini）

The email template gives you a starting place. You tweak the draft and continue to add a few personal touches before sending the email to the potential investors. After a successful meeting with them a month later, you want to draft a thank you message. You open your Google Doc with the meeting transcript and notes. You prompt Gemini in Docs to help you write an email draft. You type:

邮件模板为你提供了一个起点。你调整了草稿，并在发送给潜在投资者之前继续添加一些个人风格。一个月后，与他们成功会面，你想起草一封感谢信。你打开包含会议转录和笔记的 Google Doc（Google Docs／Google 文档）。你在 Docs（Google 文档）中提示 Gemini 帮你写一封邮件草稿。你输入：

Draft an email thanking a potential investor for the call and ask for time to schedule a follow-up meeting to
address [questions and concerns]. (Gemini in Docs)

起草一封邮件感谢潜在投资者的通话，并请求安排后续会议的时间以解决 [questions and concerns]。（Gemini in Docs／Google 文档中的 Gemini）

## Use case: Manage time off policies and tracking

## 用例：管理休假政策和跟踪

You have a lengthy handbook detailing all of your company’s policies and procedures. You want to make the time-off request policy easily digestible for new hires. You open the Google Doc with the handbook. You prompt Gemini in Docs by selecting Help me write. You type:

你有一本详细说明公司所有政策和程序的长手册。你想让新员工容易理解休假申请政策。你打开包含手册的 Google Doc（Google Docs／Google 文档）。你通过选择 Help me write（帮我写）在 Docs（Google 文档）中提示 Gemini。你输入：

Generate a step-by-step checklist summarizing the company’s time-off request policy. Ensure it is written
in plain language and easy for employees to understand. (Gemini in Docs)

生成一个总结公司休假申请政策的逐步清单。确保使用通俗易懂的语言，便于员工理解。（Gemini in Docs／Google 文档中的 Gemini）

You need a quick way to track staffing each week because many of your employees are shift-based. You open Gemini in the Sheets side panel. You type:

你需要一种快速跟踪每周人员配备的方法，因为你的许多员工是轮班制的。你打开 Sheets（Google 表格）侧边栏中的 Gemini。你输入：

Create a table that tracks weekly staffing. Create columns for date, name, shift (AM or PM), and notes.
(Gemini in Sheets)

创建一个跟踪每周人员配备的表格。创建日期、姓名、班次（上午或下午）和备注列。（Gemini in Sheets／Google 表格中的 Gemini）
（Gemini in Sheets／Google 表格中的 Gemini）

Startup leaders You thrive in fast-paced, dynamic environments where you can wear many hats and make a tangible impact. You’re driven by a passion for innovation, a desire to learn and grow, and a tolerance for risk. Your work is unique in its variety, its potential for high reward, and its direct connection to the company’s success. You’re not just executing tasks; you’re building something from the ground up, shaping the future of your company, and potentially disrupting entire industries.

创业公司领导者 你在快节奏、充满活力的环境中如鱼得水，身兼数职并产生切实的影响。你被创新的激情、学习和成长的渴望以及对风险的包容所驱动。你的工作在多样性、高回报潜力和与公司成功直接相关方面是独一无二的。你不仅仅是在执行任务；你是从零开始建立某种东西，塑造公司的未来，并可能颠覆整个行业。

Gemini for Google Workspace can help you redefine productivity and foster meaningful connections with investors, customers, and coworkers. This section provides practical prompts and real-world use cases designed specifically for you and your team. Learning to write effective prompts with Gemini for Workspace will help improve your productivity and streamline your everyday tasks, giving you more time to focus on strategic work.

Gemini for Google Workspace（Google Workspace 版 Gemini）可以帮助你重新定义生产力，并与投资者、客户和同事建立有意义的联系。本节提供专门为你和你的团队设计的实用提示词和真实用例。学习用 Gemini for Workspace（Workspace 版 Gemini）编写有效的提示词将有助于提高你的生产力并简化你的日常任务，让你有更多时间专注于战略工作。

## Getting started

## 开始上手

First, review the general prompt-writing tips on page 2 and the Prompting 101 section at the beginning of this guide.

首先，请回顾第 2 页的通用提示词写作技巧，以及本指南开头的 Prompting 101（提示词入门 101）部分。

Each prompt below is presented with an accompanying scenario to serve as inspiration for how you can collaborate with Gemini for Workspace. The prompt iteration example shows how you could write follow-up prompts to build on the initial generated response.

下面每条提示词都配有相应场景，用于启发你如何与 Gemini for Workspace（Workspace 版 Gemini）协作。“提示词迭代示例（Prompt iteration example）”展示了你如何在初次生成结果的基础上，通过追加提示词进行追问与完善。

Prompt iteration example

提示词迭代示例 (Prompt iteration example)

Use case: Brainstorm business and strategy

用例：头脑风暴业务和战略

You just had a productive planning and strategy brainstorming session with colleagues and you took many notes physically on a whiteboard. You snap a quick image with your phone and upload it directly to Gemini Advanced. You type:

你刚刚与同事进行了一次富有成效的规划和战略头脑风暴会议，你在白板上做了许多笔记。你用手机拍了一张快照，并直接上传到 Gemini Advanced（Gemini 高级版）。你输入：

I am a founder at a startup focused on [industry]. I was brainstorming with colleagues about [topic], and
we took notes on this whiteboard. Turn these notes into text. (Gemini Advanced)

我是一家专注于 [industry] 行业的创业公司的创始人。我和同事们就 [topic] 进行了头脑风暴，我们在白板上做了笔记。把这些笔记转换成文本。（Gemini Advanced／Gemini 高级版）

- Persona • Task • Context • Format
- 角色（Persona）• 任务（Task）• 上下文（Context）• 格式（Format）

## Gemini Advanced

## Gemini Advanced（Gemini 高级版）

Now you want to proactively continue brainstorming before you recap all of the ideas and notes for the group in a follow-up email. You continue the conversation and type:

现在你想在后续邮件中向小组回顾所有想法和笔记之前，主动继续头脑风暴。你继续对话并输入：

Suggest follow-up items we could discuss for our [topic of brainstorm session]. What was not covered
that could have been, and what are we potentially missing? (Gemini Advanced)

建议我们可以为 [topic of brainstorm session] 讨论的后续项目。有什么本来应该涉及但没有涉及的内容，我们可能遗漏了什么？（Gemini Advanced／Gemini 高级版）

## Gemini Advanced

## Gemini Advanced（Gemini 高级版）

You save all of your notes by clicking Share & export and Export to Docs. You are ready to send the recap message to the team, so you open your email and prompt Gemini in the Gmail side panel and tag the relevant file of notes by typing @file name. You type:

你通过点击 Share & export（分享并导出）和 Export to Docs（导出到 Google 文档）保存所有笔记。你准备给团队发送回顾信息，所以你打开邮箱并在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini，并通过输入 `@文件名` 标记相关的笔记文件。你输入：

Use @[Brainstorm Notes and Ideas 9/1/24] to write a meeting recap to the team using an upbeat and friendly tone. Share some of the ideas I have for our next meeting to discuss [topic]. (Gemini in Gmail)

使用 @[Brainstorm Notes and Ideas 9/1/24] 给团队写一份会议回顾，使用乐观友好的语气。分享我为下次会议讨论 [topic] 的一些想法。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

## Gemini in Gmail（Gmail 邮箱中的 Gemini）

## Example use cases

## 示例用例

Founder

创始人
Use case: Create an elevator pitch (speech to text)

用例：创建电梯游说（语音转文本）

You’re scheduled to present to a group of prospective investors. This will be your first time discussing your business with this audience. You need to work on your elevator pitch, so you chat with Gemini Advanced using your voice to prompt. You select the microphone icon and say:

你计划向一群潜在投资者进行演示。这将是你第一次与这群听众讨论你的业务。你需要准备电梯游说，所以你用语音与 Gemini Advanced（Gemini 高级版）聊天。你选择麦克风图标并说：

I’m the founder of [startup] in [industry], and I need help creating a short elevator pitch for [company and product description]. I need to make the pitch relevant to [audience] and I want to especially highlight [key features of product] because I want them to [take this action]. Include a compelling hook and anticipate questions an investor might have. Make the tone professional but relaxed and confident.
(Gemini Advanced)

我是 [industry] 行业 [startup] 的创始人，我需要帮助为 [company and product description] 创建一个简短的电梯游说。我需要使游说与 [audience] 相关，我特别想强调 [key features of product]，因为我希望他们 [take this action]。包括一个引人注目的钩子，并预测投资者可能提出的问题。语气要专业，但要轻松自信。（Gemini Advanced／Gemini 高级版）

## Use case: Develop your personal brand

## 用例：发展你的个人品牌

As your company grows, you’re working on increasing your social media presence, so you want to define and hone your personal brand. To brainstorm, you turn to Gemini Advanced. You type:

随着公司的成长，你正致力于增加你在社交媒体上的影响力，所以你想定义和磨练你的个人品牌。为了进行头脑风暴，你求助于 Gemini Advanced（Gemini 高级版）。你输入：

Help me grow my personal brand. I am the founder of [a startup] in [industry]. I am passionate about [topics]. I want to inspire [audience] with business tips and lessons I’ve learned from starting my own company. My goals are to build a following so that I can [generate more media] for the business.
What are some ideas you have for how to accomplish this? (Gemini Advanced)

帮我发展我的个人品牌。我是 [industry] 行业 [a startup] 的创始人。我对 [topics] 充满热情。我想用我从创办自己公司中学到的商业技巧和教训来激励 [audience]。我的目标是建立追随者群体，以便我能为企业 [generate more media]。对于如何实现这一目标，你有什么想法？（Gemini Advanced／Gemini 高级版）

Gemini returns insights into how you can begin to build messaging and content that aligns to your personal brand and that can help you achieve your goals.

Gemini 返回关于如何开始构建符合你个人品牌并能帮助你实现目标的信息和内容的见解。

Head of Operations

运营主管
Use case: Communicate and negotiate with vendors

用例：与供应商沟通和谈判

You’ve received a quote from two different manufacturers to create packaging for the company’s new product. You want to compare and contrast the offers before you negotiate. You open a new Doc and prompt Gemini in the Docs side panel and reference relevant files by typing @file name. You type:

你收到了两家不同制造商为公司新产品制作包装的报价。你想在谈判前比较和对比这些报价。你打开一个新的 Doc（Google 文档），并在 Docs（Google 文档）侧边栏中提示 Gemini，通过输入 `@文件名` 引用相关文件。你输入：

I need to make a vendor decision for packaging manufacturing. Create a table that compares the two proposals I’ve received @[Company A’s Proposal] and @[Company B’s Proposal]. (Gemini in Docs)

我需要对包装制造供应商做出决定。创建一个表格，比较我收到的两个提案 @[Company A’s Proposal] 和 @[Company B’s Proposal]。（Gemini in Docs／Google 文档中的 Gemini）

Gemini creates a table comparing the two different proposals. You make a decision, but now you want to see if you can negotiate with your preferred vendor. You go to your inbox and start a new email draft. You prompt Gemini in the Gmail side panel. You type:

Gemini 创建了一个比较两个不同提案的表格。你做出了决定，但现在你想看看是否可以与你首选的供应商进行谈判。你进入邮箱并开始起草一封新邮件。你在 Gmail（Gmail 邮箱）侧边栏中提示 Gemini。你输入：

Create an email draft to [selected vendor] telling them that I’ve decided to move forward with them as the [packaging] vendor, but I would like to negotiate [a bulk pricing discount]. Use a collaborative tone.
(Gemini in Gmail)

给 [selected vendor] 起草一封邮件，告诉他们我已决定选择他们作为 [packaging] 供应商，但我想商谈 [a bulk pricing discount]。使用合作的语气。（Gemini in Gmail／Gmail 邮箱中的 Gemini）

Gemini in Gmail returns a drafted message that is ready to send. You select Insert and send the email.

Gmail（Gmail 邮箱）中的 Gemini 返回一封准备发送的草拟邮件。你选择 Insert（插入）并发送邮件。

## Use case: Plan and track budgets

## 用例：规划和跟踪预算

You’re in planning mode and you first want to understand where previous years’ budgets were spent. You have all of this data in a Sheet. You decide to chat with Gemini Advanced. You upload the Sheet and prompt Gemini by typing:

你处于规划模式，首先想了解往年的预算都花在哪里了。这一张 Sheet（Google Sheets／Google 表格）里有所有这些数据。你决定与 Gemini Advanced（Gemini 高级版）聊天。你上传 Sheet（Google Sheets／Google 表格）并通过输入提示 Gemini：

Using the attached spreadsheet, identify trends and patterns in our expenses by category over the last three years. Identify areas where costs have increased significantly and investigate potential reasons.
(Gemini Advanced)

使用附带的电子表格，识别过去三年我们按类别支出的趋势和模式。找出成本显著增加的领域，并调查潜在原因。（Gemini Advanced／Gemini 高级版）

Gemini returns a response that helps inform your budget proposal for next year.

Gemini 返回一个回复，有助于为你的明年预算提案提供信息。

Head of Product

产品主管
Use case: Develop a product launch plan

用例：制定产品发布计划

Your team is creating a new product, and you want to conduct research to inform your launch plan in collaboration with the marketing team. Using Gemini, you want to simulate different launch scenarios based on factors like pricing, marketing strategies, and target audience. You go to Gemini Advanced to conduct research and type:

你的团队正在开发一款新产品，你想进行调查，以便与营销团队合作制定发布计划。利用 Gemini，你想根据定价、营销策略和目标受众等因素模拟不同的发布场景。你前往 Gemini Advanced（Gemini 高级版）进行调查并输入：

I am head of product at [startup] in [industry] industry. We are building a product launch plan for [product]. I want to brainstorm a few different scenarios. We are considering offering the [product] at two different price points [A and B] and we are considering launching in [December or January]. Provide pros and cons of each scenario and suggest different ideas we may not have considered. (Gemini Advanced)

我是 [industry] 行业 [startup] 的产品主管。我们正在为 [product] 制定产品发布计划。我想头脑风暴几个不同的场景。我们正在考虑以两个不同的价格点 [A and B] 提供 [product]，并且我们正在考虑在 [December or January] 发布。提供每个场景的优缺点，并建议我们可能没有考虑到的不同想法。（Gemini Advanced／Gemini 高级版）

You want to continue market research brainstorming. You type:

你想继续市场调查头脑风暴。你输入：

How do these prices compare to [competitor products’] prices? Detail what pricing strategies [competitors] use for [products], and list any common tactics they use (such as free trials, discounts, etc.). Summarize how they position the product to [audience]. Cite your sources. (Gemini Advanced)

这些价格与 [competitor products’] 的价格相比如何？详细说明 [competitors] 对 [products] 使用的定价策略，并列出他们使用的任何常见策略（如免费试用、折扣等）。总结他们如何向 [audience] 定位产品。引用你的来源。（Gemini Advanced／Gemini 高级版）

Your research helps you refine your pricing structure and go-to-market strategy for your most important target audience.

你的调查有助于你为最重要的目标受众完善定价结构和进入市场策略。

## Use case: Develop product strategy and roadmap

## 用例：制定产品战略和路线图

You want to refine your product strategy and roadmap. You’ve collected user feedback in a spreadsheet, and you want to clean it up so that it is ready for deeper analysis. You chat with Gemini Advanced and upload a file. You type:

你想完善产品战略和路线图。你在电子表格中收集了用户反馈，你想整理它以便进行更深入的分析。你与 Gemini Advanced（Gemini 高级版）聊天并上传文件。你输入：

Help me clean my [user feedback] survey spreadsheet. Specifically, fill any blank values in the name column with “Anonymous,” then if the [recommend] column shows [Yes], replace that with [Y]. Finally, remove any rows where the satisfaction column is blank. Please generate a new file for me with my
cleaned data. (Gemini Advanced)

帮我整理我的 [user feedback] 调查电子表格。具体来说，将姓名列中的任何空白值填充为“Anonymous”，然后如果 [recommend] 列显示 [Yes]，将其替换为 [Y]。最后，删除满意度列为空的任何行。请为我生成一个包含整理后数据的新文件。（Gemini Advanced／Gemini 高级版）

Gemini returns a clean file for you to conduct deeper analysis on, and from this file, you notice a few trends. You have alignment from the team on features to address recurring user feedback, and now you want to build a high-level roadmap that you can use as a starting point. You continue your conversation with Gemini Advanced. You type:

Gemini 返回一个干净的文件供你进行更深入的分析，从这个文件中，你注意到了一些趋势。团队对解决经常性用户反馈的功能达成了一致，现在你想构建一个高层路线图作为起点。你继续与 Gemini Advanced（Gemini 高级版）对话。你输入：

I am head of product at [startup] in the [industry] industry. We are adding [features] to our [product] to address recurring user feedback, including [feedback trends]. Build a high-level roadmap that will keep us
on track for a Q4 delivery. Put it in a table format. (Gemini Advanced)

我是 [industry] 行业 [startup] 的产品主管。我们正在向我们的 [product] 添加 [features] 以解决经常性的用户反馈，包括 [feedback trends]。构建一个高层路线图，确保我们按计划在第四季度交付。将其设为表格格式。（Gemini Advanced／Gemini 高级版）

Gemini returns a helpful starting point. You want to save the work so you click Export to Docs.

Gemini 返回一个有帮助的起点。你想保存工作，于是点击 Export to Docs（导出到 Google 文档）。

Leveling up your prompt writing This guide is meant to serve as inspiration, and the possibilities are nearly endless with Gemini for Google Workspace. Build on your prompt-writing skills using these tips.

提示词写作进阶 本指南旨在提供灵感，Gemini for Google Workspace（Google Workspace 版 Gemini）的可能性几乎是无限的。利用这些技巧提升你的提示词写作技能。

- Break it up. If you want Gemini for Workspace to perform several related tasks, break them into
  separate prompts.
- 分解任务。如果你希望 Gemini for Workspace（Workspace 版 Gemini）执行几个相关任务，请将它们分解成单独的提示词。

- Give constraints. To generate specific results, include details in your prompt such as character count limits
  or the number of options you’d like to generate.
- 设定约束。要生成特定结果，请在提示词中包含细节，例如字符数限制或你希望生成的选项数量。

- Assign a role. To encourage creativity, assign a role. You can do this by starting your prompt with language
  like: “You are the head of a creative department for a leading advertising agency …”
- 分配角色。为了激发创造力，分配一个角色。你可以通过像这样的语言开始你的提示词：“你是一家领先广告公司的创意部门主管……”

- Ask for feedback. In your conversation with Gemini Advanced, tell it that you’re giving it a project, include all
  the relevant details, and then describe the output you want. Continue the conversation by asking questions
  like, “What questions do you have for me that would help you provide the best output?”
- 寻求反馈。在你与 Gemini Advanced（Gemini 高级版）的对话中，告诉它你正在给它一个项目，包括所有相关细节，然后描述你想要的输出。通过问像这样的问题继续对话：“为了提供最好的输出，你有什么问题要问我吗？”

- Consider tone. Tailor your prompts to suit your intended audience. Ask for outputs to have a specific tone,
  such as formal, informal, technical, creative, or casual.
- 考虑语气。调整你的提示词以适应你的目标受众。要求输出具有特定的语气，例如正式、非正式、技术性、创造性或随意。

- Say it another way. Fine-tune your prompts if the results don’t meet your expectations or if you believe
  there’s room for improvement. An iterative process of review and refinement often yields better results.
- 换种说法。如果结果不符合你的预期，或者你认为还有改进的空间，请微调你的提示词。审查和完善的迭代过程通常会产生更好的结果。

Generative AI and all of its possibilities are exciting, but it’s still new. Even though our models are getting better every day, prompts can sometimes have unpredictable responses.

生成式 AI 及其所有可能性令人兴奋，但它仍然是新鲜事物。尽管我们的模型每天都在变得更好，但提示词有时可能会得到不可预测的回复。

Before putting an output from Gemini for Workspace into action, review it to ensure clarity, relevance, and accuracy. And of course, keep the most important thing in mind: Generative AI is meant to help humans, but the final output is yours.

在将 Gemini for Workspace（Workspace 版 Gemini）的输出付诸行动之前，请对其进行审查，以确保清晰度、相关性和准确性。当然，请牢记最重要的一点：生成式 AI 旨在帮助人类，但最终输出归你所有。

The example prompts in this guide are meant for illustrative purposes.
本指南中的示例提示词仅供说明之用。

Stay up to date workspace.google.com
workspace.google.com/blog

Happy prompting!
祝你提示词写作愉快！
