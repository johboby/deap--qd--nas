# GitHub Pages for DEAP v4.0 - QD-NAS

这个目录包含DEAP项目的GitHub Pages网站文件。

## 文件结构

```
docs/
├── index.html          # 主页面
├── styles.css          # 样式表
├── script.js           # JavaScript交互脚本
├── _config.yml         # Jekyll配置文件
├── .nojekyll           # 禁用Jekyll处理的标记
└── README.md           # 本文件
```

## 内容说明

### index.html
专业的项目展示页面，包含：
- 项目概览和特性介绍
- 性能指标展示
- 快速开始指南
- 算法列表
- 文档链接
- 联系和贡献信息
- FAQ常见问题解答

### styles.css
响应式CSS样式，提供：
- 现代化设计
- 深色导航栏
- 渐变背景
- 移动设备适配
- 平滑过渡效果
- 专业排版

### script.js
JavaScript交互功能：
- 平滑滚动导航
- 数字动画计数器
- 卡片淡入动画
- 代码复制功能
- 移动菜单支持
- 搜索功能

### _config.yml
Jekyll配置文件，定义：
- 站点信息和URL
- 构建设置
- 插件配置
- SEO设置

## 部署说明

GitHub Pages已自动从此目录构建和部署。

### 启用GitHub Pages（如未启用）

1. 在GitHub仓库的Settings中找到"Pages"
2. 选择"Source"为"Deploy from a branch"
3. 选择分支为"main"，目录为"/docs"
4. 保存设置

### 访问网站

部署后，网站将在以下地址可访问：
- https://johboby.github.io/deap--qd--nas

## 修改和更新

要修改网站内容：

1. **更新主页面**：编辑 `index.html`
2. **更新样式**：编辑 `styles.css`
3. **更新交互**：编辑 `script.js`
4. **构建配置**：编辑 `_config.yml`

修改后，提交到GitHub：

```bash
git add docs/
git commit -m "docs: update website content"
git push origin main
```

GitHub Pages将自动重建和部署更新。

## 网站特性

### 响应式设计
- 完全响应式布局
- 适配所有屏幕尺寸
- 移动设备优化

### 性能优化
- 轻量级CSS（无框架依赖）
- 高效JavaScript（无外部库）
- 快速加载时间

### 用户体验
- 平滑导航和滚动
- 动画效果
- 代码示例复制功能
- 移动菜单支持

### SEO优化
- 语义化HTML
- 完整的meta标签
- 清晰的文档结构
- 站点地图支持

## 链接

- **GitHub仓库**：https://github.com/johboby/deap--qd--nas
- **中文文档**：https://github.com/johboby/deap--qd--nas/blob/main/README_CN.md
- **English文档**：https://github.com/johboby/deap--qd--nas/blob/main/README_EN.md
- **文档导航**：https://github.com/johboby/deap--qd--nas/blob/main/DOCS_INDEX.md

## 许可证

本网站内容遵循MIT许可证。详见仓库中的LICENSE文件。

---

**最后更新**：2026-01-14
**版本**：4.0.0
