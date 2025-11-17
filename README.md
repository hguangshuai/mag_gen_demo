# Magnetic VAE Generator - Streamlit Demo

一键部署的磁材料结构生成器。

## 📁 文件说明

- `streamlit_app.py` - 主应用文件
- `simple_generator.py` - 生成器（纯 numpy）
- `weights.npz` - 模型权重
- `requirements.txt` - Python 依赖

## 🚀 部署到 Streamlit Cloud

### 方法1: GitHub + Streamlit Cloud（推荐）

1. **创建 GitHub 仓库**
   ```bash
   cd streamlit_deploy
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/你的用户名/仓库名.git
   git push -u origin main
   ```

2. **部署到 Streamlit Cloud**
   - 访问 https://share.streamlit.io
   - 点击 "New app"
   - 连接 GitHub 账户
   - 选择仓库
   - Main file path: `streamlit_app.py`
   - 点击 "Deploy!"
   - 等待 1-2 分钟完成

3. **分享链接**
   - 部署完成后会得到一个 URL
   - 直接分享给别人即可使用！

### 方法2: 直接上传（如果支持）

如果 Streamlit Cloud 支持直接上传，直接上传整个 `streamlit_deploy/` 文件夹。

## ✅ 功能

- ✅ 调整磁矩参数
- ✅ 选择 Ordered/Disordered
- ✅ 生成 CIF 文件
- ✅ 下载结构文件
- ✅ 完全免费，无需配置

## 📝 注意事项

- 首次加载需要几秒钟（加载模型权重）
- 确保所有 4 个文件都在同一目录
- 免费 tier 有使用限制，但足够 demo 使用

