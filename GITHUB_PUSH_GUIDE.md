# 🚀 Push DeepGuard-X to GitHub - Complete Guide

Follow these steps to upload your project to GitHub:

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in details:
   - **Repository name**: `deepguard-x` (or your choice)
   - **Description**: "Production-grade multi-modal deepfake detection system"
   - **Public** or **Private**: Choose your preference
   - **Add README**: NO (we already have one)
   - **Add .gitignore**: NO (we already have one)
   - **License**: MIT (or choose MIT from our LICENSE file)

3. Click "Create repository"

## Step 2: Configure Git Locally

Open PowerShell in your project directory and run:

```powershell
# Navigate to project
cd D:\deepfake_detect

# Configure Git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Production-grade multi-modal deepfake detection system"

# Add remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/deepguard-x.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Using GitHub CLI (Easier Alternative)

If you have GitHub CLI installed:

```powershell
# Login to GitHub
gh auth login

# Create repository and push in one command
gh repo create deepguard-x --source=. --remote=origin --push --public
```

To install GitHub CLI:
```powershell
choco install gh  # if you use Chocolatey
# OR
winget install --id GitHub.cli  # if you use Windows Package Manager
```

## Step 4: After Pushing

### Add GitHub topics (for discoverability):
1. Go to your repository settings
2. Add topics: `deepfake-detection`, `computer-vision`, `audio-processing`, `pytorch`, `deep-learning`

### Setup branch protection (optional):
1. Settings → Branches → Add protection rule
2. Require pull request reviews before merging
3. Require status checks to pass

### Enable GitHub Pages for documentation (optional):
1. Settings → Pages
2. Source: Deploy from branch
3. Branch: main, /root
4. Your README becomes live at: https://username.github.io/deepguard-x/

## 📋 Quick Command Reference

```powershell
# Check status
git status

# Add specific files
git add src/models/video_models.py

# Commit changes
git commit -m "Add new feature"

# Push changes
git push origin main

# Pull latest changes
git pull origin main

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Delete branch
git branch -d feature/new-feature
```

## 🔐 Authentication

### Option 1: HTTPS with Personal Access Token (Recommended)

```powershell
# When prompted for password, use GitHub Personal Access Token
# Generate at: https://github.com/settings/tokens
# - Select: repo (full control)
# - Select: workflow (for CI/CD if needed)
```

### Option 2: SSH (Most Secure)

```powershell
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add SSH key to SSH agent
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519

# Add public key to GitHub: https://github.com/settings/keys

# Update remote to use SSH
git remote set-url origin git@github.com:YOUR_USERNAME/deepguard-x.git
```

## 📦 Additional: .gitignore Details

Your project already has `.gitignore`, but it excludes:
- `data/` - Large datasets (good - don't commit raw data)
- `logs/` - Training logs
- `*.pth` - Model weights
- `.env` - Environment variables
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache
- `venv/` - Virtual environment

If you want to add data samples, create a separate branch:
```powershell
git checkout -b data/sample-data
# Commit only sample data
git push origin data/sample-data
```

## 🎯 Final Checklist

- ✅ README.md has installation instructions
- ✅ LICENSE file included (MIT)
- ✅ .gitignore configured
- ✅ requirements.txt up to date
- ✅ setup.py configured
- ✅ No sensitive data (API keys, passwords)
- ✅ No large data files
- ✅ Code is well-commented
- ✅ All tests pass

## 🚀 Example: Complete Push Sequence

```powershell
cd D:\deepfake_detect

# First time setup
git init
git add .
git commit -m "Initial commit: DeepGuard-X production system"
git remote add origin https://github.com/YOUR_USERNAME/deepguard-x.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update: Improved UI and added new features"
git push origin main
```

## 💡 Tips

1. **Commit often** with meaningful messages
2. **Don't commit data files** - use git-lfs for large files if needed
3. **Use branches** for new features: `git checkout -b feature/new-feature`
4. **Write good commit messages** - future you will thank you
5. **Review before pushing** - `git diff` to see changes

## ❓ Troubleshooting

**"Permission denied (publickey)"**
- Use HTTPS instead of SSH, or generate SSH key correctly

**"fatal: not a git repository"**
- Run `git init` in your project directory first

**"Your branch is behind 'origin/main'"**
- Run `git pull origin main` to sync

**Large file errors**
- Remove with: `git rm --cached filename`
- Then commit and push

---

Ready? Replace `YOUR_USERNAME` and run the commands! 🎉
