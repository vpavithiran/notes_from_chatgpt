# 📘 Git Commands – Organized Notes
=================================

* * *

1\. **Setup & Configuration**
-----------------------------

```bash
git --version                          # Check Git version
git config --global user.name "pavi"   # Set username
git config --global user.email "email@example.com"  # Set email
git config --global --list             # View all configs
```

* * *

2\. **Repository Initialization**
---------------------------------

```bash
git init                 # Initialize new repo
git init -b main         # Initialize repo with 'main' as default branch
git status               # Show current repo status
```

* * *

3\. **Staging & Committing**
----------------------------

```bash
git add <file-name>      # Stage a specific file
git add .                # Stage all changes
git commit -m "message"  # Commit staged files with message
git commit -a -m "msg"   # Add & commit tracked files (skip staging)
git commit -a            # Commit changes with add (prompts editor)
```

* * *

4\. **Checking Changes**
------------------------

```bash
git diff                 # Show unstaged changes
git diff --staged        # Show staged changes
git log                  # Show commit history
git log --pretty=oneline # Compact commit history
git log --graph          # Visualize branch/merge history
```

* * *

5\. **Removing Files**
----------------------

```bash
git rm --cached <file>   # Remove file from staging but keep locally
```

* * *

6\. **Clone & Remote Setup**
----------------------------

```bash
git clone <repo-link>                          # Clone repository
git branch -M main                             # Rename branch to main
git remote add origin <repo-link>              # Link remote repo
git remote -v                                  # Show remote connections
```

### 🔑 HTTPS vs SSH

*   **HTTPS** → Asks for username/password.
*   **SSH** → Uses key authentication, no password prompt.

#### SSH Setup

```bash
ssh-keygen -o   # Generates SSH key pair in ~/.ssh
# Copy contents of id_rsa.pub into GitHub → Settings → SSH Keys
```

* * *

7\. **Pushing & Pulling**
-------------------------

```bash
git push -u origin main    # Push to remote (set upstream)
git push origin main       # Push changes
git pull origin main       # Pull updates from remote
```

* * *

8\. **Tags**
------------

```bash
git tag                                 # List all tags
git tag -a v1.0 -m "Release version 1"  # Create annotated tag
git show v1.0                           # Show tag details
git push origin v1.0                    # Push specific tag
```

* * *

9\. **Branching**
-----------------

```bash
git branch                  # List local branches
git branch --all            # List local + remote branches
git checkout -b featureX    # Create + switch to new branch
git switch -c featureX      # Alternative: create + switch
git switch main             # Switch branch
git checkout main           # Alternative: switch branch
git switch -                # Switch to previous branch
git branch -d featureX      # Delete branch
```

* * *

10\. **Merging**
----------------

```bash
git merge featureX   # Merge branch 'featureX' into current branch
```

* * *

11\. **VS Code Git Status Symbols**
-----------------------------------

*   **U** → Untracked
*   **A** → Added (staged)
*   **M** → Modified

* * *

