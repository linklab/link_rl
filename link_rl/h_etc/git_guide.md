# Git Guide

## Commit
- Cancel add (unstage)
  ```commandline
  git reset HEAD
  ```
  In detail, --mixed option is the default
  ```commandline
  git reset --mixed HEAD
  ```
  

- Cancel add a specific file
  ```commandline
  git reset HEAD <file_name>
  ```

- Return files to HEAD (Note: All changes are lost)
  ```commandline
  git reset --hard HEAD
  ```

## Branch
- Create  
  ```commandline
  git branch <branch_name>
  ```

- Switch  
  ```commandline
  git switch <branch_name>
  ```
  > Not recommend ```checkout``` command.  
  > The ```checkout``` command has been split into two commands: ```switch```, ```restore``` on git >= 2.34

- Create and Switch  
  ```commandline
  git switch -c <branch_name>
  ```

- Merge
  - merge A into B (then, B is changed but A is not)
    ```commandline
    git switch B
    git merge A
    ```

- Rebase
  - branch A: 1 - 2 - 3 - 4
  - branch B: 1 - 2 - 5 - 6
  - For branch B 1 - 2 - 3 - 4 - 5 - 6
    ```commandline
    git switch B
    git rebase A
    ```


- Delete
  - local  
    ```commandline
    git branch -d <local_branch>
    ```
  - local --force  
    ```commandline
    git branch -D <local_branch>
    ```
  - remote  
    ```commandline
    git push <remote_repo> -d <remote_branch>
    ```

- View
  - local  
    ```commandline
    git branch
    ```
  - remote branch  
    ```commandline
    git branch -r
    ```
  - local and remote  
    ```commandline
    git branch -a
    ```