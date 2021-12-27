# Git Guide

## Commit
- Add cancel (unstage)
  ```commandline
  git reset HEAD
  ```
  
- Add cancel a specific file
  ```commandline
  git reset HEAD <file_name>
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