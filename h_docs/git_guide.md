# Git Guide

## Branch
- Create  
  ```git branch <branch_name>```

- Switch  
  ```git switch <branch_name>```
  > Not recommend ```checkout``` command.  
  > The ```checkout``` command has been split into two commands: ```switch```, ```restore```

- Create and Switch  
  ```git switch -c <branch_name>```

- Delete
  - local  
    ```git branch -d <local_branch>```
  - local --force  
    ```git branch -D <local_branch>```
  - remote  
    ```git push <remote_repo> -d <remote_branch>```

- View
  - local  
    ```git branch```
  - remote branch  
    ```git branch -r```
  - local and remote  
    ```git branch -a```