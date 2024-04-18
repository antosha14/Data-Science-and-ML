CONFIG
ssh-keygen -t rsa -C 'anton.kozel.97@mail.ru' #Generate rsa keys vith email comment
git config --global user.name 'Anton Kozel'
git config --global user.email 'anton.kozel.97@mail.ru'

INFO
git remote -v #To check origin
git status #shows branch and untraked files
git log # to view commit history
git show -s -pretty=raw {hash} # show more info about commit
git ls-tree {tree_ind} #show info about BLOBS the tree is referensing
git show {BLOB ind} #show content of the blob
git tag version1 #adds tag to a commit (used to give more info and as a marker to access commit in differrent comands)
git tag --list
git push --tags

MERGING AND PUSHING
git commit -m 'add thms'
git fetch #sinchronize but without adding new content of files
git merge #add new content of files
git pull = fetch and merge
git remote remove origin

STASH is used when we cant commit but need to save current changes
git stash save 'description' #create stash
git stash list # view stashes
git stash pop {stashname} # returns data from stash to file system and removes it from stash
git stash apply {stashname} # returns data from stash but leaves stash
git stash drop {stashname} # remove stash

ROLLBACK

1. File system: git checkout -- file.txt (.) # Returns files to the state known to git
   git clean -xd# to remove files without versions. x -ignore gitignore, d - also remove directories, f - force (nesessary to clean)
2. Staging area: git reset -- file.txt #Unstage file. --soft - from commit to index, --mixed - from commit to file system, --hard - deletes changes.
3. Bad commit: git commit --amend -m 'commit message' # Used to add something to existing commit
   git reset HEAD^^ (HEAD~2)# deletes commit and reverts to how many "^" commits back
4. Remote: git revert {SHA1 code frome remote repo} #creates mirrored commit, but leaves history
5. HEAD rollback and force push, git reflog

BRANCHING
git checkout -b {branchName} #to create branch
git merge {branchName} # merges current branch with specified branch
git merge --abort #to stop merging prosess
git checkout --Xtheirs # to consider others branches commits as right ones (--Xours for our commits)
git diff # to solve conflicts in git bash

git reset HEAD^ # FOR LOCAL BRANCHES TO GO BACK + git push -force
git reset --soft # to go back to staging area
git commit --amend -m 'commit message' # To modify existing commit

git revert {SHA1 of commit to get rid of} # FOR REMOTE TO GO BACK
REBASE - moving pointer to master branch forward (do it only in local branches bc it alters history)
CHERRY-PICK - used to pull specific commit to the head of the active branch

GUI Standart tools
git gui& #& to launch it in new process GUI
gitk& #tool to view history

THEORY
Everithing on git is build on graphs. Depending on the content sha1 hash is generated
.git folder in project
refs/heads/master #keeps final commit hash
3 main object types in GIT:

1. Commit (files and directories that where changed), they reference trees
2. Tree are used to define names of the files, that we commit and directories that store them
3. BLOB - Binary large object (minimum cell for info storage)

What do the directory names in the .git / objects folder mean? #First 2 letters of sha1 of some blob
How much does branch occupy in the file system? 41 bytes

BOOKS: Scott Chacon Pro Git https://git-scm.com/book/ru/v2

git blame # to see who made changes to a file
git bisect # to allocate specific comment with the isue
git log master..feature # log differences betwin master and feature branches
git submodule
git reflog
