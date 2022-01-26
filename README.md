# ECS189G_project
run on colab instructions
### Step 1: clone the repository
```shell script
!git clone https://github.com/Dong-Jie-Chen/ECS189G_project
```
### Step 2: Using git lfs to load the data

```shell script
!sudo apt-get install git-lfs
%cd ECS189G_project
!git lfs pull
```

### Step 3: Training (skip this if you only want to see the results)
uncomment the second line of script_mlp.py
```shell script
%cd /content/ECS189G_project/script/stage_2_script/
!python script_mlp.py
```

### Step 4: Testing
uncomment the second line of script_load_result.py
```shell script
!python script_load_result.py
```