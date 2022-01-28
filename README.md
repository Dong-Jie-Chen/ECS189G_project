# ECS189G_project
## if using local machine:
### Step 1: clone the repository
```shell script
git clone https://github.com/Dong-Jie-Chen/ECS189G_project
```
### Step 2: Load the data
put train.csv and test.csv into data/stage_2_data

### Step 3: Training (skip this if you only want to see the results)

```shell script
python script/stage_2_script/script_mlp.py
```
### Step 4: Testing
```shell script
python script/stage_2_script/script_load_result.py
```

## if using colab:
### Step 1: clone the repository
```shell script
!git clone https://github.com/Dong-Jie-Chen/ECS189G_project
```

### Step 2: Load the data
put train.csv and test.csv into data/stage_2_data


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
