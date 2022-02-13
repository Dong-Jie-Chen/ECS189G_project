# ECS189G_project
## if using local machine:
### Step 1: clone the repository
```shell script
git clone https://github.com/Dong-Jie-Chen/ECS189G_project
```
### Step 2: Load the data
put data into data/stage_3_data

### Step 3: Training (skip this if you only want to see the results)

change method name inside script_cnn.py before running.
```shell script
python script/stage_3_script/script_cnn.py
```
### Step 4: Testing
```shell script
python script/stage_3_script/script_load_result_CIFAR.py
python script/stage_3_script/script_load_result_MNIST.py
python script/stage_3_script/script_load_result_ORL.py
```

## if using colab:
### Step 1: clone the repository
```shell script
!git clone https://github.com/Dong-Jie-Chen/ECS189G_project
```

### Step 2: Load the data
put data into data/stage_3_data


### Step 3: Training (skip this if you only want to see the results)
uncomment the second line of script_cnn.py
change method name inside script_cnn.py before running.
```shell script
%cd /content/ECS189G_project/script/stage_3_script/
!python script_cnn.py
```

### Step 4: Testing
uncomment the second line of script_load_result.py
```shell script
!python script_load_result_CIFAR.py
!python script_load_result_MNIST.py
!python script_load_result_ORL.py
```
