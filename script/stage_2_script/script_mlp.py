import sys
sys.path.append("/content/drive/MyDrive/UCD courses/ECS189G/ECS189G_project")
from codes.stage_2_code.Dataset_Loader import Dataset_Loader
from codes.stage_2_code.Method_MLP import Method_MLP
from codes.stage_2_code.Result_Saver import Result_Saver
from codes.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from codes.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from codes.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # ---- objection initialization setction ---------------
    train_data_obj = Dataset_Loader('train', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'
    test_data_obj = Dataset_Loader('test', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'
    data_obj = {'train': train_data_obj, 'test': test_data_obj}

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Split('train test split', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    