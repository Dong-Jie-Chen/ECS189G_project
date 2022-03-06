import sys
# sys.path.append("/content/ECS189G_project") #use in colab only
from codes.stage_5_code.Dataset_Loader import Dataset_Loader
from codes.stage_5_code.Method_GNN import Method_GNN
from codes.stage_5_code.Result_Saver import Result_Saver
from codes.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from codes.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_3_code.Evaluate_Classification import Evaluate_Classification
import numpy as np
import torch

dataset = "cora"  #"cora" "citeseer" "pubmed"
# ---- CNN script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader(dName=dataset, dDescription='')
    data_obj.dataset_source_folder_path = f'../../data/stage_5_data/{dataset}/'
    data_obj.dataset_source_file_name = dataset

    method_obj = Method_GNN('GNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_5_result/GNN_{dataset}_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('GNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------

