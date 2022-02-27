import sys
# sys.path.append("/content/ECS189G_project") #use in colab only
from codes.stage_4_code.Dataset_Loader_IMDB import Dataset_Loader_IMDB
from codes.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from codes.stage_4_code.Method_RNN_IMDB import Method_RNN_IMDB
from codes.stage_4_code.Method_RNN_generation import Method_RNN_generation
from codes.stage_4_code.Result_Saver import Result_Saver
from codes.stage_4_code.Setting_IMDB import Setting_IMDB
from codes.stage_4_code.Setting_Generation import Setting_Generation
from codes.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_4_code.Evaluate_Classification import Evaluate_Classification
import numpy as np
import torch

task = "classification"  #"classification" "generation"
# ---- CNN script ----
if task == "classification":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader_IMDB('IMBD', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    data_obj.dataset_source_file_name = 'IMDB'
    data_obj.load()
    print(data_obj.vocab_size)

    method_obj = Method_RNN_IMDB('RNN', '', data_obj)


    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_IMDB_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_IMDB('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
elif task == "generation":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader_Generation('generation', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    data_obj.words = data_obj.load()
    data_obj.uniq_words = data_obj.get_uniq_words()
    data_obj.vocab_size = len(data_obj.uniq_words)

    method_obj = Method_RNN_generation('RNN', '', data_obj)


    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_generation_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Generation('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')
    # ------------------------------------------------------
