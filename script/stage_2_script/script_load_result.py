from codes.stage_2_code.Result_Loader import Result_Loader
from codes.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    result_obj.load()
    print('Result:', result_obj.data)

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    evaluate_obj.data = result_obj.data
    mean_score = evaluate_obj.evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score))
    print('************ Finish ************')