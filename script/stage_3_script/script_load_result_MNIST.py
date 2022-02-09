import sys
#sys.path.append("/content/ECS189G_project") #use in colab only
from codes.stage_3_code.Result_Loader import Result_Loader
from codes.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_3_code.Evaluate_Classification import Evaluate_Classification
if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
    result_obj.result_destination_file_name = 'prediction_result'

    result_obj.load()
    print('Result:', result_obj.data)

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    evaluate_obj.data = result_obj.data
    mean_score = evaluate_obj.evaluate()

    evaluate_obj_1 = Evaluate_Classification('classification', '')
    evaluate_obj_1.data = result_obj.data
    c_report = evaluate_obj_1.evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score))
    print('CNN Classification Report: \n' + c_report)
    print('************ Finish ************')