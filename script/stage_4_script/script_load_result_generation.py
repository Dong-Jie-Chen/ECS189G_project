import sys
# sys.path.append("/content/ECS189G_project") #use in colab only
from codes.stage_4_code.Model_Loader import Model_Loader
from codes.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
import torch
import numpy as np


def init_state(model, sequence_length=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (torch.zeros(model.num_layers, sequence_length, model.hidden_size).to(device),
            torch.zeros(model.num_layers, sequence_length, model.hidden_size).to(device))

def test(model, text, next_words=20):
    # do the testing, and result the result
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    words = text.split(' ')
    state_h, state_c = init_state(model, len(words))
    for i in range(0, next_words):
        x = torch.tensor([[model.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model.forward(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.argmax(p)
        next_word = model.index_to_word[word_index]
        words.append(next_word)
        if next_word == '<EOS>':
            break
    return " ".join(words)

if __name__ == "__main__":
    result_obj = Model_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_generation_'
    result_obj.result_destination_file_name = 'generation_model'
    result_obj.load()
    print('Result:', result_obj.data)

    val = None
    while val != "exit":
        val = input("Enter your starting 3 words: ")
        if len(val.split(' ')) != 3:
            print("input 3 words, type \"exit\" to quit")
        else:
            result = test(result_obj.data, val)
            print(result)


    print('************ Overall Performance ************')
    print('************ Finish ************')


