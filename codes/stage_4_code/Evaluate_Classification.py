from codes.base_class.evaluate import evaluate
from sklearn.metrics import classification_report


class Evaluate_Classification(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return classification_report(self.data['true_y'], self.data['pred_y'])