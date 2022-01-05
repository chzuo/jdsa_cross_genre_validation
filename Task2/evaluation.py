import json,math
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import json,math
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
def scorer(filename):

    with open(filename,'r',encoding='utf-8') as rf:
        rf.readline()
        results = [float(line.split('\t')[1][:-1]) for line in rf.readlines()]
        results_v2 = [round(s*2)/2  for s in results]
    gold_dict = json.load(open('evaluate_dict.json','r',encoding='utf-8'))
    gold_score =[float(s) for s in gold_dict['all_score']]

    print('PCC score: ',pearsonr(results,gold_score))
    print('MSE score: ', mean_squared_error(results,gold_score))
    
    count = 0
    results_file = []
    for file in gold_dict['file_score']:
        score_file = results[count:count+len(file)]
        results_file.append(score_file)
        count+=len(file)
    
    
        y_true = []

    for i,file in zip(range(len(gold_dict['file_score'])),gold_dict['file_score']):
        max_num = max(file)
        if max_num<3:
            y_true.append('U')
        elif max_num>4:
            y_true.append('S')
        else:
            y_true.append('N')

    
    y_pred = []
    count = 0
    results_file = []
    for file in gold_dict['file_score']:
        score_file = results[count:count+len(file)]
        results_file.append(score_file)
        count+=len(file)
    max_list = []
    for i,file in zip(range(len(gold_dict['file_score'])),results_file):
        max_num = max(file)
        max_list.append(file)
        max_num = math.ceil(max_num*2)/2
        if max_num<3:
            y_pred.append('U')
        elif max_num>4:
            y_pred.append('S')
        else:
            y_pred.append('N')
    
    print(metrics.classification_report(y_true, y_pred, digits=3))
    



def main():
	scorer('example_output.txt')
	
if __name__ == '__main__':
    main()
