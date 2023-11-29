import numpy as np
import pandas as pd
from sklearn.metrics import classification_report # 结果评估
from sklearn.model_selection import train_test_split # 拆分数据集
from sklearn.preprocessing import StandardScaler # 数据标准化

key_to_num= {"C" : 0, "Db" : 1, "D" : 2, "Eb" : 3, "E" : 4, "F" : 5, "F#" : 6, "G" : 7, "Ab" : 8,  "A" : 9 , "Bb" : 10, "B" : 11}


class MySoftmaxModel():
    def __init__(self, random_seed = None):
        self.random_seed = random_seed   
        self.weight = None                            #initial random weight

    def label_transform(self, train_label, classes):  #Transform scalor label to vector label
        vector_train_label = []
        for y in train_label:
            vector_y = list(np.zeros(classes))        #class = 12
            vector_y[y] = 1
            vector_train_label.append(vector_y)
        vector_train_label = np.array(vector_train_label)
        return vector_train_label
            
    def initialize_weight(self, classes, feature_nums, random_seed):   #classes = 12, feature_nums = 12
        initial = {}
        np.random.seed(random_seed)
        initial['Weight'] = np.random.randn(classes, feature_nums)     #shape of (12, 12)
        initial['Bias'] = np.random.randn(classes)                     #shape of (12,)
        return initial
    
    def linear_combination(self, train_data_matrix, weight):
        ret = np.dot(train_data_matrix, weight['Weight'].T) + np.tile(weight['Bias'], (train_data_matrix.shape[0], 1))
        return ret
    
    def softmax_activation(self, linear_ret):
        exp_linear_ret = np.exp(linear_ret)
        softmax_matrix = np.zeros_like(exp_linear_ret)
        denominator = np.sum(exp_linear_ret, axis = 1)
        lenth = exp_linear_ret.shape[0]
        for i in range(lenth):
            softmax_matrix[i]= exp_linear_ret[i]/denominator[i]
        return softmax_matrix

    def gradient_iteration(self, weight, bias, train_data_matrix, train_label, softmax_matrix, learning_rate):
        loss = softmax_matrix - train_label
        weight = weight - (learning_rate / train_data_matrix.shape[0]) * np.dot(loss.T, train_data_matrix)
        bias = bias -(learning_rate / train_data_matrix.shape[0]) * np.sum(loss, axis = 0)
        return weight, bias
    
    def train(self, train_data_matrix, train_label, test_data, test_label, class_names, classes, learning_rate, iteration_num):
        self.classes = classes
        vector_train_label = self.label_transform(train_label, self.classes)
        train_nums, feature_nums = train_data_matrix.shape
        if self.weight is None:
            self.weight = self.initialize_weight(self.classes, feature_nums, self.random_seed)

        # print("initial random weight: ")
        # for k in range(self.classes):
        #     print(f"w{k}:{self.weight['Weight'][k]},b{k}:{self.weight['Bias'][k]}")

        # set accuracy threshold of saved model
        save_accuracy_weight = 0

        print("Train starts")
        for i in range(iteration_num):
            Z = self.linear_combination(train_data_matrix, self.weight)
            A = self.softmax_activation(Z)
            self.weight['Weight'], self.weight['Bias'] = self.gradient_iteration(self.weight['Weight'], self.weight['Bias'], train_data_matrix, vector_train_label, A, learning_rate)

            # if i >= 1500:
            #     learning_rate = 0.06
            # decide if saving the new weight
            y_predict = self.predict(test_data)
            reportdic = classification_report(test_label, y_predict, target_names = class_names, output_dict = True)
            if reportdic['weighted avg']['precision'] > save_accuracy_weight:
                save_accuracy_weight = reportdic['weighted avg']['precision']
                self.save_weight_to_excel(learning_rate, i, save_accuracy_weight)
        print("Train ends")


    def get_weight(self):
        return self.weight
    
    def check_size(self):
        print(self.weight['Bias'].shape)
        print(self.weight['Weight'].shape)

    def save_weight_to_excel(self,lr, count, accuracy):

        # different model has different path, need to be modified when saving the mode_weights
        file_path = "weights_feature_a/acc_" + str(round(accuracy, 2)) + "_seed_" + str(self.random_seed) + "_lr_" + str(lr) + "_iteration_" + str(count) + ".xlsx"
        #file_path = "weights_feature_b/feature_a11_B_or_Abm/acc_" + str(round(accuracy, 2)) + "_seed_" + str(self.random_seed) + "_lr_" + str(lr) + "_iteration_" + str(count) + ".xlsx"
       
        file = pd.ExcelWriter(file_path)
        weight = pd.DataFrame(self.weight['Weight'])
        bias = pd.DataFrame(self.weight['Bias'])
        weight.to_excel(file, sheet_name = 'Weight', index = False)
        bias.to_excel(file, sheet_name = 'Bias',index = False)
        file.close()
        return 0
    
    def load_weight(self, weight_path):
        weight = {}
        weight['Weight'] = pd.read_excel(weight_path, sheet_name = "Weight").to_numpy()
        weight['Bias'] = pd.read_excel(weight_path, sheet_name = "Bias").to_numpy().T
        return weight

    
    def predict(self, validation_data_matrix, weight = None):
        if not weight:
            weight = self.weight
        Z = self.linear_combination(validation_data_matrix, weight)
        A = self.softmax_activation(Z)
        y_predict = np.argmax(A, axis=1)
        return y_predict
    



def musical_feature_A(train = False, predict = False, load_weight = None):
    
    key_class_names = ['C/Am', 'Db/Bbm', 'D/Bm', 'Eb/Cm', 'E/Dbm', 'F/Dm', 'F#/Ebm', 'G/Em', 'Ab/Fm', 'A/F#m', 'Bb/Gm', 'B/Abm']
    
    # Load feature dataset
    df = pd.read_csv("data_feature.csv")
    dataSet = df.iloc[:, 2:14].to_numpy()   #shape of (2610, 12)
    labels = df.iloc[:, 15].to_numpy()       #shape of (2610,)

    # create random_seed
    random_seed = 3

    # Divide dataset into trainset and testset
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, train_size=0.7, random_state = random_seed)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create model
    model = MySoftmaxModel(random_seed = random_seed)

    # Train model
    if train:
        #model.weight = model.load_weight("weights_feature_a/acc_0.7_seed_3_lr_0.05_iteration_4767.xlsx")
        model.train(train_data, train_label, test_data, test_label, key_class_names, classes = 12, learning_rate = 0.1, iteration_num = 2)

    # Predict
    if predict:
        load_weight = model.load_weight(load_weight)
        y_predict =  model.predict(test_data, load_weight)
        print(classification_report(test_label, y_predict, target_names=key_class_names))

    return 0

#musical_feature_A(train=True)
#musical_feature_A(predict=True, load_weight="weights_feature_a/acc_0.7_seed_3_lr_0.05_iteration_4767.xlsx")


def musical_feature_B(train = False, predict = False, load_weight = None):
    
    key_class_names = ['minor', 'major']
    
    # Load feature dataset
    df = pd.read_csv("data_feature.csv")
    df = df[df['feature_a'] == 11]
    dataSet = df.iloc[:, 2:14].to_numpy()   #shape of (2610, 12)
    labels = df.iloc[:, -1].to_numpy()       #shape of (2610,)

    # create random_seed
    random_seed = 3

    # Divide dataset into trainset and testset
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, train_size=0.7, random_state = random_seed)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create model
    model = MySoftmaxModel(random_seed = random_seed)

    # Train model
    if train:
        model.train(train_data, train_label, test_data, test_label, key_class_names, classes = 2, learning_rate = 0.1, iteration_num = 3000)

    # Predict
    if predict:
        load_weight = model.load_weight(load_weight)
        y_predict =  model.predict(test_data, load_weight)
        print(classification_report(test_label, y_predict, target_names=key_class_names))

    return 0


#musical_feature_B(train=True)
musical_feature_B(predict=True, load_weight="weights_feature_b/feature_a11_B_or_Abm/acc_0.95_seed_3_lr_0.1_iteration_102.xlsx")



