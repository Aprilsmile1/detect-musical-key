import pandas as pd
import numpy as np
import train_validation as ml
from sklearn.metrics import classification_report # 结果评估
from sklearn.model_selection import train_test_split # 拆分数据集
from sklearn.preprocessing import StandardScaler # 数据标准化

feature_a_weights_path = "weights_feature_a/acc_0.7_seed_3_lr_0.05_iteration_4767.xlsx"
feature_a0_weight_path = "weights_feature_b/feature_a0_C_or_Am/acc_0.97_seed_3_lr_0.05_iteration_283.xlsx"
feature_a1_weight_path = "weights_feature_b/feature_a1_Db_or_Bbm/Aacc_0.87_seed_3_lr_0.1_iteration_1547.xlsx"
feature_a2_weight_path = "weights_feature_b/feature_a2_D_or_Bm/acc_0.92_seed_3_lr_0.11_iteration_873.xlsx"
feature_a3_weight_path = "weights_feature_b/feature_a3_Eb_or_Cm/acc_0.93_seed_3_lr_0.1_iteration_36.xlsx"
feature_a4_weight_path = "weights_feature_b/feature_a4_E_or_Dbm/acc_0.96_seed_3_lr_0.1_iteration_1464.xlsx"
feature_a5_weight_path = "weights_feature_b/feature_a5_F_or_Dm/acc_0.94_seed_3_lr_0.1_iteration_193.xlsx"
feature_a6_weight_path = "weights_feature_b/feature_a6_F#_or_Ebm/acc_0.95_seed_3_lr_0.1_iteration_119.xlsx"
feature_a7_weight_path = "weights_feature_b/feature_a7_G_or_Em/acc_0.97_seed_3_lr_0.1_iteration_527.xlsx"
feature_a8_weight_path = "weights_feature_b/feature_a8_Ab_or_Fm/acc_0.96_seed_3_lr_0.1_iteration_277.xlsx"
feature_a9_weight_path = "weights_feature_b/feature_a9_A_or_F#m/acc_0.97_seed_3_lr_0.1_iteration_527.xlsx"
feature_a10_weight_path = "weights_feature_b/feature_a10_Bb_or_Gm/acc_0.89_seed_3_lr_0.1_iteration_1488.xlsx"
feature_a11_weight_path = "weights_feature_b/feature_a11_B_or_Abm/acc_0.95_seed_3_lr_0.1_iteration_102.xlsx"

def load_weight(path):
    weights = {}
    weights['Weight'] = pd.read_excel(path, sheet_name = "Weight").to_numpy() #(12, 12)
    weights['Bias'] = pd.read_excel(path, sheet_name = "Bias").to_numpy().T #(1, 12)
    return weights


def test_predict():
    key_class_names = ['C/Am', 'Db/Bbm', 'D/Bm', 'Eb/Cm', 'E/Dbm', 'F/Dm', 'F#/Ebm', 'G/Em', 'Ab/Fm', 'A/F#m', 'Bb/Gm', 'B/Abm']
    
    
    # Load feature dataset
    df = pd.read_csv("data_feature.csv")
    dataSet = df.iloc[:, 2:14].to_numpy()  

    df_test = pd.read_csv("test_data_feature.csv")
    test_dataSet = df_test.iloc[:, 2:14].to_numpy()   
    test_labels = df_test.iloc[:, 15].to_numpy()       

    scaler = StandardScaler()
    dataSet = scaler.fit_transform(dataSet)
    test_dataSet = scaler.transform(test_dataSet)

    model = ml.MySoftmaxModel(3)

    # Predict feature_a report
    load_weight = model.load_weight(feature_a_weights_path)
    feature_a_predict =  model.predict(test_dataSet, load_weight)
    print(classification_report(test_labels, feature_a_predict, target_names=key_class_names))
    
    # Based on feature_a, to predict mode
    feature_B = []
    for index, feature_A in enumerate(feature_a_predict):
        if feature_A == 0:
            load_weight = model.load_weight(feature_a0_weight_path)
        if feature_A == 1:
            load_weight = model.load_weight(feature_a1_weight_path)
        if feature_A == 2:
            load_weight = model.load_weight(feature_a2_weight_path)
        if feature_A == 3:
            load_weight = model.load_weight(feature_a3_weight_path)
        if feature_A == 4:
            load_weight = model.load_weight(feature_a4_weight_path)
        if feature_A == 5:
            load_weight = model.load_weight(feature_a5_weight_path)
        if feature_A == 6:
            load_weight = model.load_weight(feature_a6_weight_path)
        if feature_A == 7:
            load_weight = model.load_weight(feature_a7_weight_path)
        if feature_A == 8:
            load_weight = model.load_weight(feature_a8_weight_path)
        if feature_A == 9:
            load_weight = model.load_weight(feature_a9_weight_path)
        if feature_A == 10:
            load_weight = model.load_weight(feature_a10_weight_path)
        if feature_A == 11:
            load_weight = model.load_weight(feature_a11_weight_path)

        feature_b = model.predict(test_dataSet, load_weight)
        feature_B.append(feature_b[index])
    
    ground_truth = df_test.iloc[:, 15:17].to_numpy()  
    result = []
    for i, (gt_feature_A, gt_feature_B) in enumerate(ground_truth):
        if (gt_feature_A, gt_feature_B) == (feature_a_predict[i], feature_B[i]):
            result.append(True)
        else:
            result.append(False)
    final_test_accuracy = float(result.count(True) / len(result))
    print("The test accuracy =", round(final_test_accuracy,2))
    print(result)
    return 0

test_predict()

