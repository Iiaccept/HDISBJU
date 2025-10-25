import numpy as np


def get_metrics(real_score, predict_score):
    # 对预测分数去重并排序
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)

    # 生成阈值
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds).T  # 变成列向量
    thresholds_num = thresholds.shape[0]  # 阈值数量

    # 创建预测分数矩阵
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))

    # 获取预测值大于或小于阈值的位置
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    # 设置符合阈值条件的预测值
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # 计算TP、FP、FN、TN
    TP = predict_score_matrix.dot(real_score.T)  # 真阳性
    FP = predict_score_matrix.sum(axis=1) - TP  # 假阳性
    FN = real_score.sum() - TP  # 假阴性
    TN = len(real_score.T) - TP - FP - FN  # 真阴性

    # 计算FPR和TPR
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    # ROC 曲线计算
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    # PR 曲线计算
    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    # 计算F1分数、准确率、特异性等
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    # 获取最大F1分数对应的指标
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision