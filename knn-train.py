import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

PIE_PATH = 'PIE_32x32'
YALE_PATH = 'YaleB_32x32'
NumFILES = 5

def load_face_file(path):
    raw_file = np.loadtxt(path)
    X = raw_file[:, :-1]
    y = raw_file[:, -1].astype(int)
    return X, y


def load_all_faces(path):
    train_X = []
    train_y = []

    test_X = []
    test_y = []

    for i in range(NumFILES):
        train_i_X, train_i_y = load_face_file(os.path.join(path, f'StTrainFile{i + 1}.txt'))
        test_i_X, test_i_y = load_face_file(os.path.join(path, f'StTestFile{i + 1}.txt'))

        train_X.append(train_i_X)
        train_y.append(train_i_y)

        test_X.append(test_i_X)
        test_y.append(test_i_y)

    return train_X, train_y, test_X, test_y


def show_face(mtx):
    plt.imshow(np.reshape(mtx, (32, 32)).T, cmap='gray', vmin=0, vmax=1)
    plt.show()

def evaluate(test_x, test_y, model):
  print('Evaluating...')

  pred_y = model.predict(test_x)
  prob_y = model.predict_proba(test_x)

  report = classification_report(test_y, pred_y, output_dict=True)
  report['auc'] = roc_auc_score(test_y, prob_y, multi_class='ovr')

  print("Accuracy:", report['accuracy'])
  print("Precision:", report['macro avg']['precision'])
  print("Recall:", report['macro avg']['recall'])
  print("F1-Score:", report['macro avg']['f1-score'])
  print("AUC:", report['auc'])
  return report


def GridSearch(X, y):
    grid_params = {
        'n_neighbors': [1, 3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    gs = GridSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose = 1,
        cv = 3,
        n_jobs= -1
    )

    gs_results = gs.fit(X, y)
    return gs_results

def train_knn(X, y):
  model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')
  model.fit(X, y)
  return model

if __name__ == '__main__':
    pie_train_X, pie_train_y, pie_test_X, pie_test_y = load_all_faces(PIE_PATH)
    # yale_train_X, yale_train_y, yale_test_X, yale_test_y = load_all_faces(YALE_PATH)

    # show_face(pie_train_X[0][1])
    # show_face(yale_train_X[0][1])

    # model = GridSearch(pie_train_X[0], pie_train_y[0])
    # print("Best estimator:", model.best_estimator_)
    # print(model.score(pie_test_X[0], pie_test_y[0]))


    results = []
    for i in range(NumFILES):
      print("="*50)
      print("For i =", i)
      model = train_knn(pie_train_X[i], pie_train_y[i])
      report = evaluate(pie_test_X[i], pie_test_y[i], model)
      results.append(report)

    print('='*50)
    print('On average')
    print("Accuracy:", report['accuracy'])
    print("Precision:", report['macro avg']['precision'])
    print("Recall:", report['macro avg']['recall'])
    print("F1-Score:", report['macro avg']['f1-score'])
    print("AUC:", report['auc'])
