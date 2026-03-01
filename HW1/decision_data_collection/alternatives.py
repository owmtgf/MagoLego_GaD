from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


MODELS = {

    # ================= TREE MODELS =================
    "DecisionTree_default":
        DecisionTreeClassifier(random_state=42),

    "DecisionTree_shallow":
        DecisionTreeClassifier(max_depth=5, random_state=42),

    "DecisionTree_pruned":
        DecisionTreeClassifier(min_samples_leaf=20, random_state=42),


    "RandomForest_small":
        RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,
            random_state=42
        ),

    "RandomForest_large":
        RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        ),

    "ExtraTrees_fast":
        ExtraTreesClassifier(
            n_estimators=50,
            n_jobs=-1,
            random_state=42
        ),

    "ExtraTrees_large":
        ExtraTreesClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        ),

    # ================= BOOSTING =================
    "GradientBoosting_fast":
        GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.2,
            random_state=42
        ),

    "GradientBoosting_slow":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),

    "AdaBoost_light":
        AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),

    "AdaBoost_heavy":
        AdaBoostClassifier(
            n_estimators=300,
            random_state=42
        ),

    # ================= LINEAR =================
    "LogReg_strong_reg":
        LogisticRegression(
            C=0.1,
            max_iter=2000,
            n_jobs=-1,
            random_state=42
        ),

    "LogReg_weak_reg":
        LogisticRegression(
            C=10,
            max_iter=2000,
            n_jobs=-1,
            random_state=42
        ),

    "Ridge_alpha_small":
        RidgeClassifier(alpha=0.1),

    "Ridge_alpha_large":
        RidgeClassifier(alpha=10),

    # ================= PROBABILISTIC =================
    "GaussianNB_default":
        GaussianNB(),

    "GaussianNB_smoothed":
        GaussianNB(var_smoothing=1e-7),

    "LDA_svd":
        LinearDiscriminantAnalysis(solver="svd"),

    "LDA_lsqr":
        LinearDiscriminantAnalysis(solver="lsqr"),

    # ================= INSTANCE =================
    "KNN_3":
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1),

    "KNN_15":
        KNeighborsClassifier(n_neighbors=15, n_jobs=-1),

    # ================= KERNEL =================
    "SVC_rbf":
        SVC(kernel="rbf", probability=True, random_state=42),

    "SVC_linear":
        SVC(kernel="linear", probability=True, random_state=42),

    "SVC_poly":
        SVC(kernel="poly", degree=3, probability=True, random_state=42),
}


INTERPRETABILITY = {
    DecisionTreeClassifier: 5,
    LogisticRegression: 5,
    GaussianNB: 4,
    LinearDiscriminantAnalysis: 4,
    RidgeClassifier: 4,
    RandomForestClassifier: 3,
    ExtraTreesClassifier: 3,
    GradientBoostingClassifier: 2,
    AdaBoostClassifier: 2,
    SVC: 2,
    KNeighborsClassifier: 1,
}


def get_interpretability(model):
    return INTERPRETABILITY.get(type(model), 3)
