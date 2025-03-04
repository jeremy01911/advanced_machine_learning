import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score


#ce script sert à trouver le meilleur classifieur et les meilleurs hyperparamètres (meilleur accuracy, precision, recall...) sur le dataset que je lui fourni
#ce dataset peut avoir été créé avec embedding BERT, longform, TF-IDF, et même contenir des features supplémentaires








#on test plusieurs modèles pour évaluer les plus performants

df_balanced_expanded = 'blabla.csv'

y = df_balanced_expanded['label']
X = df_balanced_expanded.loc[:, df_balanced_expanded.columns.str.startswith('feature')]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=35)

X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)
y_train = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=float)

X_train = np.stack(X_train)
X_test = np.stack(X_test)




model_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 10]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 10]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 10]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'depth': [3, 6, 10]
        }
    },
    'BalancedRandomForest': {
        'model': BalancedRandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    }
}
results = []

if __name__ == "__main__":

    for model_name, mp in model_params.items(): #.items() permet d'énumérer sur les clefs et contenu du dictionnaire
        print(f"\nModel: {model_name}")
        
        model = mp['model']
        params = mp['params']
        
        # On applique un GridsearchCV pour chaque model avec sa combinaison associée
        grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring='accuracy', verbose=1) #entraine pour chaque combi en utilisant la validation croisée
        grid_search.fit(X_train, y_train)
        
        # Meilleur modèle et hyperparamètres trouvés
        best_model = grid_search.best_estimator_
        print(f"Best Params: {grid_search.best_params_}")
        
        # Prédiction sur l'ensemble de test
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        # Stocker les résultats
        results.append({
            'Model': model_name,
            'Best Params': grid_search.best_params_,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'AUC': auc
        })

        # Afficher les métriques
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nClassification Report :\n", classification_report(y_test, y_pred))

    # Afficher les résultats sous forme de tableau

    results_df = pd.DataFrame(results)
    results_df.to_csv("./report_resultats.csv")
 


    