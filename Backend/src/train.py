from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=12,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "../model/wine_model.pkl")
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test target values
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training Random Forest model...")
    model = fit_model(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)