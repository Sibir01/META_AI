import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def analyze_and_generate_functions(df):
    functions = []

    if df.isnull().sum().any():
        functions.append(fill_missing_values)

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        functions.append(group_by_column_and_calculate_mean)

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        functions.append(encode_categorical_data)

    functions.append(train_and_predict_model)

    return functions


def fill_missing_values(df):
    print("Filling missing values with column means...")
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def group_by_column_and_calculate_mean(df):
    print("Grouping by a column and calculating mean values...")

    if 'Department' in df.columns:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result = df.groupby('Department')[numeric_cols].mean()
            print(result)
        else:
            print("No numeric columns to calculate mean.")
    else:
        print("'Department' column not found for grouping.")
    return df


def encode_categorical_data(df):
    print("Encoding categorical data...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df


def train_and_predict_model(df):
    print("Training a model on the data...")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for training.")
        return df

    X = df[numeric_cols[:-1]]
    y = df[numeric_cols[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained. MSE: {mse}")
    return df


def main():
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Helen", "Ivan", "Judy"],
        "Age": [25, 30, 35, None, 28, 33, 40, None, 45, 50],
        "Salary": [50000, 60000, None, 80000, 55000, 62000, 70000, 75000, 85000, None],
        "Department": ["HR", "IT", "IT", "HR", "Finance", "Finance", "IT", "HR", "Finance", "IT"]
    }
    df = pd.DataFrame(data)
    print("Environment Data:")
    print(df)

    print("\nAnalyzing environment and generating functions...\n")
    functions = analyze_and_generate_functions(df)

    for func in functions:
        print(f"\nApplying function: {func.__name__}")
        df = func(df)

    print("\nFinal Processed Data:")
    print(df)


if __name__ == "__main__":
    main()
