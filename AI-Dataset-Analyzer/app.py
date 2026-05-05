from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')

    if not file:
        return "No file uploaded"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    # BASIC INFO
    rows, cols = df.shape
    total_missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    preview = df.head().to_html(index=False)

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    # NUMERIC ANALYSIS
    numeric_analysis = []
    for col in numeric_cols:
        numeric_analysis.append({
            "name": col,
            "mean": round(df[col].mean(), 2),
            "min": df[col].min(),
            "max": df[col].max()
        })

    # CATEGORICAL ANALYSIS
    categorical_analysis = []
    for col in categorical_cols:
        categorical_analysis.append({
            "name": col,
            "unique": df[col].nunique(),
            "top": df[col].mode()[0] if not df[col].mode().empty else "N/A"
        })

    # OUTLIERS
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers[col] = int(df[(df[col] < lower) | (df[col] > upper)].shape[0])

    # -------- GRAPHS --------

    # HISTOGRAM
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10,6))
        df[numeric_cols[0]].hist()
        plt.title(f"Histogram of {numeric_cols[0]}")
        plt.tight_layout()
        plt.savefig("static/hist.png")
        plt.close()

    # BAR
    if len(categorical_cols) > 0:
        plt.figure(figsize=(10,6))
        df[categorical_cols[0]].value_counts().head(10).plot(kind='bar')
        plt.xticks(rotation=45)
        plt.title(f"Top values of {categorical_cols[0]}")
        plt.tight_layout()
        plt.savefig("static/bar.png")
        plt.close()

    # HEATMAP
    corr = df.select_dtypes(include='number').corr()
    if not corr.empty:
        plt.figure(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/heatmap.png")
        plt.close()

    # BOX
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=df[numeric_cols])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/box.png")
        plt.close()

    # KDE
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10,6))
        sns.kdeplot(df[numeric_cols[0]], fill=True)
        plt.tight_layout()
        plt.savefig("static/kde.png")
        plt.close()

    # COUNT
    if len(categorical_cols) > 0:
        plt.figure(figsize=(10,6))
        sns.countplot(y=df[categorical_cols[0]])
        plt.tight_layout()
        plt.savefig("static/count.png")
        plt.close()

    # INSIGHTS
    insights = []

    if total_missing > 0:
        insights.append("Dataset contains missing values")

    if duplicates > 0:
        insights.append("Duplicate rows found")

    for col, val in outliers.items():
        if val > 0:
            insights.append(f"{col} has {val} outliers")

    if len(insights) == 0:
        insights.append("Dataset looks clean")

    return render_template(
        "result.html",
        rows=rows,
        cols=cols,
        total_missing=total_missing,
        duplicates=duplicates,
        preview=preview,
        insights=insights,
        numeric_analysis=numeric_analysis,
        categorical_analysis=categorical_analysis,
        outliers=outliers
    )


if __name__ == "__main__":
    app.run(debug=True)