from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

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

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    # BASIC INFO
    rows, cols = df.shape
    total_missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    preview = df.head().to_html(index=False)

    # COLUMN TYPES
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
        top = df[col].mode()[0] if not df[col].mode().empty else "N/A"
        categorical_analysis.append({
            "name": col,
            "unique": df[col].nunique(),
            "top": top
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

    # -------- VISUALIZATIONS --------
    plots = []

    # Histogram
    if len(numeric_cols) > 0:
        plt.figure(figsize=(6,4))
        df[numeric_cols[0]].hist()
        plt.title("Histogram")
        plt.savefig("static/hist.png")
        plt.close()
        plots.append("hist.png")

    # KDE
    if len(numeric_cols) > 0:
        plt.figure(figsize=(6,4))
        sns.kdeplot(df[numeric_cols[0]], fill=True)
        plt.title("KDE Plot")
        plt.savefig("static/kde.png")
        plt.close()
        plots.append("kde.png")

    # Boxplot
    if len(numeric_cols) > 0:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[numeric_cols[0]])
        plt.title("Box Plot")
        plt.savefig("static/box.png")
        plt.close()
        plots.append("box.png")

    # Bar
    if len(categorical_cols) > 0:
        plt.figure(figsize=(6,4))
        df[categorical_cols[0]].value_counts().head(10).plot(kind='bar')
        plt.title("Bar Chart")
        plt.savefig("static/bar.png")
        plt.close()
        plots.append("bar.png")

    # Countplot
    if len(categorical_cols) > 0:
        plt.figure(figsize=(6,4))
        sns.countplot(x=df[categorical_cols[0]])
        plt.xticks(rotation=45)
        plt.title("Count Plot")
        plt.savefig("static/count.png")
        plt.close()
        plots.append("count.png")

    # Scatter
    if len(numeric_cols) > 1:
        plt.figure(figsize=(6,4))
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title("Scatter Plot")
        plt.savefig("static/scatter.png")
        plt.close()
        plots.append("scatter.png")

    # Heatmap
    corr = df.select_dtypes(include='number').corr()
    if not corr.empty:
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Heatmap")
        plt.savefig("static/heatmap.png")
        plt.close()
        plots.append("heatmap.png")

    # INSIGHTS
    insights = []
    if total_missing > 0:
        insights.append("Dataset has missing values")
    if duplicates > 0:
        insights.append("Dataset has duplicate rows")
    for col, val in outliers.items():
        if val > 0:
            insights.append(f"{col} has {val} outliers")

    if not insights:
        insights.append("Dataset looks clean")

    return render_template("result.html",
                           rows=rows, cols=cols,
                           total_missing=total_missing,
                           duplicates=duplicates,
                           preview=preview,
                           numeric_analysis=numeric_analysis,
                           categorical_analysis=categorical_analysis,
                           outliers=outliers,
                           insights=insights,
                           plots=plots)


# PDF DOWNLOAD
@app.route('/download')
def download():
    pdf_path = "uploads/report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("AI Data Analysis Report", styles['Title']))

    for img in os.listdir("static"):
        if img.endswith(".png"):
            elements.append(Image(f"static/{img}", width=400, height=250))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)