# Automated-Report-Generation
*COMPANY* : CODTECH IT SOLUTIONS
*NAME* : KOLUSU CHAITANYA VARMA
*INTERN ID* : CT08DF1081
*DOMAIN* : PYTHON PROGRAMMING
*DURATION* : 8 WEEKS
*MENTOR* : NEELA SANTHOSH

  **DESCRPITIPON** :
  In this project, I developed a comprehensive automated data analysis and report generation system using Python. The goal of the task was to build a reusable and modular script that can load a dataset (CSV or JSON format), analyze it, create visualizations, and finally generate a neatly formatted PDF report. This script was completely written and executed using PyCharm.
The script is organized into a class-based structure called ReportGenerator, which contains methods for different parts of the process. It starts by either loading a dataset provided by the user or generating a sample dataset if none is given. The sample data includes realistic fields such as dates, sales amounts, customer counts, product categories, regions, discounts, and satisfaction scores. This allows for a meaningful demonstration of analysis and report creation.
Once the data is loaded, the script performs basic data analysis. It uses the pandas library to calculate summary statistics, detect null values, understand data types, and determine the shape of the dataset. It also identifies numeric and categorical columns separately, which allows for more specific visualizations and summaries.
For visualization, the script uses matplotlib and seaborn to create charts and save them as images. These visualizations include:

A distribution plot of a numeric feature.

A correlation heatmap (if there are multiple numeric columns).

A bar chart showing the top categories in a non-numeric column.

These visual outputs are saved as PNG files and later embedded into the final report.

The PDF report is generated using the reportlab library, which allows for rich text formatting, layout control, and image embedding. The report includes:

A title page with generation time and dataset info.

An executive summary describing the dataset characteristics.

A data quality section that highlights missing values and column-wise null counts.

A table showing detailed summary statistics.

Embedded visualizations for quick data interpretation.

Key insights and practical recommendations based on the analysis.

All of this is packaged into a single PDF file stored in a reports directory. The output is readable, professional-looking, and useful for both technical and non-technical stakeholders.

**OUTPUT** :

[analysis_report.pdf](https://github.com/user-attachments/files/21194818/analysis_report.pdf)

<img width="2970" height="1767" alt="Image" src="https://github.com/user-attachments/assets/466bade5-ed3c-4f16-a845-7275662eb78c" />
