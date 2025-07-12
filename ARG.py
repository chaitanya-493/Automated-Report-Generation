#!/usr/bin/env python3
"""
Automated Report Generation Script
Reads data from CSV/JSON files, analyzes it, and generates formatted PDF reports
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os
import numpy as np
import io
from PIL import Image as PILImage


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        """Initialize the report generator with output directory."""
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))

    def load_csv_data(self, file_path):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            print(f"Successfully loaded CSV data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def load_json_data(self, file_path):
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"Successfully loaded JSON data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None

    def analyze_data(self, data):
        """Perform basic data analysis."""
        if data is None or data.empty:
            return None

        analysis = {
            'summary_stats': data.describe(),
            'null_values': data.isnull().sum(),
            'data_types': data.dtypes,
            'shape': data.shape,
            'columns': list(data.columns)
        }

        # Additional analysis for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['correlations'] = data[numeric_cols].corr()
            analysis['numeric_columns'] = list(numeric_cols)

        return analysis

    def create_visualizations(self, data, analysis):
        """Create visualizations and save them as images."""
        if data is None or data.empty:
            return []

        image_files = []

        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')

        # 1. Distribution plot for first numeric column
        numeric_cols = analysis.get('numeric_columns', [])
        if len(numeric_cols) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(data[numeric_cols[0]], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {numeric_cols[0]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
            dist_file = os.path.join(self.output_dir, 'distribution_plot.png')
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(dist_file)

        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = analysis['correlations']
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            corr_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(corr_file)

        # 3. Bar chart for categorical data (if exists)
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            value_counts = data[col].value_counts().head(10)

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            plt.title(f'Top 10 Values in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            bar_file = os.path.join(self.output_dir, 'categorical_bar_chart.png')
            plt.savefig(bar_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(bar_file)

        return image_files

    def generate_pdf_report(self, data, analysis, image_files, output_filename):
        """Generate a comprehensive PDF report."""
        doc = SimpleDocTemplate(
            os.path.join(self.output_dir, output_filename),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        story = []

        # Title
        title = Paragraph("Data Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Report metadata
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Data Shape:</b> {analysis['shape'][0]} rows, {analysis['shape'][1]} columns<br/>
        <b>Data Columns:</b> {', '.join(analysis['columns'])}
        """
        story.append(Paragraph(report_info, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))

        numeric_cols = analysis.get('numeric_columns', [])
        summary_text = f"""
        This report presents a comprehensive analysis of the provided dataset containing {analysis['shape'][0]} 
        records and {analysis['shape'][1]} features. The dataset includes {len(numeric_cols)} numeric columns 
        and {analysis['shape'][1] - len(numeric_cols)} non-numeric columns. 
        """

        if len(numeric_cols) > 0:
            avg_values = analysis['summary_stats'].loc['mean', numeric_cols]
            summary_text += f"Key metrics show average values ranging from {avg_values.min():.2f} to {avg_values.max():.2f} across numeric features."

        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        # Data Quality Assessment
        story.append(Paragraph("Data Quality Assessment", self.styles['CustomHeading']))

        null_counts = analysis['null_values']
        total_nulls = null_counts.sum()
        null_percentage = (total_nulls / (analysis['shape'][0] * analysis['shape'][1])) * 100

        quality_text = f"""
        Data quality analysis reveals {total_nulls} missing values across all columns, 
        representing {null_percentage:.2f}% of the total dataset. 
        """

        if total_nulls > 0:
            cols_with_nulls = null_counts[null_counts > 0]
            quality_text += f"Columns with missing values: {', '.join(cols_with_nulls.index.tolist())}."
        else:
            quality_text += "The dataset is complete with no missing values."

        story.append(Paragraph(quality_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        # Summary Statistics Table
        if len(numeric_cols) > 0:
            story.append(Paragraph("Summary Statistics", self.styles['CustomHeading']))

            # Create summary statistics table
            summary_stats = analysis['summary_stats']
            table_data = [['Statistic'] + list(summary_stats.columns)]

            for index in summary_stats.index:
                row = [index] + [f"{val:.2f}" if pd.notna(val) else "N/A"
                                 for val in summary_stats.loc[index]]
                table_data.append(row)

            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))

        # Add visualizations
        story.append(Paragraph("Data Visualizations", self.styles['CustomHeading']))

        for img_file in image_files:
            if os.path.exists(img_file):
                # Resize image to fit page
                img = Image(img_file, width=6 * inch, height=4 * inch)
                story.append(img)
                story.append(Spacer(1, 12))

        # Insights and Recommendations
        story.append(Paragraph("Key Insights and Recommendations", self.styles['CustomHeading']))

        insights_text = """
        Based on the comprehensive analysis of the dataset, several key insights emerge:

        1. <b>Data Completeness:</b> The dataset quality assessment provides a foundation for understanding 
           data reliability and areas requiring attention.

        2. <b>Statistical Patterns:</b> The summary statistics reveal the central tendencies and 
           distributions of key metrics within the dataset.

        3. <b>Visual Trends:</b> The accompanying visualizations highlight important patterns and 
           relationships that warrant further investigation.

        <b>Recommendations:</b>
        • Implement data validation procedures to maintain data quality
        • Consider additional feature engineering based on observed patterns
        • Establish monitoring systems for key metrics identified in this analysis
        • Plan for regular data quality assessments to ensure ongoing reliability
        """

        story.append(Paragraph(insights_text, self.styles['CustomBody']))

        # Build PDF
        doc.build(story)
        print(f"Report generated successfully: {os.path.join(self.output_dir, output_filename)}")

    def generate_sample_data(self):
        """Generate sample data for demonstration."""
        np.random.seed(42)

        # Generate sample sales data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_records = len(dates)

        sample_data = pd.DataFrame({
            'date': dates,
            'sales_amount': np.random.normal(1000, 200, n_records),
            'customer_count': np.random.poisson(50, n_records),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_records),
            'discount_percentage': np.random.uniform(0, 30, n_records),
            'satisfaction_score': np.random.normal(4.2, 0.8, n_records)
        })

        # Introduce some realistic patterns
        sample_data['sales_amount'] = sample_data['sales_amount'] * (1 + sample_data['discount_percentage'] / 100)
        sample_data['satisfaction_score'] = np.clip(sample_data['satisfaction_score'], 1, 5)

        # Save sample data
        sample_file = os.path.join(self.output_dir, 'sample_data.csv')
        sample_data.to_csv(sample_file, index=False)

        return sample_file

    def run_full_analysis(self, data_file=None, output_filename='analysis_report.pdf'):
        """Run the complete analysis pipeline."""
        # Generate sample data if no file provided
        if data_file is None:
            print("No data file provided. Generating sample data...")
            data_file = self.generate_sample_data()

        # Load data
        if data_file.endswith('.csv'):
            data = self.load_csv_data(data_file)
        elif data_file.endswith('.json'):
            data = self.load_json_data(data_file)
        else:
            print("Unsupported file format. Please provide CSV or JSON file.")
            return

        if data is None:
            print("Failed to load data. Exiting.")
            return

        # Analyze data
        print("Analyzing data...")
        analysis = self.analyze_data(data)

        # Create visualizations
        print("Creating visualizations...")
        image_files = self.create_visualizations(data, analysis)

        # Generate PDF report
        print("Generating PDF report...")
        self.generate_pdf_report(data, analysis, image_files, output_filename)

        print(f"Analysis complete! Check the '{self.output_dir}' directory for outputs.")


def main():
    """Main function to run the report generator."""
    # Initialize report generator
    generator = ReportGenerator()

    # Run analysis with sample data
    generator.run_full_analysis()

    # Example of running with custom data file
    # generator.run_full_analysis('your_data.csv', 'custom_report.pdf')


if __name__ == "__main__":
    main()
