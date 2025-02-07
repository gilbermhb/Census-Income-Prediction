# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:13:32 2024

@author: Gilbert Hernandez
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file):
    data = pd.read_csv(file, header=None)
    column_names = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country','income']
    data.columns = column_names
    
    return data

def plot_graphs(data):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data['age'], bins=25, kde=True, edgecolor='black', palette='pastel', ax=ax)
    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Age Distribution Histogram with Density Curve', fontsize=16)
    ax.legend(['Density Curve', 'Histogram'])
    plt.tight_layout()
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='workclass', hue='income', data=data, palette='pastel')
    plt.title('Income by Workclass')
    plt.show()
    
    
    ###########
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='workclass', hue='income', data=data, palette='pastel')
    plt.title('Income by Workclass')
    plt.show()
    
    ###########
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='education', hue='income', data=data, palette='pastel')
    plt.title('Income by Education')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='education', hue='income', data=data, palette='pastel')
    plt.title('Income by Education')
    plt.show()
    
        
    ###########
    plt.figure(figsize=(8, 6))
    sns.countplot(x='marital_status', hue='income', data=data, palette='pastel')
    plt.title('Income by Marital Status')
    plt.show()
    
    
    ###########
    plt.figure(figsize=(8, 6))
    sns.countplot(x='marital_status', hue='income', data=data, palette='pastel')
    plt.title('Income by Marital Status')
    plt.show()
    
    
    ###########
    plt.figure(figsize = (14,5))
    sns.countplot(x = 'marital_status', data = data, palette='pastel');
    plt.xticks(rotation = 90);
    
    
    ###########
    plt.figure(figsize = (14,5))
    sns.countplot(x = 'race', data = data, palette='pastel');
    plt.xticks(rotation = 90);
    
    
    ###########
    plt.figure(figsize = (14,5))
    sns.countplot(x = 'race', data = data, palette='pastel');
    plt.xticks(rotation = 90);
    
    
    ###########
    plt.figure(figsize=(5, 5))  # Set the figure size
    plt.pie(data['gender'].value_counts(), labels=data['gender'].unique(), autopct='%1.1f%%', startangle=140)
    plt.title('Gender Distributions')  # Set the title of the pie chart
    plt.show()
    
    
    ###########
    sns.countplot(x='gender', hue='income', data=data, palette='pastel')
    plt.title('Income by gender')
    plt.show()
    
    
    ###########
    contingency_table = pd.crosstab(data['gender'], data['income'])
    # Calculate chi-square values
    chi2_values = contingency_table.apply(lambda x: x / x.sum(), axis=1)
    # Plot heatmap
    sns.heatmap(chi2_values, annot=True)
        
    
    

data = load_data('C:/Users/Gilbert Hernandez/OneDrive - Fordham University/CISC 5790 - Data Mining/Project/census-income.data.csv')
plot_graphs(data)