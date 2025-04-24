import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
from data_cleaning import create_sample_dataset, handle_missing_values, encode_categorical_features, normalize_features, remove_outliers

class DataCleaningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaning and Preprocessing Tool")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.df = None
        self.processed_df = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create buttons
        ttk.Button(self.button_frame, text="Load Dataset", command=self.load_dataset).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Create Sample Dataset", command=self.create_sample).grid(row=0, column=1, padx=5)
        ttk.Button(self.button_frame, text="Process Data", command=self.process_data).grid(row=0, column=2, padx=5)
        ttk.Button(self.button_frame, text="Save Processed Data", command=self.save_data).grid(row=0, column=3, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data View")
        self.notebook.add(self.stats_tab, text="Statistics")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        
        # Create treeview for data display
        self.tree = ttk.Treeview(self.data_tab)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbars
        self.vsb = ttk.Scrollbar(self.data_tab, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self.data_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Create text widget for statistics
        self.stats_text = tk.Text(self.stats_tab, wrap=tk.WORD, width=80, height=30)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create frame for visualizations
        self.viz_frame = ttk.Frame(self.visualization_tab)
        self.viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        self.stats_tab.columnconfigure(0, weight=1)
        self.stats_tab.rowconfigure(0, weight=1)
        self.visualization_tab.columnconfigure(0, weight=1)
        self.visualization_tab.rowconfigure(0, weight=1)
        
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_data_view()
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
    
    def create_sample(self):
        self.df = create_sample_dataset()
        self.update_data_view()
        messagebox.showinfo("Success", "Sample dataset created successfully!")
    
    def process_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load or create a dataset first!")
            return
        
        try:
            # Process the data
            self.processed_df = handle_missing_values(self.df)
            self.processed_df = encode_categorical_features(self.processed_df)
            self.processed_df = normalize_features(self.processed_df)
            self.processed_df = remove_outliers(self.processed_df)
            
            # Update views
            self.update_data_view()
            self.update_statistics()
            self.update_visualizations()
            
            messagebox.showinfo("Success", "Data processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing data: {str(e)}")
    
    def save_data(self):
        if self.processed_df is None:
            messagebox.showwarning("Warning", "Please process the data first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.processed_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving data: {str(e)}")
    
    def update_data_view(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.processed_df is not None:
            df = self.processed_df
        else:
            df = self.df
        
        if df is not None:
            # Configure columns
            self.tree["columns"] = list(df.columns)
            self.tree["show"] = "headings"
            
            for column in df.columns:
                self.tree.heading(column, text=column)
                self.tree.column(column, width=100)
            
            # Add data
            for i, row in df.iterrows():
                self.tree.insert("", "end", values=list(row))
    
    def update_statistics(self):
        self.stats_text.delete(1.0, tk.END)
        if self.processed_df is not None:
            df = self.processed_df
        else:
            df = self.df
        
        if df is not None:
            # Add basic statistics
            self.stats_text.insert(tk.END, "=== Basic Information ===\n\n")
            self.stats_text.insert(tk.END, f"Dataset Shape: {df.shape}\n\n")
            
            self.stats_text.insert(tk.END, "Data Types:\n")
            self.stats_text.insert(tk.END, str(df.dtypes) + "\n\n")
            
            self.stats_text.insert(tk.END, "Missing Values:\n")
            self.stats_text.insert(tk.END, str(df.isnull().sum()) + "\n\n")
            
            self.stats_text.insert(tk.END, "Basic Statistics:\n")
            self.stats_text.insert(tk.END, str(df.describe()))
    
    def update_visualizations(self):
        # Clear existing visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        if self.processed_df is not None:
            df = self.processed_df
        else:
            df = self.df
        
        if df is not None:
            # Create figure for visualizations
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle("Data Visualizations")
            
            # Plot 1: Boxplot of numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                sns.boxplot(data=df[numerical_cols], ax=axes[0, 0])
                axes[0, 0].set_title("Boxplot of Numerical Features")
            
            # Plot 2: Histogram of first numerical column
            if len(numerical_cols) > 0:
                sns.histplot(data=df, x=numerical_cols[0], ax=axes[0, 1])
                axes[0, 1].set_title(f"Histogram of {numerical_cols[0]}")
            
            # Plot 3: Correlation heatmap
            if len(numerical_cols) > 1:
                sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1, 0])
                axes[1, 0].set_title("Correlation Heatmap")
            
            # Plot 4: Scatter plot of first two numerical columns
            if len(numerical_cols) > 1:
                sns.scatterplot(data=df, x=numerical_cols[0], y=numerical_cols[1], ax=axes[1, 1])
                axes[1, 1].set_title(f"Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}")
            
            plt.tight_layout()
            
            # Embed the figure in the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = DataCleaningGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 