import pandas as pd
import sqlalchemy as sa
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Dict
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, gridplot
from bokeh.palettes import Category10

# =============================================================================
# EXCEPTION HANDLING
# =============================================================================

class ProjectError(Exception):
    """
    Base exception class for all errors related to this project.
    
    This class serves as a parent for all custom exceptions. It allows the 
    application to catch any project-specific failure within a single block.
    """
    def __init__(self, message: str = "A project-specific error occurred"):
        self.message = message
        super().__init__(self.message)

class DataMismatchError(ProjectError):
    """
    Exception raised when the X-axis grids of the datasets are incompatible.
    
    This occurs if the training and ideal datasets have different lengths or 
    if the test dataset contains X-values not present in the ideal grid.
    """
    def __init__(self, message: str):
        super().__init__(message)

class NoMatchingFunctionError(ProjectError):
    """
    Exception raised when a test data point does not satisfy the mapping criterion.
    
    This is triggered when the minimum deviation between a test point and 
    the selected ideal functions exceeds the allowed threshold (max_dev * sqrt(2)).
    """
    def __init__(self, x_value: float, min_deviation: float):
        msg = f"Point x={x_value} failed mapping. Min deviation {min_deviation:.4f} exceeds threshold."
        super().__init__(msg)

# =============================================================================
# DATA & DATABASE MANAGEMENT
# =============================================================================

class Dataset:
    """
    Handles the loading and basic metadata management of CSV datasets.
    
    Attributes:
        file_path (str): Absolute or relative path to the CSV file.
        name (str): Stem of the filename (filename without extension).
        df (pd.DataFrame): The dataset loaded into a Pandas DataFrame.
    """
    def __init__(self, file_path: str):
        """
        Loads a CSV file into a Pandas DataFrame.

        Args:
            file_path (str): Path to the target CSV file.
        """
        self.file_path = file_path
        self.name = pathlib.Path(self.file_path).stem
        self.df: pd.DataFrame = pd.read_csv(file_path)

class DatabaseManager:
    """
    Manages connection and data persistence using SQLAlchemy.
    """
    def __init__(self, connection_string: str):
        """
        Initializes the SQLAlchemy engine.

        Args:
            connection_string (str): Database connection string (e.g., 'sqlite:///db.db').
        """
        self.engine = sa.create_engine(connection_string)

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Persists a Pandas DataFrame into a specified SQL table.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            table_name (str): The name of the target table in the database.
        """
        df.to_sql(name=table_name, con=self.engine, if_exists='replace', index=False)

@dataclass
class SelectionResult:
    """
    Data container for storing the result of a function fitting operation.

    Attributes:
        train_column (str): The name of the original training data column.
        ideal_column (str): The name of the best fitting ideal function.
        sse (float): Sum of Squared Errors for the selected function.
        max_deviation (float): The maximum absolute difference observed during training.
    """
    train_column: str
    ideal_column: str
    sse: float
    max_deviation: float

# =============================================================================
# LOGIC CLASSES
# =============================================================================

class IdealFunctionSelector:
    """
    Implements the logic to identify the best fitting ideal functions for training data.
    """
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        """
        Initializes the selector and validates the X-axis alignment.

        Args:
            train_df (pd.DataFrame): DataFrame containing training x-y pairs.
            ideal_df (pd.DataFrame): DataFrame containing the 50 ideal functions.

        Raises:
            DataMismatchError: If the X-columns of train_df and ideal_df are not identical.
        """
        self.train_df = train_df
        self.ideal_df = ideal_df
        
        # STRICT VALIDATION: X-grids must be identical for matrix broadcasting
        if not self.train_df['x'].equals(self.ideal_df['x']):
            raise DataMismatchError("Training and Ideal datasets must have identical X-grids for SSE calculation.")

    def select_best_functions(self) -> Dict[str, SelectionResult]:
        """
        Iterates through training columns and finds the ideal function that minimizes SSE.

        Returns:
            Dict[str, SelectionResult]: A mapping of training columns to their corresponding SelectionResult.
        """
        train_y_cols = [c for c in self.train_df.columns if c != 'x']
        ideal_y_cols = [c for c in self.ideal_df.columns if c != "x"]

        ideal_values = self.ideal_df[ideal_y_cols].to_numpy() 
        results: Dict[str, SelectionResult] = {}

        for train_col in train_y_cols:
            train_values = self.train_df[train_col].to_numpy().reshape(-1, 1)
            
            # Least-Squares Calculation: Sum of Squared Errors (SSE)
            squared_errors = (ideal_values - train_values) ** 2
            sse_per_candidate = squared_errors.sum(axis=0)
            
            best_idx = int(np.argmin(sse_per_candidate))
            best_ideal_col = ideal_y_cols[best_idx]
            best_sse = float(sse_per_candidate[best_idx])
            
            # Calculate maximum deviation for the mapping phase
            max_dev = float(np.abs(ideal_values[:, best_idx] - self.train_df[train_col].to_numpy()).max())

            results[train_col] = SelectionResult(
                train_column=train_col,
                ideal_column=best_ideal_col,
                sse=best_sse,
                max_deviation=max_dev,
            ) 

        return results

class TestDataMapper:
    """
    Implements the logic to map test data points to the selected ideal functions.
    """
    def __init__(self, test_df: pd.DataFrame, idea_df: pd.DataFrame, selected: Dict[str, SelectionResult]):
        """
        Initializes the mapper.

        Args:
            test_df (pd.DataFrame): DataFrame containing test x-y pairs.
            idea_df (pd.DataFrame): DataFrame containing ideal functions.
            selected (Dict[str, SelectionResult]): Results from the IdealFunctionSelector.

        Test points whose x-value is not present in the ideal grid
        cannot be evaluated; they are reported with a warning and
        excluded from the result of map_test_data(), rather than
        aborting the whole run.
        """
        self.test_df = test_df
        self.idea_df = idea_df
        self.selected = selected
        
        # SOFT VALIDATION: a test point whose x isn't on the ideal grid simply
        # can't be evaluated later by pd.merge(); report it, but don't abort
        # the whole run for the sake of a few outlier points.
        off_grid = ~np.isin(self.test_df['x'], self.idea_df['x'])
        if off_grid.any():
            missing_x = self.test_df.loc[off_grid, 'x'].tolist()
            print(f"Warning: {len(missing_x)} test point(s) have x-values not in the ideal grid and will be excluded: {missing_x}")

    def map_test_data(self) -> pd.DataFrame:
        """
        Maps test data points to ideal functions based on the sqrt(2) deviation criterion.

        Returns:
            pd.DataFrame: A DataFrame containing the mapped points, their deviations, 
                          and the assigned ideal function.
        """
        chosen_ideal_cols = []
        tolerances = []

        for func, res in self.selected.items():
            chosen_ideal_cols.append(res.ideal_column)
            tolerances.append(res.max_deviation * np.sqrt(2))

        tolerances = np.array(tolerances)

        # Merge test data with selected ideal functions to align Y values for the same X
        merged_df = pd.merge(
            self.test_df[['x', 'y']],
            self.idea_df[['x'] + chosen_ideal_cols],
            on='x',
            how='inner')

        test_x_col = merged_df['x'].to_numpy()
        test_y_col = merged_df['y'].to_numpy().reshape(-1, 1)
        ideal_values = merged_df[chosen_ideal_cols].to_numpy()

        # Calculate absolute deviations and apply the threshold filter
        deviations = np.abs(test_y_col - ideal_values)
        valid_deviations = np.where(deviations <= tolerances, deviations, np.inf)

        mapped_records = []
        discarded_records = []

        for i in range(len(test_x_col)):
            row_devs = valid_deviations[i]
            min_dev = np.min(row_devs)

            try:
                if np.isinf(min_dev):
                    # min_dev is already np.inf here (post-filter); use the
                    # real, un-filtered deviation so the error message is
                    # actually informative.
                    real_min_dev = float(np.min(deviations[i]))
                    raise NoMatchingFunctionError(test_x_col[i], real_min_dev)
                
                best_idx = np.argmin(row_devs)
                assigned_col = chosen_ideal_cols[best_idx]
                delta_y = float(min_dev)
                
            except NoMatchingFunctionError as e:
                print(f"Excluded: {e.message}")
                assigned_col = None
                delta_y = None

                discarded_records.append({'x': test_x_col[i],
                                          'y': test_y_col[i][0],
                                          })

            mapped_records.append({
                'x': test_x_col[i],
                'y': test_y_col[i][0],
                'delta_y': delta_y,
                'ideal_function': assigned_col
            })

        return pd.DataFrame(mapped_records).dropna(), pd.DataFrame(discarded_records)

# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """
    Builds a Bokeh report of the training, ideal and mapped test data.

    """
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame,
                 selected: Dict[str, SelectionResult], mapped_df: pd.DataFrame, discarded_df: pd.DataFrame):
        """
        Args:
            train_df (pd.DataFrame): Training x-y pairs.
            ideal_df (pd.DataFrame): The 50 ideal functions.
            selected (Dict[str, SelectionResult]): Chosen ideal function per training column.
            mapped_df (pd.DataFrame): Successfully mapped test points: (x, y, delta_y, ideal_function).
        """
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.selected = selected
        self.mapped_df = mapped_df
        self.discarded_df = discarded_df

    def _training_plots(self):
        """
        Builds one scatter+line plot per training column: the raw training data against the ideal function chosen for it.

        Returns:
            list: One Bokeh figure per training column.
        """
        plots = []
        for train_col, res in self.selected.items():
            p = figure(title=f"{train_col} vs {res.ideal_column} (SSE={res.sse:.1f})", width=400, height=400, x_axis_label="x", y_axis_label="y")
            p.scatter(self.train_df['x'], self.train_df[train_col], size=5, color="navy", legend_label="training data")
            p.line(self.ideal_df['x'], self.ideal_df[res.ideal_column], color="orange", line_width=2, legend_label="ideal function")
            p.legend.location = "top_left"
            p.legend.click_policy = 'hide'
            plots.append(p)
        return plots

    def _mapping_plot(self):
        """
        Builds a single plot with the four chosen ideal functions and the
        mapped test points overlaid on top of them.

        Returns:
            The combined Bokeh figure.
        """
        p = figure(title="Test data mapped to the chosen ideal functions",
                   width=800, height=500, x_axis_label="x", y_axis_label="y")
       
        palette = Category10[10]
        
        for i, res in enumerate(self.selected.values()):
            p.line(self.ideal_df['x'], self.ideal_df[res.ideal_column], color=palette[i], line_width=2, legend_label=res.ideal_column)
        if not self.mapped_df.empty:
            p.scatter(self.mapped_df['x'], self.mapped_df['y'], size=6, color="black", legend_label="mapped test points")
        if not self.discarded_df.empty:
            p.scatter(self.discarded_df['x'], self.discarded_df['y'], size=8, color="grey",marker='x',legend_label="discarded test points")

        p.legend.location = "top_left"
        p.legend.click_policy = 'hide'
        return p

    def save_report(self, output_path: str = "visualization.html"):
        """
        Renders every plot into a single self-contained HTML report.

        Args:
            output_path (str): Destination HTML file.
        """
        output_file(output_path, title="DLMDSPWP01 - Ideal Function Assignment")
        layout = column(gridplot(self._training_plots(), ncols=2), self._mapping_plot())
        save(layout)
        print(f"Visualization saved to {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main application pipeline:
    1. Loads training, test, and ideal datasets.
    2. Persists raw data into an SQLite database.
    3. Selects the best fitting ideal functions for the training set.
    4. Maps test data points to these functions using the deviation criterion.
    5. Saves final mapping results to the database.
    """
    script_dir = pathlib.Path(__file__).resolve().parent
    
    try:
        # 1. Load Datasets
        train_set = Dataset(script_dir / "train.csv")
        test_set = Dataset(script_dir / "test.csv")
        ideal_set = Dataset(script_dir / "ideal.csv")  

        # 2. Database Setup and Raw Data Persistence
        db = DatabaseManager("sqlite:///project_database.db")
        db.save_dataframe(train_set.df, "train_data")
        db.save_dataframe(ideal_set.df, "ideal_data")

        # 3. Ideal Function Selection
        selector = IdealFunctionSelector(train_set.df, ideal_set.df)
        best_functions = selector.select_best_functions()

        print("--- Selection Results ---")
        for train_col, res in best_functions.items():
            print(f"Training: {res.train_column} -> Best Ideal: {res.ideal_column} | "
                  f"SSE: {res.sse:.4f} | Max Dev: {res.max_deviation:.4f}")

        # 4. Test Data Mapping
        mapper = TestDataMapper(test_set.df, ideal_set.df, best_functions)
        mapped_records, discarded_records = mapper.map_test_data()
        
        
        # 5. Persistence of Mapping Results
        db.save_dataframe(mapped_records, table_name='test_mapping_results')
        print("\nProcess completed successfully. Results saved to database.")

        # 6. Visualization
        viz = Visualizer(train_set.df, ideal_set.df, best_functions, mapped_records, discarded_records )
        viz.save_report(str(script_dir / "visualization.html"))

    except FileNotFoundError as e:
        print(f"System Error: CSV file not found. {e}")
    except ProjectError as e:
        print(f"Project-specific error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
