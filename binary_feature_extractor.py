import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import seaborn as sns

class BinaryFeatureExtractor:
    """
    Extracts binary features from strain time-series data for SGNN input.
    
    Features extracted:
    1. High-Strain Exposure: Strain > 0.0569 for > 30% of time
    2. Low-Strain Dominance: Strain < 0.0068 for > 70% of time  
    3. Spike Presence: Strain > 0.00703 for ≥ 5% of time
    4. Middle-Band Suppression: Strain between 0.00101-0.00141 for < 10% of time
    5. Wide Distribution: ≥20% in each band (<0.00102, 0.00102-0.00219, >0.00219)
    """
    
    def __init__(self):
        # Thresholds for binary feature extraction
        self.high_strain_threshold = 0.0569
        self.low_strain_threshold = 0.0068
        self.spike_threshold = 0.00703
        self.middle_band_lower = 0.00101
        self.middle_band_upper = 0.00141
        self.wide_dist_low_max = 0.00102
        self.wide_dist_mid_min = 0.00102
        self.wide_dist_mid_max = 0.00219
        self.wide_dist_high_min = 0.00219
        
        # Time percentage thresholds
        self.high_strain_time_threshold = 0.30  # 30%
        self.low_strain_time_threshold = 0.70   # 70%
        self.spike_time_threshold = 0.05        # 5%
        self.middle_band_time_threshold = 0.10  # 10%
        self.wide_dist_band_threshold = 0.20    # 20%
    
    def extract_features_single_sensor(self, strain_series: np.ndarray) -> np.ndarray:
        """
        Extract 5 binary features from a single sensor's strain time-series.
        
        Args:
            strain_series: 1D numpy array of strain values over time
            
        Returns:
            binary_features: 1D numpy array of 5 binary features [0 or 1]
        """
        total_timesteps = len(strain_series)
        binary_features = np.zeros(5, dtype=int)
        
        # Feature 1: High-Strain Exposure
        high_strain_count = np.sum(strain_series > self.high_strain_threshold)
        high_strain_percentage = high_strain_count / total_timesteps
        binary_features[0] = 1 if high_strain_percentage > self.high_strain_time_threshold else 0
        
        # Feature 2: Low-Strain Dominance
        low_strain_count = np.sum(strain_series < self.low_strain_threshold)
        low_strain_percentage = low_strain_count / total_timesteps
        binary_features[1] = 1 if low_strain_percentage > self.low_strain_time_threshold else 0
        
        # Feature 3: Spike Presence
        spike_count = np.sum(strain_series > self.spike_threshold)
        spike_percentage = spike_count / total_timesteps
        binary_features[2] = 1 if spike_percentage >= self.spike_time_threshold else 0
        
        # Feature 4: Middle-Band Suppression
        middle_band_count = np.sum((strain_series >= self.middle_band_lower) & 
                                 (strain_series <= self.middle_band_upper))
        middle_band_percentage = middle_band_count / total_timesteps
        binary_features[3] = 1 if middle_band_percentage < self.middle_band_time_threshold else 0
        
        # Feature 5: Wide Distribution
        low_band_count = np.sum(strain_series < self.wide_dist_low_max)
        mid_band_count = np.sum((strain_series >= self.wide_dist_mid_min) & 
                               (strain_series <= self.wide_dist_mid_max))
        high_band_count = np.sum(strain_series > self.wide_dist_high_min)
        
        low_band_percentage = low_band_count / total_timesteps
        mid_band_percentage = mid_band_count / total_timesteps
        high_band_percentage = high_band_count / total_timesteps
        
        wide_distribution = (low_band_percentage >= self.wide_dist_band_threshold and
                           mid_band_percentage >= self.wide_dist_band_threshold and
                           high_band_percentage >= self.wide_dist_band_threshold)
        binary_features[4] = 1 if wide_distribution else 0
        
        return binary_features
    
    def extract_features_all_sensors(self, strain_data: np.ndarray) -> np.ndarray:
        """
        Extract binary features from all sensors.
        
        Args:
            strain_data: 2D numpy array of shape (timesteps, num_sensors)
            
        Returns:
            binary_matrix: 2D numpy array of shape (num_sensors, 5)
        """
        num_sensors = strain_data.shape[1]
        binary_matrix = np.zeros((num_sensors, 5), dtype=int)
        
        for sensor_idx in range(num_sensors):
            sensor_strain = strain_data[:, sensor_idx]
            binary_matrix[sensor_idx, :] = self.extract_features_single_sensor(sensor_strain)
        
        return binary_matrix
    
    def load_and_process_excel(self, filepath: str, sheet_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load strain data from Excel file and extract binary features.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Name of sheet to read (if None, reads first sheet)
            
        Returns:
            strain_data: Raw strain data (timesteps, sensors)
            binary_features: Binary feature matrix (sensors, 5)
        """
        try:
            print(f"Loading Excel file: {filepath}")
            
            # Read Excel file
            if sheet_name is not None:
                data = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                # Try to read the first sheet or handle multiple sheets
                excel_file = pd.ExcelFile(filepath)
                print(f"Available sheets: {excel_file.sheet_names}")
                
                # Use the first sheet
                first_sheet = excel_file.sheet_names[0]
                print(f"Reading sheet: {first_sheet}")
                data = pd.read_excel(filepath, sheet_name=first_sheet)
            
            # Ensure we have a DataFrame
            if not isinstance(data, pd.DataFrame):
                print(f"Error: Expected DataFrame, got {type(data)}")
                return None, None
            
            print(f"Raw data shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            
            # Handle different column naming conventions
            # Look for numeric columns that could be sensor data
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter out obvious non-sensor columns
            sensor_columns = []
            for col in numeric_columns:
                col_str = str(col).lower()
                # Skip time, index, or other non-sensor columns
                if not any(skip_word in col_str for skip_word in ['time', 'index', 'step', 'frame']):
                    sensor_columns.append(col)
            
            if len(sensor_columns) == 0:
                print("Warning: No sensor columns found, using all numeric columns")
                sensor_columns = numeric_columns
            
            print(f"Using sensor columns: {sensor_columns}")
            
            # Extract strain data
            strain_data = data[sensor_columns].values
            
            # Remove any NaN values
            if np.any(np.isnan(strain_data)):
                print("Warning: Found NaN values, removing rows with NaN")
                valid_rows = ~np.any(np.isnan(strain_data), axis=1)
                strain_data = strain_data[valid_rows]
            
            print(f"Final strain data shape: {strain_data.shape}")
            print(f"Number of sensors: {strain_data.shape[1]}")
            print(f"Number of timesteps: {strain_data.shape[0]}")
            
            # Check if we have reasonable data
            if strain_data.shape[0] == 0:
                print("Error: No valid data rows found")
                return None, None
            
            if strain_data.shape[1] == 0:
                print("Error: No sensor columns found")
                return None, None
            
            # Display some statistics
            print(f"Data statistics:")
            print(f"  Min value: {np.min(strain_data):.6f}")
            print(f"  Max value: {np.max(strain_data):.6f}")
            print(f"  Mean value: {np.mean(strain_data):.6f}")
            print(f"  Std value: {np.std(strain_data):.6f}")
            
            # Extract binary features
            binary_features = self.extract_features_all_sensors(strain_data)
            
            return strain_data, binary_features
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def visualize_features(self, binary_features: np.ndarray, title: str = "Binary Features"):
        """
        Visualize the binary feature matrix as a heatmap.
        
        Args:
            binary_features: Binary feature matrix (sensors, 5)
            title: Title for the plot
        """
        feature_names = [
            'High-Strain\nExposure',
            'Low-Strain\nDominance', 
            'Spike\nPresence',
            'Middle-Band\nSuppression',
            'Wide\nDistribution'
        ]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(binary_features, 
                   annot=True, 
                   cmap='RdYlBu_r',
                   xticklabels=feature_names,
                   yticklabels=[f'Sensor {i+1}' for i in range(binary_features.shape[0])],
                   cbar_kws={'label': 'Feature Active (1) / Inactive (0)'})
        plt.title(title)
        plt.xlabel('Binary Features')
        plt.ylabel('Sensors')
        plt.tight_layout()
        plt.show()
    
    def compare_intact_vs_damaged(self, intact_features: np.ndarray, 
                                 damaged_features: np.ndarray) -> Dict:
        """
        Compare binary features between intact and damaged states.
        
        Args:
            intact_features: Binary features for intact state (sensors, 5)
            damaged_features: Binary features for damaged state (sensors, 5)
            
        Returns:
            comparison_stats: Dictionary with comparison statistics
        """
        stats = {}
        
        # Feature activation rates
        intact_activation = np.mean(intact_features, axis=0)
        damaged_activation = np.mean(damaged_features, axis=0)
        
        feature_names = [
            'High-Strain Exposure',
            'Low-Strain Dominance', 
            'Spike Presence',
            'Middle-Band Suppression',
            'Wide Distribution'
        ]
        
        stats['intact_activation'] = dict(zip(feature_names, intact_activation))
        stats['damaged_activation'] = dict(zip(feature_names, damaged_activation))
        stats['activation_difference'] = dict(zip(feature_names, damaged_activation - intact_activation))
        
        # Sensor-wise differences
        sensor_differences = np.sum(np.abs(damaged_features - intact_features), axis=1)
        stats['sensor_change_count'] = sensor_differences
        stats['most_affected_sensor'] = np.argmax(sensor_differences)
        
        return stats
    
    def print_comparison_report(self, stats: Dict):
        """Print a formatted comparison report."""
        print("\n" + "="*60)
        print("BINARY FEATURE COMPARISON: INTACT vs DAMAGED")
        print("="*60)
        
        print("\nFeature Activation Rates:")
        print("-" * 40)
        for feature in stats['intact_activation'].keys():
            intact = stats['intact_activation'][feature]
            damaged = stats['damaged_activation'][feature]
            diff = stats['activation_difference'][feature]
            print(f"{feature:20s}: Intact={intact:.3f}, Damaged={damaged:.3f}, Diff={diff:+.3f}")
        
        print(f"\nSensor-wise Changes:")
        print("-" * 20)
        for i, changes in enumerate(stats['sensor_change_count']):
            print(f"Sensor {i+1:2d}: {changes} features changed")
        
        print(f"\nMost affected sensor: Sensor {stats['most_affected_sensor'] + 1}")

if __name__ == "__main__":
    # Example usage
    extractor = BinaryFeatureExtractor()
    
    # Load and process intact data
    print("Processing intact sensor data...")
    intact_strain, intact_features = extractor.load_and_process_excel('AllSensors-Intact.xlsx')
    
    if intact_features is not None:
        print("\nIntact Binary Features:")
        print(intact_features)
        extractor.visualize_features(intact_features, "Intact State - Binary Features")
    
    # Load and process damaged data
    print("\nProcessing damaged sensor data...")
    damaged_strain, damaged_features = extractor.load_and_process_excel('AllSensors-Damaged.xlsx')
    
    if damaged_features is not None:
        print("\nDamaged Binary Features:")
        print(damaged_features)
        extractor.visualize_features(damaged_features, "Damaged State - Binary Features")
    
    # Compare intact vs damaged
    if intact_features is not None and damaged_features is not None:
        stats = extractor.compare_intact_vs_damaged(intact_features, damaged_features)
        extractor.print_comparison_report(stats) 