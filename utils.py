import os
import shutil
from pathlib import Path

def reorganize_cars_by_company(base_path):
    """
    Reorganize car data from model-specific folders to company-specific folders
    """
    
    # Define paths
    train_path = Path(base_path) / "car_data" / "car_data" / "train"
    test_path = Path(base_path) / "car_data" / "car_data" / "test"
    
    # Create new organization folders
    new_train_path = Path(base_path) / "organized_data" / "train"
    new_test_path = Path(base_path) / "organized_data" / "test"
    
    new_train_path.mkdir(parents=True, exist_ok=True)
    new_test_path.mkdir(parents=True, exist_ok=True)
    
    def process_folder(source_path, destination_path):
        """Process train or test folder"""
        
        if not source_path.exists():
            print(f"Warning: {source_path} does not exist")
            return
        
        # Get all model folders
        model_folders = [f for f in source_path.iterdir() if f.is_dir()]
        
        print(f"Processing {len(model_folders)} model folders in {source_path.name}")
        
        for model_folder in model_folders:
            # Extract company name (first word before space)
            folder_name = model_folder.name
            company_name = folder_name.split()[0]
            
            # Create company folder if it doesn't exist
            company_folder = destination_path / company_name
            company_folder.mkdir(exist_ok=True)
            
            # Copy all images from model folder to company folder
            if model_folder.is_dir():
                image_files = list(model_folder.glob("*"))
                
                print(f"  Moving {len(image_files)} files from {folder_name} to {company_name}/")
                
                for image_file in image_files:
                    if image_file.is_file():
                        # Create unique filename to avoid conflicts
                        # Format: original_model_name + original_filename
                        safe_model_name = folder_name.replace(" ", "_").replace("/", "_")
                        new_filename = f"{safe_model_name}_{image_file.name}"
                        
                        destination_file = company_folder / new_filename
                        
                        # Copy the file
                        try:
                            shutil.copy2(image_file, destination_file)
                        except Exception as e:
                            print(f"    Error copying {image_file}: {e}")
    
    # Process both train and test folders
    print("Starting reorganization...")
    print("=" * 50)
    
    process_folder(train_path, new_train_path)
    print()
    process_folder(test_path, new_test_path)
    
    print("=" * 50)
    print("Reorganization complete!")
    
    # Show summary
    print("\nSummary:")
    print(f"Original train path: {train_path}")
    print(f"New train path: {new_train_path}")
    print(f"Original test path: {test_path}")
    print(f"New test path: {new_test_path}")
    
    # Count companies in new structure
    if new_train_path.exists():
        train_companies = [f.name for f in new_train_path.iterdir() if f.is_dir()]
        print(f"\nTrain companies found: {len(train_companies)}")
        print("Companies:", sorted(train_companies)[:10], "..." if len(train_companies) > 10 else "")
    
    if new_test_path.exists():
        test_companies = [f.name for f in new_test_path.iterdir() if f.is_dir()]
        print(f"Test companies found: {len(test_companies)}")

# Run the reorganization
import os
import shutil
import pandas as pd
from pathlib import Path

def reorganize_cars_by_company_from_csv(base_path):
    """
    Reorganize car data using the names.csv file to group by company
    """
    
    # Read the CSV file with car names
    csv_path = Path(base_path) / "names.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    # Read car names from CSV
    car_names = pd.read_csv(csv_path, header=None, names=['car_name'])
    
    # Extract company names (first word before space)
    car_names['company'] = car_names['car_name'].str.split().str[0]
    car_names['class_id'] = range(len(car_names))  # 0-based indexing
    
    print(f"Found {len(car_names)} car classes")
    print("Companies found:", sorted(car_names['company'].unique()))
    print()
    
    # Define paths
    train_path = Path(base_path) / "car_data" / "car_data" / "train"
    test_path = Path(base_path) / "car_data" / "car_data" / "test"
    
    # Create new organization folders
    new_train_path = Path(base_path) / "organized_data" / "train"
    new_test_path = Path(base_path) / "organized_data" / "test"
    
    new_train_path.mkdir(parents=True, exist_ok=True)
    new_test_path.mkdir(parents=True, exist_ok=True)
    
    def process_folder(source_path, destination_path, split_name):
        """Process train or test folder"""
        
        if not source_path.exists():
            print(f"Warning: {source_path} does not exist")
            return
        
        print(f"Processing {split_name} data...")
        
        # Create company folders
        for company in car_names['company'].unique():
            company_folder = destination_path / company
            company_folder.mkdir(exist_ok=True)
        
        # Get all numbered folders (these correspond to class IDs)
        class_folders = [f for f in source_path.iterdir() if f.is_dir() and f.name.isdigit()]
        
        print(f"Found {len(class_folders)} class folders")
        
        for class_folder in class_folders:
            class_id = int(class_folder.name)
            
            # Find corresponding car info from CSV
            if class_id < len(car_names):
                car_info = car_names.iloc[class_id]
                company = car_info['company']
                full_car_name = car_info['car_name']
                
                # Get company folder
                company_folder = destination_path / company
                
                # Get all images in this class folder
                image_files = [f for f in class_folder.iterdir() if f.is_file()]
                
                print(f"  Moving {len(image_files)} images from class {class_id} ({full_car_name}) to {company}/")
                
                # Create safe filename prefix from full car name
                safe_car_name = full_car_name.replace(" ", "_").replace("/", "_").replace(".", "")
                
                for image_file in image_files:
                    # Create new filename: Company_CarModel_ClassID_OriginalName
                    new_filename = f"{safe_car_name}_class{class_id}_{image_file.name}"
                    destination_file = company_folder / new_filename
                    
                    try:
                        shutil.copy2(image_file, destination_file)
                    except Exception as e:
                        print(f"    Error copying {image_file}: {e}")
            else:
                print(f"  Warning: Class ID {class_id} not found in CSV file")
    
    # Process both train and test folders
    print("Starting reorganization based on names.csv...")
    print("=" * 60)
    
    process_folder(train_path, new_train_path, "train")
    print()
    process_folder(test_path, new_test_path, "test")
    
    print("=" * 60)
    print("Reorganization complete!")
    
    # Show summary
    print(f"\nSummary:")
    print(f"Total car classes: {len(car_names)}")
    print(f"Total companies: {len(car_names['company'].unique())}")
    print(f"New train path: {new_train_path}")
    print(f"New test path: {new_test_path}")
    
    # Show company distribution
    company_counts = car_names['company'].value_counts()
    print(f"\nCompany distribution:")
    for company, count in company_counts.head(10).items():
        print(f"  {company}: {count} models")
    
    if len(company_counts) > 10:
        print(f"  ... and {len(company_counts) - 10} more companies")

def show_csv_preview(base_path):
    """
    Show a preview of the CSV file and company mapping
    """
    csv_path = Path(base_path) / "names.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    # Read car names from CSV
    car_names = pd.read_csv(csv_path, header=None, names=['car_name'])
    car_names['company'] = car_names['car_name'].str.split().str[0]
    car_names['class_id'] = range(len(car_names))
    
    print("CSV Preview (first 10 entries):")
    print("-" * 50)
    for i, row in car_names.head(10).iterrows():
        print(f"Class {row['class_id']:3d}: {row['company']:15s} - {row['car_name']}")
    
    print(f"\n... and {len(car_names) - 10} more entries")
    
    print(f"\nCompanies found ({len(car_names['company'].unique())} total):")
    companies = sorted(car_names['company'].unique())
    for i in range(0, len(companies), 5):
        print("  " + ", ".join(companies[i:i+5]))

