import pandas as pd  # noqa: I001
import argparse , time ,re
import numpy as np

##### data #####

#Messy_Employee_dataset.csv

def main():
    ##### timer start #####
    start_ = time.perf_counter()


    ##### argparser #####

    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    # parser.add_argument("--output")
    args = parser.parse_args()

    ##### load csv file #####
    def load_csv_file(file_name):
        try :
            dirty_df = pd.read_csv(f"dirty_data/{file_name}",header=None)
            
        except FileNotFoundError :
            print(f"file not found : dirty_data/{file_name}")
            
        except pd.errors.EmptyDataError :
            print(f"file is empty : dirty_data/{file_name}")
            
        except pd.errors.ParserError :
            print("Wrong delimiter (e.g., commas vs. semicolons),mismatched quotes, or a row has more columns than the header.")
            
        else :
            dirty_df.columns = dirty_df.iloc[0]
            dirty_df = dirty_df.iloc[1:]
            print(f"-number of rows and columns :{dirty_df.shape} \n-first 5 rows :\n{dirty_df.head()} \ncolumns types :\n{dirty_df.dtypes}\n{dirty_df.describe()}")
            
            return dirty_df 
    dirty_df = load_csv_file(args.file)
    

    ##### cleaning clolumn names #####
    def clean_column_names(df) :
        print(f"befor the cleaning {df.columns}")
        clean_columns = []

        for ind , col in  enumerate(df.columns):
            cleaned = col.strip().lower()
            cleaned_v2 = re.sub(r"[ -]", "_",cleaned)
            cleaned_v3 = re.sub(r"[^a-zA-Z0-9_]", "",cleaned_v2)
            cleaned_v4 = re.sub(r"_+" , "_" , cleaned_v3)
            if not cleaned_v4 :
                cleaned_v4 = f"column_{ind}"
            
            clean_columns.append(cleaned_v4)
        seen = {}
        for ind , col in enumerate(clean_columns):
            if col not in seen :
                seen[col]= 0 
            else :
                seen[col]+= 1 
                clean_columns[ind]= col + "_" + f"{seen[col]}"

        df.columns = clean_columns
        
        

        # print(f"after the cleaning {dirty_df.columns}")
        return df
    clean_column_names_df = clean_column_names(df=dirty_df)
    print(f"after the cleaning{clean_column_names_df.columns}")
    print(clean_column_names_df.head())

    def handle_missing_values(df):
        numeric_count = 0
        non_numeric_count = 0
        count_per_column ={}
        
        df = df.replace(["N/A", "NA", "null", "None", "", "-"], np.nan)
        print(f"missing values for each column :\n{df.isna().sum()}")
        
        
        for col in df.columns :
            if pd.api.types.is_numeric_dtype(df[col]) :
                median_ = df[col].median()
                v1 = df[col].isna().sum()
                numeric_count += v1
                count_per_column[col] = v1
                df.fillna( {col:median_} ,inplace = True)
                
            else :
                v2 = df[col].isna().sum()
                non_numeric_count += v2
                count_per_column[col]=v2
                df.fillna({col:"unknown"},inplace = True)
                


            
        total = numeric_count+non_numeric_count
        print(numeric_count , non_numeric_count , count_per_column , f"total : {total}" )
        return df , count_per_column
        
    clean_df, missing_counts = handle_missing_values(df=clean_column_names_df)
    
    print(clean_df.head(5))

    ##### timer stops #####
    end_ = time.perf_counter()
    print(f"it takes {end_ - start_}s")

if __name__ == "__main__" :
    main()




