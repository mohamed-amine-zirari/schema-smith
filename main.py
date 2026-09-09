import pandas as pd 
import argparse , time ,re
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
    # print(f"after the cleaning{clean_column_names_df.columns}")
    # print(clean_column_names_df.head())

    def handle_missing_values(df):
        
        print(f"missing values for each column :\n{df.isna().sum()}")
        
    handle_missing_values(df = clean_column_names_df)

    ##### timer stops #####
    end_ = time.perf_counter()
    print(f"it takes {end_ - start_}s")

if __name__ == "__main__" :
    main()




