import pandas as pd 
import argparse , time

def main():
    ##### timer start #####
    start_ = time.perf_counter()


    ##### argparser #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    # parser.add_argument("--output")
    args = parser.parse_args()

    ##### load csv file #####
    #Messy_Employee_dataset.csv
    def load_csv_file(file_name):
        try :
            dirty_df = pd.read_csv(f"dirty_data/{file_name}")
        except FileNotFoundError :
            print(f"file not found : dirty_data/{file_name}")
        except pd.errors.EmptyDataError :
            print(f"file is empty : dirty_data/{file_name}")
        except pd.errors.ParserError :
            print("Wrong delimiter (e.g., commas vs. semicolons),mismatched quotes, or a row has more columns than the header.")
        else :
            print(f"-number of rows and columns :{dirty_df.shape} \n-first 5 rows :\n{dirty_df.head()} \ncolumns types :\n{dirty_df.dtypes}\n{dirty_df.describe()}")
    load_csv_file(args.file)










    ##### timer stops #####
    end_ = time.perf_counter()
    print(f"it takes {end_ - start_}s")

if __name__ == "__main__" :
    main()





