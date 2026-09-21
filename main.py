import pandas as pd  # noqa: I001
import argparse , time ,re
import numpy as np
from sklearn.ensemble import IsolationForest
from canonical_column_names import CANONICAL_FIELDS
from sentence_transformers import SentenceTransformer

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
    dirty_df  = load_csv_file(args.file)
    

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
    # print(clean_column_names_df.dtypes)


    def normalisation(df):
        df = df.replace(["N/A", "NA", "null", "None", "", "-","n/a", "#", "?", "--"], np.nan)
        print(f"missing values for each column :\n{df.isna().sum()}")
        return df
    normalised_df = normalisation(df=clean_column_names_df)

    def Convert_Types(df):

        for col in df.columns:

            print(f"\n{col} : {df[col].dtype}")

            # Remember values that were already missing
            missing_before = df[col].isna()

            # ==================================================
            # 1. Try BOOLEAN
            # ==================================================

            boolean_values = {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
            }

            converted_boolean = (
                df[col]
                .astype(str)
                .str.lower()
                .map(boolean_values)
            )

            boolean_failed = (
                converted_boolean.isna() & ~missing_before
            )

            number_failed = boolean_failed.sum()

            number_tested = (~missing_before).sum()

            if number_tested > 0:
                success_rate = (
                    number_tested - number_failed
                ) / number_tested
            else:
                success_rate = 0

            if success_rate >= 0.95:
                df[col] = converted_boolean
                print("→ detected as boolean")
                continue

            # ==================================================
            # 2. Try NUMERIC
            # ==================================================

            converted_numeric = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            missing_after = converted_numeric.isna()

            conversion_failed = missing_after & ~missing_before

            number_failed = conversion_failed.sum()

            if number_tested > 0:
                success_rate = (
                    number_tested - number_failed
                ) / number_tested
            else:
                success_rate = 0

            if success_rate >= 0.95:

                # Check if all numeric values are whole numbers
                if (converted_numeric.dropna() % 1 == 0).all():
                    df[col] = converted_numeric.astype("Int64")
                    print("→ detected as integer")
                else:
                    df[col] = converted_numeric.astype("Float64")
                    print("→ detected as float")

                continue

            # ==================================================
            # 3. Try DATETIME (only if values look date-like)
            # ==================================================

            sample = df[col].dropna().astype(str).head(20)
            looks_like_date = sample.str.contains(r"[-/:]").mean() >= 0.5

            if looks_like_date:

                converted_datetime = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                missing_after = converted_datetime.isna()

                conversion_failed = missing_after & ~missing_before

                number_failed = conversion_failed.sum()

                if number_tested > 0:
                    success_rate = (
                        number_tested - number_failed
                    ) / number_tested
                else:
                    success_rate = 0

                if success_rate >= 0.95:
                    df[col] = converted_datetime
                    print("→ detected as datetime")
                    continue

            # ==================================================
            # 4. Nothing matched
            # ==================================================

            print("→ keeping as string/object")

        return df

    col_converted_types = Convert_Types(df=normalised_df)
    # print(col_converted_types.dtypes)


    ##### Duplicate Detection & Removal (AFTER type conversion) #####

    def Duplicate_Detection_Removal(df):
        
        number_duplicates_rows = df.duplicated().sum()
        print(f"number is : {number_duplicates_rows}")
        
        
        cleaned_total_numb_rows = df.shape[0]
        percentage = (number_duplicates_rows / cleaned_total_numb_rows ) * 100
        print(f"percentage :{round(percentage,2)}%")

        n_rows_befor = df.shape[0]
        print(f"number of rows and columns befor Duplicate_Detection_Removal :\n{n_rows_befor}")
        cleaned_df = df.drop_duplicates(keep="first")
        n_rows_after = cleaned_df.shape[0]
        print(f"number of rows and columns after Duplicate_Detection_Removal :\n{n_rows_after}")
        duplicate_metrics = {
                                "duplicates_found": int(df.duplicated().sum()),
                                "duplicates_percentage": float(round(percentage , 2)),
                                "rows_before": int(n_rows_befor),
                                "rows_after": int(n_rows_after),
                            }
        
        return (cleaned_df,duplicate_metrics) 
        
    cleaned_from_dup_df, duplicate_metrics = Duplicate_Detection_Removal(df = col_converted_types )

    # print( cleaned_from_dup_df, duplicate_metrics)


    def handle_missing_values(df):
        numeric_count = 0
        non_numeric_count = 0
        count_per_column ={}
        
        
        for col in df.columns :
            if pd.api.types.is_numeric_dtype(df[col]) :
                median_ = df[col].median()
                v1 = df[col].isna().sum()
                numeric_count += v1
                count_per_column[col] = v1
                df[col] = df[col].fillna(median_)
                
            else :
                v2 = df[col].isna().sum()
                non_numeric_count += v2
                count_per_column[col]=v2
                df[col] = df[col].fillna("unknown")
        
        total = numeric_count+non_numeric_count
        missing_metrics = {"count_per_column" :count_per_column ,
                           "non_numeric_count" : non_numeric_count ,
                           "numeric_count" : numeric_count ,
                           "total" : total
                           }
        
        return df , missing_metrics
        
    clean_df, missing_metrics = handle_missing_values(df=cleaned_from_dup_df)  # noqa: RUF059
    print(clean_df.isna().sum())

    ##### detect anomalies #####
    def detect_anomalies(df, contamination) :
        model = IsolationForest(contamination=contamination, random_state=42)
        
        only_num_col = df.select_dtypes(include="number")
        copy_df = only_num_col.fillna(only_num_col.median())
        
        if only_num_col.empty :
            print("warnning , there is no numeric columns ! ")
            return df , {}
        else :
            print("data has numeric data")
            copy_df = only_num_col.copy()
            
            prediction = model.fit_predict(copy_df)
            anomalies = prediction == -1
            anomalies_count = anomalies.sum()
            
            percentage = (anomalies_count / len(prediction) ) * 100
            metrics = {
                            "anomalies_found": int(anomalies.sum()),
                            "anomaly_percentage": float(round(percentage , 2)),
                            "contamination": float(contamination),
                            "numeric_columns_used": list(copy_df.columns),
                        }
            df["isolaation_flag"] = anomalies
            return df,metrics 

    df_with_flag , metrics =detect_anomalies(df = clean_df , contamination=0.05)
    

          
    ##### build_embeddings #####
    messy_columns = list(df_with_flag.columns)
    canonical_column_names_ = CANONICAL_FIELDS
    def build_embeddings(canonical_column_names ,messy_columns_):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        messy_embeddings = model.encode(messy_columns_)
        canonical_embeddings = model.encode(canonical_column_names)
        print(messy_embeddings.shape)
        print(canonical_embeddings.shape)
        similarities = model.similarity(messy_embeddings , canonical_embeddings)
        print(similarities)
        return (model ,canonical_column_names ,canonical_embeddings)
    model ,canonical_column_names,canonical_embeddings = build_embeddings(canonical_column_names =canonical_column_names_ ,messy_columns_ = messy_columns )

    def preprocess_messy_column__name(name) : 
        ABBREVIATIONS = {    
                            "nm": "name",
                            "dt": "date",
                            "id": "identifier",
                            "dob": "date of birth",
                            "addr": "address",
                            "tel": "telephone",
                            "ph": "phone",
                            "qty": "quantity",
                            "amt": "amount",
                            "num": "number",
                            "no": "number",
                            "dept": "department",
                            "emp": "employee",
                            "cust": "customer",
                            "prod": "product",
                        }
        v1 = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
        v2 = v1.replace("_"," ").replace("-"," ")
        v3 = re.sub(r"\s+", " ", v2).lower()

        if " " in v3 :
            splited_v3 = v3.split()

            for ab in ABBREVIATIONS :
                for index , item in enumerate(splited_v3) :
                    if ab == item :
                        splited_v3[index] = ABBREVIATIONS[ab] 

            v3 = " ".join(splited_v3)
        else :
            for ab in ABBREVIATIONS :
                if ab == v3 :
                    v3 = ABBREVIATIONS[ab] 
        print(v3)          
        return v3 
    preprocess_messy_column__name("FirsNm")
    
            

        
    ##### timer stops #####
    end_ = time.perf_counter()
    print(f"it takes {end_ - start_}s")

if __name__ == "__main__" :
    main()