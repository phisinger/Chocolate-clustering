import requests
import pandas as pd
import psycopg2 as ps
from sqlalchemy import create_engine
import json
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import re
from sklearn.preprocessing import LabelEncoder


def load_api_key():
    # loading API secret key
    f = open("secrets/FoodData_central_key.txt", "r")
    return f.read()


def parse_response(response):
    # function for parsing the incoming JSON into a dictionary
    # in order to be able to build da Pandas Dataframe

    item_page_list = []
    for item in response["foods"]:
        choco = {
            "id": item.get("fdcId"),
            "description": item.get("description"),
            "brandOwner": item.get("brandOwner"),
            "brandName": item.get("brandName"),
            "ingredients": item.get("ingredients"),
            "marketCountry": item.get("marketCountry"),
            "dataSource": item.get("dataSource"),
            "packageWeight": item.get("packageWeight"),
            "servingSizeUnit": item.get("servingSizeUnit"),
            "servingSize": item.get("servingSize"),
            "score": item.get("score"),
            "Protein": None,
            "Protein_unit": None,
            "fat": None,
            "fat_unit": None,
            "Carbohydrate": None,
            "Carbohydrate_unit": None,
            "Energy": None,
            "Energy_unit": None,
            "Fiber": None,
            "Fiber_unit": None,
            "Calcium": None,
            "Calcium_unit": None,
            "Iron": None,
            "Iron_unit": None,
            "Vitamin_D": None,
            "Vitamin_D_unit": None,
            "Sugars_added": None,
            "Sugars_added_unit": None,
            "Cholesterol": None,
            "Cholesterol_unit": None
        }

        # the following attributes are a little tricky.
        # They are not always given and you have to search for them
        for nutrient in item['foodNutrients']:
            if "Protein" in nutrient["nutrientName"]:
                choco["Protein"] = nutrient["value"]
                choco["Protein_unit"] = nutrient["unitName"]
            elif "fat" in nutrient["nutrientName"]:
                choco["fat"] = nutrient["value"]
                choco["fat_unit"] = nutrient["unitName"]
            elif "Carbohydrate" in nutrient["nutrientName"]:
                choco["Carbohydrate"] = nutrient["value"]
                choco["Carbohydrate_unit"] = nutrient["unitName"]
            elif "Energy" in nutrient["nutrientName"]:
                choco["Energy"] = nutrient["value"]
                choco["Energy_unit"] = nutrient["unitName"]
            elif "Fiber" in nutrient["nutrientName"]:
                choco["Fiber"] = nutrient["value"]
                choco["Fiber_unit"] = nutrient["unitName"]
            elif "Calcium" in nutrient["nutrientName"]:
                choco["Calcium"] = nutrient["value"]
                choco["Calcium_unit"] = nutrient["unitName"]
            elif "Iron" in nutrient["nutrientName"]:
                choco["Iron"] = nutrient["value"]
                choco["Iron_unit"] = nutrient["unitName"]
            elif "Vitamin D" in nutrient["nutrientName"]:
                choco["Vitamin_D"] = nutrient["value"]
                choco["Vitamin_D_unit"] = nutrient["unitName"]
            elif "Sugars" in nutrient["nutrientName"]:
                choco["Sugars_added"] = nutrient["value"]
                choco["Sugars_added_unit"] = nutrient["unitName"]
            elif "Cholesterol" in nutrient["nutrientName"]:
                choco["Cholesterol"] = nutrient["value"]
                choco["Cholesterol_unit"] = nutrient["unitName"]

        item_page_list.append(choco)
    return item_page_list


def request_data():
    # function to request data from the API, saves it into a data Frame

    url = "https://api.nal.usda.gov/fdc/v1/"
    endpoint = "foods/search"
    params = {
        "api_key": load_api_key(),
        "query": "foodCategory:Chocolate",
        "pageSize": "200",
        "dataType": "Branded",
        "pageNumber": 1
    }

    total_list = []
    for page in range(1, (requests.get(url=url + endpoint, params=params).json())['totalPages']+1):
        # use paging
        params["pageNumber"] = page
        response = requests.get(url=url + endpoint, params=params).json()
        if page == 2:
            f = open("..\\data\\response_foods.json", "w")
            json.dump(response['foods'][0], f)
        total_list.extend(parse_response(response))

    # convert to Pandas DataFrame
    data_df = pd.DataFrame(total_list)
    # save the raw data to csv
    data_df.to_csv("api_raw_data.csv", index=False)

    return data_df


def handling_unit_columns(raw_data):
    # Convert values
    raw_data["servingSize"] = [serving if unit == 'g' else float(
        serving)*1.3 for serving, unit in zip(raw_data["servingSize"], raw_data["servingSizeUnit"])]
    raw_data["Energy"] = [energy if unit == 'KCAL' else float(
        energy)*0.2388 for energy, unit in zip(raw_data["Energy"], raw_data["Energy_unit"])]

    # I will add the unit to the column name
    rename_cols = {"Protein": "protein_in_g",
                   "fat": "fat_in_g",
                   "Carbohydrate": "carbohydrate_in_g",
                   "Fiber": "fiber_in_g",
                   "Calcium": "calcium_in_mg",
                   "Iron": "iron_in_mg",
                   "Sugars_added": "sugars_added_in_g",
                   "Cholesterol": "cholesterol_in_g",
                   "servingSize": "serving_size_in_g",
                   "Energy": "energy_in_kcal"
                   }
    # rename column names
    raw_data = raw_data.rename(columns=rename_cols)
    # drop unit columns
    return_data = raw_data.drop(columns=["servingSizeUnit", "Protein_unit", "fat_unit", "Carbohydrate_unit",
                                "Energy_unit", "Fiber_unit", "Calcium_unit", "Iron_unit", "Sugars_added_unit", "Cholesterol_unit"])

    return return_data


def handling_missing_values(raw_data):
    data_reduced_rows = raw_data.dropna(
        subset=["protein_in_g", "fat_in_g", "carbohydrate_in_g", "energy_in_kcal", "sugars_added_in_g"])
    data_imputed = data_reduced_rows.copy()
    data_imputed[["fiber_in_g", "calcium_in_mg", "iron_in_mg", "cholesterol_in_g"]] = data_reduced_rows[[
        "fiber_in_g", "calcium_in_mg", "iron_in_mg", "cholesterol_in_g"]].fillna(value=0)
    data_imputed["brandName"] = [brand_name if brand_name else brand_owner for brand_name,
                                 brand_owner in zip(data_reduced_rows["brandName"], data_reduced_rows["brandOwner"])]

    # clean package weight: take only the gram value, every other value will get the mean value
    data_imputed["packageWeight"] = [float(re.search(r"\d+\.?\d*\s?g", weight).group()[:-1])
                                     if re.search(r"\d+\.?\d*\s?g", str(weight or ""))
                                     else np.nan
                                     for weight in data_reduced_rows["packageWeight"]]
    # fill nan with mean
    data_imputed["packageWeight"] = data_imputed["packageWeight"].fillna(
        data_imputed["packageWeight"].mean())

    return data_imputed


def clean_list(list_):
    # remove point at the end
    list_ = list_[0:-1]

    # make sure that after every comma is a space
    list_ = re.sub(r',\S', r', ', list_)

    # add quotation marks and clean sub-lists
    list_ = list_.replace('"', '')
    list_ = list_.replace(';', '","')
    list_ = list_.replace('{', '","')
    list_ = list_.replace('}', '","')
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('. ', '","')
    list_ = list_.replace(' (', '","')
    list_ = list_.replace(')', '')
    list_ = list_.replace(' [', '","')
    list_ = list_.replace(']', '')
    list_ = list_.replace('*', '')
    list_ = list_.replace('&', '","')
    list_ = list_.replace(' AND ', '","')

    # add brackets at the beginning and end
    list_ = '["' + list_
    list_ = list_ + '"]'

    return list_


def strip_elements(list_):
    return [string.strip() for string in list_]


def handling_lists(raw_data):
    raw_data["ingredients"] = raw_data["ingredients"].apply(clean_list)
    raw_data["ingredients"] = raw_data["ingredients"].apply(
        eval)  # makes list in strings to real lists
    raw_data["ingredients"] = raw_data["ingredients"].apply(strip_elements)

    # create temp dataframe for ingredients
    ingredient_columns = ["milk", "corn_syrup", "artificial_flavor", "vanilla",
                          "water", "cream", "dark_chocolate", "palm_oil", "lemon", "salt", "almonds", "soy",
                          "coconut", "pecans", "hazelnuts", "white_chocolate", "honey", "total_ingredients"]
    temp_ing = pd.DataFrame(np.zeros((raw_data.shape[0], len(
        ingredient_columns))), columns=ingredient_columns)

    # search for ingredient in list and set the corresponding column to 1
    for index in range(raw_data.shape[0]):
        # print(raw_data.iloc[index]["ingredients"])
        if any("MILK" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["milk"] = 1
        if "CORN SYRUP" in raw_data.iloc[index]["ingredients"]:
            temp_ing.iloc[index]["corn_syrup"] = 1
        if any("ARTIFICIAL FLAVOR" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["artificial_flavor"] = 1
        if any("VANILL" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["vanilla"] = 1
        if "WATER" in raw_data.iloc[index]["ingredients"]:
            temp_ing.iloc[index]["water"] = 1
        if "CREAM" in raw_data.iloc[index]["ingredients"]:
            temp_ing.iloc[index]["cream"] = 1
        if "DARK CHOCOLATE" in raw_data.iloc[index]["ingredients"]:
            temp_ing.iloc[index]["dark_chocolate"] = 1
        if any("PALM" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["palm_oil"] = 1
        if any("LEMON" in s for s in raw_data.iloc[index]["ingredients"]) or any("CITRIC" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["lemon"] = 1
        if any("SALT" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["salt"] = 1
        if any("ALMONDS" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["almonds"] = 1
        if any("SOY" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["soy"] = 1
        if any("COCONUT" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["coconut"] = 1
        if any("PECANS" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["pecans"] = 1
        if any("HAZELNUT" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["hazelnuts"] = 1
        if "WHITE CHOCOLATE" in raw_data.iloc[index]["ingredients"]:
            temp_ing.iloc[index]["white_chocolate"] = 1
        if any("HONEY" in s for s in raw_data.iloc[index]["ingredients"]):
            temp_ing.iloc[index]["honey"] = 1
        temp_ing.iloc[index]["total_ingredients"] = len(
            raw_data.iloc[index]["ingredients"])

    # transform floats to bool, ignore the last column which is a real float
    temp_ing[ingredient_columns[:-1]
             ] = temp_ing[ingredient_columns[:-1]].astype(bool)

    # merge temp df with raw_data
    data_ingredients = pd.concat([raw_data.reset_index(), temp_ing], axis=1)
    # drop old ingredients column
    data_ingredients = data_ingredients.drop(columns=["ingredients", "index"])

    return data_ingredients


def handling_outliers(raw_data):

    col_list = ["protein_in_g",
                "fat_in_g",
                "carbohydrate_in_g",
                "fiber_in_g",
                "calcium_in_mg",
                "iron_in_mg",
                "sugars_added_in_g",
                "cholesterol_in_g",
                "energy_in_kcal"]

    data_norm = raw_data.copy()
    for col in col_list:
        mean = raw_data[col].mean()
        st_dev_up = mean + 3*raw_data[col].std()
        st_dev_down = mean - 3*raw_data[col].std()
        median = raw_data[col].median()

        data_norm[col] = [value if (value >= 0 and value >= st_dev_down and value <=
                                    st_dev_up) else median for value in raw_data[col]]

    # save data for visualisation in other projects/apps
    data_norm.to_csv("cleaned_data.csv", index=False)

    return data_norm


def encode_data(raw_data):
    data_encoded = raw_data.copy()

    le_brandOwner = LabelEncoder()
    le_brandOwner.fit(data_encoded["brandOwner"])
    data_encoded["brandOwner"] = le_brandOwner.transform(
        data_encoded["brandOwner"])

    le_brandName = LabelEncoder()
    le_brandName.fit(data_encoded["brandName"])
    data_encoded["brandName"] = le_brandName.transform(
        data_encoded["brandName"])

    return data_encoded


def clean_data(raw_data):

    data_reduced_col = raw_data.copy().drop(columns=[
        "marketCountry", "dataSource", "Vitamin_D", "Vitamin_D_unit", "score", "id", "description"])
    data_wo_units = handling_unit_columns(data_reduced_col)
    print("units removed")
    data_wo_miss = handling_missing_values(data_wo_units)
    print("missing values cleansed")
    data_wo_list = handling_lists(data_wo_miss)
    print("handling lists completed")
    data_wo_outliers = handling_outliers(data_wo_list)
    print("outliers removed")
    data_encoded = encode_data(data_wo_outliers)

    data_encoded.to_csv("encoded_data.csv", index=False)
    print("data encoded - execution finished")


# execute all functions
if __name__ == "__main__":
    raw_data = request_data()
    print("raw data received")
    clean_data(raw_data)
