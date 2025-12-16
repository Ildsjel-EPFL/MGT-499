import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def restauration_employement_to_population_ratio(url_restauration : str, url_population : str) -> pd.DataFrame:
    df_restauration = pd.read_csv(url_restauration, encoding="ANSI")
    df_population = pd.read_csv(url_population, encoding="ANSI")
    df_population = df_population.groupby(["Année", "Sexe"], as_index=False)["Effectif au 1er janvier"].agg("sum")
    
    years : List[int] = [i for i in range(2010, 2025) for _ in range(8)]
    quarters : List[int] = [i for _ in range(15*2) for i in range(1, 5)]
    female : List[int] = [i for _ in range(15) for i in [1]*4+[0]*4]  # 1 if female, 0 if male
    indexes_of_columns_to_go : List[int] = [2+i for j in range((df_restauration.shape[1]-3)//4) for _ in range(2) for i in range(4*j+1, 4*j+5)] #to search efficiently in the dataframe for having the final datafram in the right format easily
    y : List[float] = []

    for index, gender, year in zip(indexes_of_columns_to_go, female, years):
        y.append(df_restauration.iloc[gender, index]/df_population.loc[df_population["Année"].eq(year)&df_population["Sexe"].eq("Femme" if gender==1 else "Homme"), "Effectif au 1er janvier"].iat[0]*100)

    dict : Dict[List[Any]] = {"Year" : years,
            "Quarter" : quarters,
            "Employment-to-population Ratio [%]" : y, 
            "Female" : female}
    return pd.DataFrame(dict)

def create_quarter_dummies(df : pd.DataFrame):
    quarter_lists_list = [[], [], [], []]
    for i in range(df.shape[0]):
        for j in range(1, 5):
            if df["Quarter"][i]==j:
                quarter_lists_list[j-1].append(1)
            else:
                quarter_lists_list[j-1].append(0)
    for i in range(4):
        df[f"Q{i+1}"]=quarter_lists_list[i]
    return df

def create_before_lockdown_dummies(df : pd.DataFrame) -> pd.DataFrame:
    df["Before_Lockdown"] = [1 if i<80 else 0 for i in range(df.shape[0])]
    return df

def create_lockdown_dummies(df : pd.DataFrame, include_after : bool = False) -> pd.DataFrame:
    indexes = [i for i in range(80, 98)]
    if include_after:
        indexes.extend([i for i in range(98, df.shape[0])])
    else:
        indexes.extend([100, 101])
    dummy_list = [1 if i in indexes else 0 for i in range(df.shape[0])]
    df[f"{"After_" if include_after else "During_"}Lockdown"] = dummy_list
    return df

def create_not_lockdown_dummies(df : pd.DataFrame) -> pd.DataFrame:
    indexes = [i for i in range(80, 98)]
    indexes.extend([100, 101])
    dummy_list = [0 if i in indexes else 1 for i in range(df.shape[0])]
    df["Not_Covid"] = dummy_list
    return df

def create_after_lockdown_dummies(df: pd.DataFrame) -> pd.DataFrame:
    indexes_list = [98, 99]
    indexes_list.extend(i for i in range(102, df.shape[0]))
    dummy_list = [1 if i in indexes_list else 0 for i in range(df.shape[0])]
    df["After_End_Measures"] = dummy_list
    return df

def create_year_pre_post_dummies(df : pd.DataFrame) -> pd.DataFrame:
    pre_dummies = [1 if 72<=i<80 else 0 for i in range(df.shape[0])]
    post_dummies = [1 if i in [98, 99, 102, 103, 104, 105, 108, 109] else 0 for i in range(df.shape[0])]
    df["Pre"] = pre_dummies
    df["Post"] = post_dummies
    return df

def plot_ratio(df : pd.DataFrame) -> None:
    time = np.linspace(0, 59, 60)
    _, ax = plt.subplots(figsize=(15, 10))
    ax.plot(time, df.loc[df["Female"]==0, "Employment-to-population Ratio [%]"], marker="o", label="Male")
    ax.plot(time, df.loc[df["Female"]==1, "Employment-to-population Ratio [%]"], marker="o", label="Female")
    ax.axvline(40, c="r", ls="--", label="Lockdown : $16^{th}$ March 2020 (Q1)")
    ax.axvline(50, c="green", ls="--", label="End of measures : $1^{st}$ April 2022 (Q2)")
    ax.axvspan(40, 50, color="grey", alpha=0.2)
    plt.xlim(0, 59)
    plt.xticks([i for i in time if i%4==0], labels=[f"20{10+i}" for i in range(15) for j in range(4) if j==0])
    plt.ylabel("Employment-to-population ratio [%]")
    plt.legend()
    plt.grid()
    plt.title("Evolution of the Employment-to-population ratio by genders in the restoration sector between 2010 and 2025")
    plt.show()

def plot_comparative(df : pd.DataFrame) -> pd.DataFrame:
    before_covid_female = df.iloc[75, 2]
    before_covid_male = df.iloc[79, 2]
    time = np.linspace(0, 59, 60)
    _, ax = plt.subplots(figsize=(15, 10))
    ax.plot(time, df.loc[df["Female"]==0, "Employment-to-population Ratio [%]"].to_numpy()/before_covid_male-1, marker="o", label="Male")
    ax.plot(time, df.loc[df["Female"]==1, "Employment-to-population Ratio [%]"].to_numpy()/before_covid_female-1, marker="o", label="Female")
    ax.axvline(40, c="r", ls="--", label="Lockdown : $16^{th}$ March 2020 (Q1)")
    ax.axvline(50, c="green", ls="--", label="End of measures : $1^{st}$ April 2022 (Q2)")
    ax.axvspan(40, 50, color="grey", alpha=0.2)
    ax.hlines(0, xmin=0, xmax=59, color="k")
    plt.xlim(0, 59)
    plt.ylim(-0.2, 0.2)
    plt.xticks([i for i in time if i%4==0], labels=[f"20{10+i}" for i in range(15) for j in range(4) if j==0])
    plt.ylabel("Difference with 2019Q4 [%]")
    plt.legend()
    plt.grid()
    plt.title("Employment-to-population ratio compared to the last quarter before Covid Lockdown by genders in the restoration sector between 2010 and 2025")
    plt.show()