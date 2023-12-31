import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv('Groceries_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], format= "%d-%m-%Y")

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustur","September","Oktober","November","Desember"], inplace=True)
df["day"].replace([i for i in range(6 + 1)], ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"],inplace=True)

st.title("UAS Grocery Basket Analysis Algoritma Apriori")

def get_data( month ='' , day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", ['whole milk','other vegetables','rolls/buns','soda','yogurt','root vegetables','tropical fruit','bottled water','sausage','citrus fruit','pastry','pip fruit','shopping bags','canned beer','bottled beer','whipped/sour cream','newspapers','frankfurter','brown bread','pork','domestic eggs','butter','fruit/vegetable juice','beef','curd'])
    month = st.select_slider("Month", ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"])
    day = st.select_slider("Day", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"], value='Senin')

    return item, month, day

item, month, day = user_input_features()

data = get_data(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(data) != type ("No Result"):
    item_count = data.groupby(['Member_number', 'itemDescription'])["itemDescription"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Member_number', columns='itemDescription', values='Count', aggfunc='sum').fillna(0) 
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False,inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
     
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return []

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    result = return_item_df(item)
    if result :
        st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
    else:
        st.warning("Tidak ditemukan rekomendasi untuk item yang dipilih")
    
