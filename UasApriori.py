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
    item = st.selectbox("Item", ['whole milk','other vegetables','rolls/buns','soda','yogurt','root vegetables','tropical fruit','bottled water','sausage','citrus fruit','pastry','pip fruit','shopping bags','canned beer','bottled beer','whipped/sour cream','newspapers','frankfurter','brown bread','pork','domestic eggs','butter','fruit/vegetable juice','beef','curd','margarine','coffee','frozen vegetables','chicken','white bread','cream cheese','chocolate','dessert','napkins','hamburger meat','berries','UHT-milk','onions','salty snack','waffles','long life bakery product','sugar','butter milk','ham','meat','frozen meals','beverages','specialty chocolate','misc. beverages','ice cream','oil','hard cheese','grapes','candy','sliced cheese','specialty bar','hygiene articles','chewing gum','cat food','white wine','herbs','red/blush wine','soft cheese','processed cheese','flour','semi-finished bread','dishes','pickled vegetables','detergent','packaged fruit/vegetables','baking powder','pasta','pot plants','canned fish','seasonal products','liquor','frozen fish','spread cheese','condensed milk','cake bar','mustard','frozen dessert','salt','pet care','canned vegetables','roll products','turkey','photo/film','mayonnaise','cling film/bags','dish cleaner','frozen potato products','specialty cheese','flower (seeds)','sweet spreads','liquor (appetizer)','dog food','candles','finished products','instant coffee','chocolate marshmallow','Instant food products','zwieback','vinegar','liver loaf','rice','soups','popcorn','sparkling wine','curd cheese','house keeping products','sauces','cereals','softener','female sanitary products','spices','brandy','male cosmetics','meat spreads','jam','nuts/prunes','dental care','rum','ketchup','cleaner','kitchen towels','light bulbs','fish','artif. sweetener','specialty fat','snack products','tea','potato products','nut snack','abrasive cleaner','organic sausage','tidbits','canned fruit','syrup','skin care','soap','prosecco','pudding powder','cookware','bathroom cleaner','flower soil/fertilizer','cocoa drinks','cooking chocolate','ready soups','honey','cream','specialty vegetables','frozen fruits','organic products','liqueur','hair spray','decalcifier','whisky','salad dressing','make up remover','toilet cleaner','frozen chicken','rubbing alcohol','bags','baby cosmetics','kitchen utensil','preservation products'])
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

    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0,:])

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
    
