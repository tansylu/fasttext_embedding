import pandas as pd

# Read excel file with sheet name
dict_df = pd.read_excel('database.xlsx', sheet_name=['Sheet1','ProductTypeName_of_hiot (2)', 'ProductTypeName (2)'])
# Get DataFrame from Dict
unique_type = dict_df.get('ProductTypeName (2)')
unique_hiot_type = dict_df.get('ProductTypeName_of_hiot (2)')
main_df = dict_df.get('Sheet1')

main_df = main_df[main_df['ProductTypeName'].isin(unique_type['ProductTypeName'])]
main_df.to_excel("food_related_emissions.xlsx")