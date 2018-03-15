# import pandas
import pandas as pd

# Series is a single column
a = pd.Series(['San Francisco','San Jose','Sacramento'])
print(a)

city_names = pd.Series(['San Francisco','San Jose','Sacramento'])
population = pd.Series([852469,1015798,485199])

# Dataframe imagine as relational data table, (rows and columns)
b = pd.DataFrame({'City_name': city_names, 'Population':population})
print(b) 

