# Pandas

## Pandas Tricks

### Select row dataframe by value

use [isin](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html) method

```python
>>> df = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
                      index=['falcon', 'dog'])
                  
>>> df
        num_legs  num_wings
falcon         2          2
dog            4          0

```

```python
>>> df.num_wings.isin([2])
falcon     True
dog       False
Name: num_wings, dtype: bool
```
