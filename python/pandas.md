# Pandas

https://pandas.pydata.org

pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.

[User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

[API reference](https://pandas.pydata.org/docs/reference/index.html)

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
