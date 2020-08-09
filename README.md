# GDP_Predictor

Short Python Machine Learning Program that uses different kinds of regressions and data sets to predict GDP.

The best data set uses the following:
```
GDPC1: Real Gross Domestic Product, Percent Change of (Billions of Chained 2012 Dollars), Quarterly, Seasonally Adjusted Annual Rate

CCSA: Continued Claims (Insured Unemployment), Number, Quarterly, Seasonally Adjusted

NEWORDER: Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft, Percent Change of (Millions of Dollars), Quarterly, Seasonally Adjusted

UMCSENT: University of Michigan: Consumer Sentiment, Index 1966:Q1=100, Quarterly, Not Seasonally Adjusted

PAYEMS: All Employees, Total Nonfarm, Percent Change of (Thousands of Persons), Quarterly, Seasonally Adjusted

CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, Percent Change of (Index 1982-1984=100), Quarterly, Seasonally Adjusted

MSACSR: Monthly Supply of Houses in the United States, Months' Supply, Quarterly, Seasonally Adjusted
```

All data comes directly from the St. Louis Federal Reserve Bank, https://fred.stlouisfed.org/

The best regressions used were Elastic Net Regression and Ridge Regression. For 2020 Quarter 2, the Elastic Net Regression predicts the drop in GDP to be
```
Predicted: -7.86 percent
Annualized: -31.45
```
The Ridge Regression predicts the drop in GDP to be 
```
Predicted : -7.49 percent
Annualized: -29.96
```

From the July 30, 2020 Federal Reserve Report the true drop in GDP was
```
-9.49 percent (Annualized: -32.9 percent)
```

