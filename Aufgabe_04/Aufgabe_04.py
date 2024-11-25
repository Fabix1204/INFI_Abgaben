import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1
df = pd.read_excel('Aufgabe_04/data/bev_meld.xlsx')

# 2
# 2.1
sum = df.sum(axis=0)[3:]
plt.plot(sum)
plt.show()

# 2.2
df_reg = pd.DataFrame({"years": df.columns[3:], "sum": sum})
df_reg = df_reg.astype({'years':'int', 'sum':'int'})

model = sm.OLS.from_formula('sum ~ years', df_reg).fit()

a = model.params[1]
b = model.params[0]

y = a*2030 +b
# y(2030) = 794332.8014778234

# Ich habe die Prediction so angepasst, dass sie schon ab 2021 beginnt,
# damit die Prognose direkt an den vorhandenen Daten anschließt. (schöner Übergang in der Grafik)
pred_years = np.arange(2021, 2100)

df_pred = pd.DataFrame({"years": pred_years})
predictions = model.predict(df_pred)

plt.plot(df_pred.years, predictions)
plt.xlim([2021, 2100])
plt.show()

plt.plot(df_reg.years, df_reg['sum'])
plt.plot(df_pred.years, predictions)
plt.xlim([1993, 2100])
plt.show()
# Interpretation: Die Grafik zeigt einen kontinuierlichen Anstieg der Bevölkerung in Tirol von
# etwa 640.000 im Jahr 1995 auf über 760.000 im Jahr 2020. Dies deutet auf ein stabiles
# Bevölkerungswachstum in diesem Zeitraum hin.
# Die Prognose zeigt ein gleichmäßiges Bevölkerungswachstum in Tirol
# von etwa 750.000 im Jahr 2030 auf über 1 Million im Jahr 2100. Dies deutet auf einen
# kontinuierlichen Anstieg ohne größere Schwankungen hin.

# 3 Bevölkerungsentwicklung der Heimatgemeinde
df_oberperfuss = df[df['Gemeinde'] == 'Oberperfuss']
df_oberperfuss_reg = pd.DataFrame({"years": df_oberperfuss.columns[3:], "sum": df_oberperfuss.sum(axis=0)[3:]})
df_oberperfuss_reg = df_oberperfuss_reg.astype({'years':'int', 'sum':'int'})

model_oberperfuss = sm.OLS.from_formula('sum ~ years', df_oberperfuss_reg).fit()

plt.plot(df_oberperfuss_reg.years, df_oberperfuss_reg['sum'])
plt.plot(df_pred.years, model_oberperfuss.predict(df_pred))
plt.xlim([1993, 2100])
plt.show()

# Interpretation: Die Grafik zeigt einen kontinuierlichen Anstieg der Bevölkerung
# in Oberperfuss von etwa 3.000 Einwohnern im Jahr 2021 auf ca. 5.000 Einwohner
# in der Prognose von 2100.

# 4 Bezirke IL und RE vergleichen
df_IL = df[df['Bezirk'] == 'IL']
df_RE = df[df['Bezirk'] == 'RE']

# Summiere die Bevölkerung pro Jahr für beide Bezirke
sum_IL = df_IL.sum(axis=0)[3:]
sum_RE = df_RE.sum(axis=0)[3:]

# Erstelle DataFrames für die Regression
df_IL_reg = pd.DataFrame({"years": df_IL.columns[3:], "sum": sum_IL})
df_IL_reg = df_IL_reg.astype({'years':'int', 'sum':'int'})

df_RE_reg = pd.DataFrame({"years": df_RE.columns[3:], "sum": sum_RE})
df_RE_reg = df_RE_reg.astype({'years':'int', 'sum':'int'})

# Modelle erstellen
model_IL = sm.OLS.from_formula('sum ~ years', df_IL_reg).fit()
model_RE = sm.OLS.from_formula('sum ~ years', df_RE_reg).fit()

# Vorhersagen erstellen
predictions_IL = model_IL.predict(df_pred)
predictions_RE = model_RE.predict(df_pred)

# Grafiken erstellen
plt.plot(df_IL_reg.years, df_IL_reg['sum'], label='IL')
plt.plot(df_RE_reg.years, df_RE_reg['sum'], label='RE')
plt.plot(df_pred.years, predictions_IL, label='IL Prediction')
plt.plot(df_pred.years, predictions_RE, label='RE Prediction')
plt.xlim([1993, 2100])
plt.show()

# Interpretation: Die Grafik zeigt die Bevölkerungsentwicklung zweier Bezirke:
# Im Bezirk Innsbruck-Land (blau/grün) wächst die Bevölkerung deutlich und prognostisch stark,
# während im Bezirk Reutte (orange/rot) nur ein geringfügiger Anstieg zu sehen ist.
# Der Unterschied in der Bevölkerungsgröße zwischen den Bezirken ist erheblich, wobei Innsbruck-Land
# durchgehend dominiert. Die Prognosen setzen die Trends fort.









