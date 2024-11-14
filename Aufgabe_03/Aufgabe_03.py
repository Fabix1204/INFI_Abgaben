import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_excel('Aufgabe_03/data/Zeitreihe-Winter-2024011810.xlsx')
df2 = pd.read_excel('Aufgabe_03/data/bev_meld.xlsx')

both = pd.merge(df, df2, on='Gemnr', how='inner', suffixes=('_touristen', '_einwohner'))
both.columns = both.columns.astype(str)
both = both.loc[:, ~both.columns.str.endswith('_dup')]

base = ['Bez', 'Gemnr', 'Gemeinde']
years_both = both.columns[3:].astype(str)
base.extend('x' + years_both)
years = df.columns[3:].astype(str)
base.extend('x' + years)

print(tabulate(both.head(10), headers='keys', tablefmt='psql'))

#print(df.describe())
# Bei Describe werden statistische Kennzahlen wie Mittelwert, Standardabweichung, Minimum, Maximum,
# Quantile und Anzahl der Werte ausgegeben. Dies ist sehr hilfreich, um einen ersten Überblick über die Daten zu bekommen.

# 2.1

bez = df[df.Bez == 'I']
val = bez[df.columns[3:]].values[0, :]
plt.plot(val)
plt.plot(years, val)
plt.xticks(rotation=90)
plt.show()
# Interpretation: Die Übernachtungen sind von 2000 bis 2019 gestiegen.
# Wegen Corona gab es jedoch in den Jahren 2020 und 2021 einen starken Rückgang.
# 2022 sind die Übernachtungen fast wieder auf den Wert von 2019 gestiegen.

# 2.2

my_bez = df[df['Bez'] == 'IL']
yearly_sums = my_bez[df.columns[3:]].sum(axis=0)
years = df.columns[3:]
plt.plot(years, yearly_sums)
plt.xticks(rotation=90)
plt.ylabel('Summe der Werte')
plt.xlabel('Jahr')
plt.title('Summe der Werte pro Jahr fuer den Bezirk IL')
plt.show()

# Interpretation: Die Übernachtungen sind von 2000 bis 2019 sehr stark am schwanken.
# Im Jahr 2008 waren sie am höchsten, vermutlich wegen der Fußball-Europameisterschaft.
# Auch hier gab es 2020 und 2021 einen starken Rückgang.

# 3.1

df['min'] = df.iloc[:, 3:].min(axis=1)

df['max'] = df.iloc[:, 3:].max(axis=1)

df['range'] = df['max'] - df['min']

df['mean'] = df.iloc[:, 3:].mean(axis=1)

# 3.2

# Berechnung der Gesamtzahl an Touristen pro Jahr
yearly_sum = df.iloc[:, 2:].sum(axis=0).drop(['Gemeinde'])  # Start bei der 3. Spalte für die Jahreszahlen
print("Gesamtzahl der Touristen pro Jahr:\n", yearly_sum)

yearly_sum.drop(['Gemeinde'])
# Gesamtzahl der Touristen über alle Jahre
total_sum = yearly_sum.sum()
print("Gesamtzahl der Touristen über alle Jahre:", total_sum)

# Zusammenfassung nach Bezirken (Gesamtzahl der Touristen je Bezirk über alle Jahre)
sum_bez = df.groupby('Bez').sum(numeric_only=True)
print("Zusammenfassung der Gesamtzahl der Touristen nach Bezirk:\n", sum_bez)

# Plotten der Zusammenfassung nach Bezirk als Balkendiagramm
sum_bez.sum(axis=1).plot.bar()  # Summiert über die Jahre für jeden Bezirk
plt.ylabel('Gesamtzahl der Touristen')
plt.title('Touristen nach Bezirk')
plt.show()

# 4.1

# a)

df.boxplot(column='range', by='Bez')
plt.show()
# Interpretation: Einige Bezirke, wie zum Beispiel "I" und "LA," weisen besonders hohe Ausreißer auf,
# die weit über dem Durchschnitt der Daten liegen,
# während andere Bezirke wie "IL" und "RE" eher niedrige Werte mit kleineren Variationsbreiten haben.

# b)

pos = 0
labels = df['Bez'].unique()
positions = []
for b in labels:
    bez = df[df.Bez == b]
    plt.boxplot(bez['range'], positions=[pos])
    positions.append(pos)
    pos += 1
plt.xticks(positions, labels)
plt.show()

# c)

sns.boxplot(x=df['Bez'], y=df['range'], data=df)
plt.show()

# 4.2

innsbruck_data = df[df['Bez'] == 'I']

years = list(range(2000, 2024))
yearly_values = innsbruck_data[years].values.flatten()

# Create the bar plot
plt.bar(years, yearly_values)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Yearly Values for Innsbruck')
plt.show()
# Interpretation: Das Balkendiagramm zeigt einen ansteigenden Trend der jährlichen Werte für Innsbruck
# von 2000 bis etwa 2019, gefolgt von einem deutlichen Rückgang in den Jahren 2020 und 2021,
# bevor die Werte 2022 wieder ansteigen.
# Der Rückgang im Jahr 2020 könnte auf Corona zurückzuführen sein.

#5
# a)
both['Touristen_pro_Einwohner'] = both['2018_touristen'] / both['2018_einwohner']
tourist_per_bev_2018 = both['2018_touristen'] / both['2018_einwohner']
# b)
plt.figure(figsize=(10, 6))
sns.boxplot(x=both['Bez'], y=both['Touristen_pro_Einwohner'])
plt.xlabel('Bezirk')
plt.ylabel('Touristen pro Einwohner')
plt.title('Standardisierte Anzahl Naechtigungen im Jahr 2018 pro Bezirk')
plt.xticks(rotation=45)
plt.show()
# Interpretation: Das Boxplot zeigt, dass der Bezirk "LA" im Jahr 2018 eine hohe Variabilität
# in der Anzahl der Touristen pro Einwohner aufweist, mit mehreren Ausreißern,
# die über 600 Touristen pro Einwohner erreichen.
# Im Vergleich dazu haben die meisten anderen Bezirke eine niedrigere und weniger variable Rate,
# wobei einige ebenfalls kleinere Ausreißer aufweisen.

# c)
top_10 = both.nlargest(10, 'Touristen_pro_Einwohner')[['Gemeinde_touristen', 'Touristen_pro_Einwohner']]
print("Top 10 Gemeinden mit dem größten Verhältnis von Touristen pro Einwohner:")
print(top_10)

bottom_10 = both.nsmallest(10, 'Touristen_pro_Einwohner')[['Gemeinde_touristen', 'Touristen_pro_Einwohner']]
print("\nTop 10 Gemeinden mit dem kleinsten Verhältnis von Touristen pro Einwohner:")
print(bottom_10)

# d)

both['Gemeinde_touristen'] = both['Gemeinde_touristen'].str.strip()

Oberperfuss = both[both['Gemeinde_touristen'].str.contains('Oberperfuss', case=False, na=False)]

touristen_pro_einwohner_oberperfuss = Oberperfuss['Touristen_pro_Einwohner'].values[0]
