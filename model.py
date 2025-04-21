import numpy as np
import pandas as pd
from keras_core.models import Sequential
from keras_core.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras_core.optimizers import (Adam)
from keras_core.losses import MeanSquaredError
from keras_core.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error



df = pd.read_csv('cleaned_common_players_filled_modified.csv')

df = df.drop(columns=['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Born'])
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


pierwsze_trzy_df = pd.DataFrame()
czwarty_df = pd.DataFrame()
piaty_df = pd.DataFrame()

for i in range(0, len(df), 5):
    pierwsze_trzy_df = pd.concat([pierwsze_trzy_df, df.iloc[i:i + 3]])

for i in range(3, len(df), 5):
    czwarty_df = pd.concat([czwarty_df, df.iloc[i:i + 1]])

for i in range(4, len(df), 5):
    piaty_df = pd.concat([piaty_df, df.iloc[i:i + 1]])

pierwsze_trzy_df = pierwsze_trzy_df.reset_index(drop=True)
czwarty_df = czwarty_df.reset_index(drop=True)
piaty_df = piaty_df.reset_index(drop=True)

new_data = []
for i in range(0, len(pierwsze_trzy_df), 3):
    player_data = df.iloc[i:i+3]
    new_data.append(player_data)

pierwsze_trzy_data = np.array(new_data)

czwarty_df = czwarty_df.drop(columns=['Age'])
piaty_df = piaty_df.drop(columns=['Age'])

liczba_cech_wejscie = pierwsze_trzy_data.shape[2]
liczba_cech_wyjscie = czwarty_df.shape[1]

pierwsze_trzy_data_treningowe=pierwsze_trzy_data[:715]
pierwsze_trzy_data_testowe=pierwsze_trzy_data[715:]
czwarty_etykieta_treningowa=czwarty_df.iloc[:715]
czwarty_etykieta_testowa=czwarty_df.iloc[715:]
piaty_etykieta_treningowa=piaty_df.iloc[:715]
piaty_etykieta_testowa=piaty_df.iloc[715:]

model4 = Sequential([
    LSTM(64,input_shape=(3,liczba_cech_wejscie)),
    Dense(32, activation='relu'),
    Dense(liczba_cech_wyjscie)
])

model4.compile(optimizer=Adam(learning_rate=0.0005),
              loss=MeanSquaredError(),
              metrics=[MeanAbsoluteError()])

model5 = Sequential([
    LSTM(64,input_shape=(3,liczba_cech_wejscie)),
    Dense(32, activation='relu'),
    Dense(liczba_cech_wyjscie)

])

model5.compile(optimizer=Adam(learning_rate=0.0005),
              loss=MeanSquaredError(),
              metrics=[MeanAbsoluteError()])

history = model4.fit(
    pierwsze_trzy_data_treningowe, czwarty_etykieta_treningowa,
    validation_data=(pierwsze_trzy_data_testowe, czwarty_etykieta_testowa),
    epochs=125,
    batch_size=16,
    verbose=1
)

history2 = model5.fit(
    pierwsze_trzy_data_treningowe, piaty_etykieta_treningowa,
    validation_data=(pierwsze_trzy_data_testowe, piaty_etykieta_testowa),
    epochs=125,
    batch_size=16,
    verbose=1
)



czwarty_etykieta_testowa['Age'] = 0
czwarty_etykieta_testowa = czwarty_etykieta_testowa[['Age'] + [col for col in czwarty_etykieta_testowa.columns if col != 'Age']]
piaty_etykieta_testowa['Age'] = 0
piaty_etykieta_testowa = piaty_etykieta_testowa[['Age'] + [col for col in piaty_etykieta_testowa.columns if col != 'Age']]
y_test_4 = scaler.inverse_transform(czwarty_etykieta_testowa)
y_test_5 = scaler.inverse_transform(piaty_etykieta_testowa)
y_test_4 = np.delete(y_test_4,0,axis=1)
y_test_5 = np.delete(y_test_5,0,axis=1)


y_pred_4 = model4.predict(pierwsze_trzy_data_testowe)
zeros_column = np.zeros((y_pred_4.shape[0], 1))
y_pred_4 = np.hstack((zeros_column, y_pred_4))
y_pred_4 = scaler.inverse_transform(y_pred_4)
y_pred_4 = np.delete(y_pred_4,0,axis=1)

mae_4 = mean_absolute_error(y_test_4, y_pred_4, multioutput='raw_values')


y_pred_5 = model5.predict(pierwsze_trzy_data_testowe)
zeros_column = np.zeros((y_pred_5.shape[0], 1))
y_pred_5 = np.hstack((zeros_column, y_pred_5))
y_pred_5 = scaler.inverse_transform(y_pred_5)
y_pred_5 = np.delete(y_pred_5,0,axis=1)

mae_5 = mean_absolute_error(y_test_5, y_pred_5, multioutput='raw_values')


mean_mae_4 = np.mean(mae_4)
mean_mae_5 = np.mean(mae_5)


print("MAE dla każdej cechy w modelu 4:", mae_4)
print("Średni MAE dla modelu 4:", mean_mae_4)
print("MAE dla każdej cechy w modelu 5:", mae_5)
print("Średni MAE dla modelu 5:", mean_mae_5)



plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss during training')
plt.show()
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss during training')
plt.show()

