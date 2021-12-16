from json import load
from pandas.io import excel
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import altair as alt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import model_from_json




def line_chart(data):
    chart_data_alt = alt.Chart(chart_data).transform_fold(
        ["2017", "2018", "2019", "2020", "2021"],
        as_=['Año', 'Dias']
        ).mark_line().encode(
            x='Mes:T',
            y='Dias:Q',
            color='Año:N'
        )
    return(st.altair_chart(chart_data_alt, use_container_width=True))

def line_chart2(data):
    chart_data_alt2 = alt.Chart(chart_data2).transform_fold(
        ["2017", "2018", "2019", "2020", "2021"],
        as_=['Año', 'Orden de Trabajo']
        ).mark_line().encode(
            x='Mes:T',
            y='Orden de Trabajo:Q',
            color='Año:N'
        )
    return(st.altair_chart(chart_data_alt2, use_container_width=True))



#Header
st.title("Dashboard Área producción - Mallas Dorstener Chile    ")
currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")
month = date.strftime("%m")
day = date.strftime("%d")

#Metrics
st.header("KPI'S - Metricas")
#st.caption('This is a string that explains something above.')



#Sidebar
st.sidebar.subheader("Configuraciones de visualización")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file")


#d = st.sidebar.date_input("Seleccione una fecha",datetime.date(2019, 7, 1))
col1, col2, col3 = st.columns(3)



if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        print("csv")
    except:
        df =  pd.read_excel(uploaded_file)
        print("excel")

    df.rename(columns={'NV':'Nota Venta', 'NUMOT': 'Orden de Trabajo', 'CODIGO': 'Codigo', 'NOMBRE': 'Nombre', 'UNIDMED':'Unidad de Medida', 'CANTPP':'Cantidad Pedido', 'CANTOK': 'Cantidad Terminada', 'FECHACREA': 'Fecha Creacion', 'FECHAENT':'Fecha Entrega', 'FECHAFIN': 'Fecha Final', 'ESTADO':'Estado', 'TIPO MP':'Materia Prima', 'EB':'Ancho', 'BORDE':'Borde', 'DIAM.':'Diametro', 'LUZ':'Luz', 'Luz en mm':'Luz_mm'}, inplace=True)
    df["Fecha Creacion"] = pd.to_datetime(df["Fecha Creacion"])
    df["Fecha Entrega"] = pd.to_datetime(df["Fecha Entrega"])
    df["Fecha Final"] = pd.to_datetime(df["Fecha Final"])
    df["Dias"] = (df["Fecha Final"] - df["Fecha Entrega"]).dt.days
    df['Año'] = df['Fecha Creacion'].dt.strftime('%Y')
    df['Mes'] = df['Fecha Creacion'].dt.strftime('%m')

    d = st.sidebar.date_input("Seleccione una fecha", df["Fecha Creacion"].min())
    
    año = d.strftime("%Y")
    mes = d.strftime("%m")
    mes_anterior = int(mes)-1 
    

    if mes_anterior < 10:
        mes_anterior = '0'+str(mes_anterior)
    else:
        mes_anterior = str(mes_anterior)

    if mes == '01':
        mes_anterior = '12'
        año = int(año) - 1
        año = str(año)   
        print("Hola", año)

    
  

    



    condicion =  df["Año"] == año
    condicion2 = df["Mes"] == mes
    condicion3 = df["Mes"] == mes_anterior
    atime = df[df["Dias"]<=0]
    indicador1 = df[condicion & condicion2]
    indicador2 = atime[condicion & condicion2]
    indicador3 = atime[condicion & condicion3]
    indicador4 = df[condicion & condicion3]

    

    #a = int(month) - 1
    #print(str(a))

#Metrics    
    pedidos = indicador1["Orden de Trabajo"].count()
    pedidos_mes_anterior = indicador4["Orden de Trabajo"].count()
    entregas_a_tiempo = indicador2["Dias"].count()
    entregas_a_tiempo_periodo_anterior = indicador3["Dias"].count()
    variacion = round(((entregas_a_tiempo - entregas_a_tiempo_periodo_anterior) / entregas_a_tiempo_periodo_anterior)  * 100)
    indicador_mes_anterior = (entregas_a_tiempo_periodo_anterior/pedidos_mes_anterior)*100
    print(entregas_a_tiempo_periodo_anterior)
 
    
    col1.metric("Cantidad de pedidos", str(pedidos), str(pedidos - pedidos_mes_anterior))
    col2.metric("Entregas a tiempo", str(entregas_a_tiempo), str(entregas_a_tiempo - entregas_a_tiempo_periodo_anterior))
    col3.metric("Δ% Entregas a tiempo - mes anterior", str(variacion) + " %")
    col1.metric("% de pedidos entregados a tiempo", str(round((entregas_a_tiempo/pedidos)*100)) + '%', str(round(indicador_mes_anterior)) + '%', delta_color="off")

#Chart
    st.title("Gráficos")
    st.subheader("Cantidad de pedidos")
    chart_data2 = df.pivot_table(index='Mes', columns="Año",values='Orden de Trabajo', aggfunc="count").reset_index()
    line_chart2(chart_data2)
    

#Chart - atrasos


    st.subheader("Entregas Atrasadas")
    atrasos = df[df["Dias"] <= 0]
    chart_data = atrasos.pivot_table(index="Mes" , columns='Año', values="Dias",  aggfunc="count").reset_index()
    line_chart(chart_data)

    #RNN Sidebar
    st.sidebar.header("Características del pedido - RNN")


 

     #
    def user_input_features():
        ancho_pedido = st.sidebar.slider('Ancho Pedido', 660, 2480)
        borde_pedido = st.sidebar.slider('Borde Pedido', 790, 2000)
        diametro_pedido = st.sidebar.slider('Diametro Pedido', 0, 19)
        cantidad_pedido = st.sidebar.slider('Cantidad Pedido', 1, 31)
        dias_planificacion = st.sidebar.slider('dias_planificacion', 1, 17)
        materia_prima = st.sidebar.number_input('materia_prima', 0, 1)
        # telar = st.sidebar.slider('telar', 1, 4, 2)
        # telar = st.sidebar.multiselect('Seleccione el número de telar', [1, 2, 3, 4])
        # materia_prima = st.sidebar.slider('materia_prima', 0, 21, 15)
        
    
        data = {
            'ancho_pedido': ancho_pedido,
            'borde_pedido': borde_pedido,
            'diametro_pedido': diametro_pedido,
            'cantidad_pedido': cantidad_pedido,
            'dias_planificacion': dias_planificacion,
            'materia_prima': materia_prima
          
        }
        features = pd.DataFrame(data, index=[0])
        return features

    st.header('Red Neuronal')
    st.subheader('Características a predecir')

    #RNN home
    df = user_input_features()
    st.write(df)

    # data = pd.read_csv("Regression.csv")
    # X = data.drop(["Dias"], axis=1).values
    # y = data["Dias"].values
    # n_cols = X.shape[1]



    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # print(X_train)

    # input_scaler = StandardScaler()
    # output_scaler = StandardScaler()
    # input_scaler.fit(X_train)
    # y_train = y_train.reshape(len(y_train), 1)
    # output_scaler.fit(y_train)

    # df = input_scaler.transform(df)
   
    #load json and create model
    json_file = open('clas.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights("clas.hdf5")
    # Compilar modelo cargado y listo para usar.
    loaded_model.compile(loss='binary_crossentropy',
                    optimizer= 'adam',
                    metrics=['accuracy'])
    
    
    


    prediction = loaded_model.predict(df)
    # prediction = output_scaler.inverse_transform(prediction)
    #prediction_proba = model.predict_proba(df)

    #st.subheader('Class labels and their corresponding index number')
    #st.write(iris.target_names)
    
    st.subheader('Resultado de la predicción')

    st.write(prediction)

    #st.subheader('Prediction Probability')
    #st.write(prediction_proba)

    




else:
    col1.metric("Cantidad de pedidos", "-", "-")
    col2.metric("Entregas a tiempo", "-", "-")
    col3.metric("Δ% Entregas a tiempo - mes anterior", "-", "-")
    col1.metric("% de pedidos entregados a tiempo", "-", "-", delta_color="off")



