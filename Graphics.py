import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# URL del dataset del Titanic
url_titanic = "https://raw.githubusercontent.com/cvazquezlos/machine-learning-prework/main/04-matplotlib/assets/titanic_train.csv"

# --- 1. Crear el DataFrame ---
try:
    df_titanic = pd.read_csv(url_titanic)
    print("DataFrame de Titanic cargado exitosamente. Se muestran las primeras 5 filas:\n")
    print(df_titanic[['PassengerId', 'Age', 'Fare', 'Survived']].head())
    print("\n" + "="*50 + "\n")
    
except Exception as e:
    print(f"Error al cargar el DataFrame: {e}")
    # Detener la ejecución si la carga falla
    exit()

# --- 2. Mostrar las Distribuciones de Edad y Billete (Fare) ---

# Configuración de los subplots: 1 fila, 2 columnas
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plt.suptitle('Distribución de Variables en el Dataset de Titanic', fontsize=16)

# Distribución 1: Edad (Age)
# Usamos .dropna() para excluir los valores nulos (NaN) de 'Age' antes de graficar,
# ya que el histograma no puede procesarlos.
sns.histplot(
    df_titanic['Age'].dropna(), 
    kde=True, # Muestra la estimación de densidad del kernel (curva suave)
    bins=30, 
    ax=axes[0],
    color='skyblue'
)
axes[0].set_title('Distribución de la Edad (Age)')
axes[0].set_xlabel('Edad (Años)')
axes[0].set_ylabel('Frecuencia')
axes[0].grid(axis='y', alpha=0.5)

# Distribución 2: Importe del Billete (Fare)
# Esta columna contiene muchos valores cerca de cero y una cola larga (asimetría positiva).
sns.histplot(
    df_titanic['Fare'], 
    kde=True, 
    bins=40, 
    ax=axes[1],
    color='lightcoral'
)
axes[1].set_title('Distribución del Importe del Billete (Fare)')
axes[1].set_xlabel('Importe del Billete (Fare)')
axes[1].set_ylabel('Frecuencia')
axes[1].grid(axis='y', alpha=0.5)

# Ajustar el espaciado entre subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
