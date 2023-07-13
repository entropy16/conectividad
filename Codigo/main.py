import numpy as np
import matplotlib.pyplot as plt

# Ruta completa del archivo CSV
ruta_archivo = '/home/netropy/Documentos/Conectividad/Databases/Evoked Potential/Filtered/s01_ex01_s01.csv'

# Cargar el archivo CSV
data = np.genfromtxt(ruta_archivo, delimiter=',', dtype=float, skip_header=1)

def leer_data(data: np.ndarray):
    """ Retorna los indices y los electrodos de un EEG (data)

    Args:
        data (np.ndarray): Datos de un EEG

    Returns:
        tuple (np.array, np.array): 
            (indices_muestra, electrodos)
            electrodos es un arreglo de arreglos, cada uno representa los datos de un electrodo
    """
    indice_muestra = data[:, 0]

    electrodos = np.empty((data.shape[0], data.shape[1] - 1))
    for i in range(data.shape[1] - 1):
        electrodos[:,i] = data[:, i + 1]

    return indice_muestra, electrodos

def calc_gfp(electrodes: np.ndarray):
    """ Retorna la GFP de un grupo de electrodos

    Args:
        electrodes (np.ndarray): 
            Contiene los valores de un EEG, separado por electrodos

    Returns:
        GFP: Potencia de campo global de un grupo de electrodos
    """
    N = electrodes.shape[1]
    v_mean = np.mean(electrodes, axis=1)
    GFP = np.zeros(electrodes.shape[0])
    
    for i in range(0,N):
        GFP = GFP+(electrodes[:,i]-v_mean)**2
    GFP = np.sqrt(GFP/N)
    
    return GFP

def get_microstates_sequences(gfp: np.ndarray, percentage: float):
    """ Retorna una tupla con los indices y los valores de la GFP que superan el
        umbral, separados por microestado

    Args:
        gfp (np.ndarray): Potencia de campo global GFP
        percentage (float): Porcentaje de umbral

    Returns:
        tuple (np.array, np.array): 
            (indices_sequencia, valores_sequencia) 
            Ambos arreglos pueden ser NO homogeneos. Se crean de tipo object
    """
    max_value = np.max(gfp)
    indices_mayores = np.where(gfp > max_value*percentage)
    
    sequence_indexes = []
    sequence = np.empty((0,), dtype=int)

    for i in range(len(indices_mayores[0])):
        if i == 0 or indices_mayores[0][i] == indices_mayores[0][i-1] + 1:
            sequence = np.append(sequence, indices_mayores[0][i])
        else:
            if len(sequence) >= 12:
                sequence_indexes.append(sequence)
            sequence = np.array([indices_mayores[0][i]], dtype=int)
            
    if len(sequence) >= 12:
        sequence_indexes.append(sequence)

    sequence_indexes = np.array(sequence_indexes, dtype=object)
    sequence_values = np.array(
        [gfp[sequence_indexes[i]] for i in range(len(sequence_indexes))],
        dtype=object
    )

    return sequence_indexes, sequence_values

def get_microstates_samples(gfp: np.ndarray, percentage: float):
    """ Retorna una tupla con los indices y los valores de las muestras del 
        gfp que superan el umbral, sin separar por microestado

    Args:
        gfp (np.ndarray): Potencia de campo global GFP
        percentage (float): Porcentaje de umbral

    Returns:
        tuple (np.array, np.array): (indices_muestras, valores_muestras)
    """
    max_value = np.max(gfp)
    indices_mayores = np.where(gfp > max_value*percentage)
    
    sequences = []
    sequence = []

    for i in range(len(indices_mayores[0])):
        if i == 0 or indices_mayores[0][i] == indices_mayores[0][i-1] + 1:
            sequence.append(indices_mayores[0][i])
        else:
            if len(sequence) >= 12:
                sequences += sequence
            sequence = [indices_mayores[0][i]]

    if len(sequence) >= 12:
        sequences += sequence
    
    return np.array(sequences), gfp[sequences]

def get_electrodes_value(indices: np.array, electrodes: np.array):
    """ Retorna un array con los valores de cada electrodo para los indices dados.

    Args:
        indices (np.array): Indices separados por microestado
        electrodes (np.array): Array con los valores de los electrodos

    Returns:
        np.array: 
            Array que contiene los valores de los electrodos por cada muestra
            que supera el umbral. Si los indices están separados por microestado,
            el array también lo estará. En el caso de que NO estén
            separados por microestado, al array contiene todas las muestras independientes
    """
    electrodes_values = []
    
    for indice in indices:
        electrodes_values.append(electrodes[indice])
    
    return np.array(electrodes_values, dtype=object)
    
    
# Obtener los datos de los electrodos
indice_muestra, electrodos = leer_data(data)
# Calcular la gfp para los electrodos
gfp = calc_gfp(electrodos)

# Obtener los microestados (secuencias en la gfp con 12 o más elementos)
ms_sequence_indexes, ms_sequence_values = get_microstates_sequences(gfp, percentage=0.4)
print("microstates_indexes: (", len(ms_sequence_indexes), ")\n", ms_sequence_indexes)
print("microstates_values: (", len(ms_sequence_values), ")\n", ms_sequence_values)
ms_electrodes_values = get_electrodes_value(ms_sequence_indexes, electrodos)
print("valores de los electrodos:\n", ms_electrodes_values)
print()

# Obtener las muestras (muestras individuales en la gfp que pertenecen a una secuencia de 12 o más elementos)
sample_indexes, sample_values = get_microstates_samples(gfp, percentage=0.4)
print("sample_indexes: (", len(sample_indexes), ")\n", sample_indexes)
print("sample_values: (", len(sample_values), ")\n", sample_values)
sample_electrodes_values = get_electrodes_value(sample_indexes, electrodos)
print("valores de los electrodos:\n", sample_electrodes_values)
print()


plt.plot(indice_muestra, gfp, label="gfp")
plt.axhline(y=16, color='red', linestyle='--')

# Agregar etiquetas y leyenda
plt.xlabel('Índice de muestra')
plt.ylabel('Amplitud')
plt.legend()

# Mostrar el gráfico
plt.show()

