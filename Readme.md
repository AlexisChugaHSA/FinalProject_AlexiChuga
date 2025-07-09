# Juego de la Vida de Conway

![unit: Departamento de Ingeniería de Sistemas y Computación](https://img.shields.io/badge/course-Departamento%20de%20Ingenier%C3%ADa%20de%20Sistemas%20y%20Computaci%C3%B3n-blue?logo=coursera)
![institution: Universidad Católica del Norte](https://img.shields.io/badge/institution-Universidad%20Cat%C3%B3lica%20del%20Norte-blue?logo=google-scholar)

**Autor:** *Hernán Alexis Chugá Portilla*  
**Año:** 2025

## 📄 Descripción del Proyecto

Este proyecto implementa una versión optimizada y extensible del **Juego de la Vida de Conway** utilizando Python.  
El juego se basa en un autómata celular que simula la evolución de células vivas y muertas en una cuadrícula bidimensional,  
siguiendo reglas sencillas de nacimiento, supervivencia y muerte.

### Reglas del Juego de la Vida

El **Juego de la Vida** fue creado por John Conway en 1970. Sus reglas definen cómo cada célula de la cuadrícula evoluciona  
de una generación a la siguiente en función de sus vecinos:

- **Supervivencia:** Una célula viva con 2 o 3 vecinos vivos continúa viva.
- **Muerte por soledad:** Una célula viva con menos de 2 vecinos vivos muere.
- **Muerte por sobrepoblación:** Una célula viva con más de 3 vecinos vivos muere.
- **Nacimiento:** Una célula muerta con exactamente 3 vecinos vivos se convierte en una célula viva.

Estas reglas, aunque simples, pueden dar lugar a patrones extremadamente complejos y fascinantes, como osciladores (*blinkers*),  
naves espaciales (*gliders*) y estructuras estables.

###  Funcionalidades

El proyecto incluye:

- Representación y manipulación de la matriz del juego con **NumPy**.
- Evolución paralela del juego usando **multiprocesamiento** o **multi-threading** para alto rendimiento.
- Visualización en tiempo real con **Streamlit** y **Matplotlib**.
- Interfaz web interactiva para pausar, reanudar y reiniciar la simulación.
- Registro de estadísticas en **SQLite** con **SQLAlchemy**.
- Detección automática de patrones especiales como *gliders* y *blinkers*.
- Detención automática de la simulación en estados estables.

Todo está organizado para ser modular, claro y fácil de mantener.


## ⚙️ Instalación

Este proyecto usa **Python 3.11** y un entorno reproducible administrado con **conda**.  
Puedes instalar todas las dependencias fácilmente siguiendo estos pasos:

1. Clonar este repositorio:
```bash
  git clone https://github.com/AlexisChugaHSA/FinalProject_AlexiChuga.git
  cd FinalProject_AlexiChuga
```
2. Crear el entorno con conda usando el archivo environment.yml:
```bash
  conda env create -f environment.yml
```
3. Activar el entorno:
```bash
  conda activate FP_env
```
## 🚀 Instrucciones de uso

### ⚙️ 1. Configurar el estado inicial del juego

Para definir o modificar el estado inicial de la grilla:

1. Dentro del proyecto ingresar a la ruta:
  scripts/game_of_life

2. Luego ingresar al archivo save_initial_state.py y en la línea **14**, 
se encuentra la variable `initial_state`.  
Se debe modificar a la nueva configuración deseada, el estado inicial actual es:  

```python
initial_state = np.array([
    [1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0]
], dtype=np.uint8)
```
3. Luego se debe guardar el archivo y ejecutarlo:
```python
python save_initial_state.py
```
Esto generará el archivo initial.pkl dentro del directorio data/, el cual será 
leído automáticamente por el archivo principal del proyecto.

## License

This project is open-sourced software licensed under the [Apache license](LICENSE).

![License](https://img.shields.io/github/license/godiecl/template)