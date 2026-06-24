from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los archivos del modelo
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_final.pkl")
RUTA_COLUMNAS = os.path.join(BASE_DIR, "columnas_modelo.pkl")

# Carga del pipeline final y columnas utilizadas por el modelo
modelo = joblib.load(RUTA_MODELO)
columnas_modelo = joblib.load(RUTA_COLUMNAS)


def convertir_float(valor, nombre_campo):
    """
    Convierte un valor recibido desde el formulario a tipo float.
    También permite que el usuario escriba coma decimal, por ejemplo 70,5.
    """
    if valor is None or valor == "":
        raise ValueError(f"El campo {nombre_campo} es obligatorio.")

    try:
        valor = valor.replace(",", ".")
        return float(valor)
    except ValueError:
        raise ValueError(f"El campo {nombre_campo} debe ser numérico.")


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    error = None

    # Valores iniciales del formulario
    valores = {
        "age": "",
        "gender": "",
        "height_cm": "",
        "weight_kg": "",
        "diastolic": "",
        "systolic": "",
        "gripForce": "",
        "sit_and_bend_forward_cm": "",
        "sit_ups_counts": "",
        "broad_jump_cm": "",
        "class": ""
    }

    if request.method == "POST":
        try:
            # Recuperamos los valores del formulario
            valores["age"] = request.form.get("age", "")
            valores["gender"] = request.form.get("gender", "")
            valores["height_cm"] = request.form.get("height_cm", "")
            valores["weight_kg"] = request.form.get("weight_kg", "")
            valores["diastolic"] = request.form.get("diastolic", "")
            valores["systolic"] = request.form.get("systolic", "")
            valores["gripForce"] = request.form.get("gripForce", "")
            valores["sit_and_bend_forward_cm"] = request.form.get("sit_and_bend_forward_cm", "")
            valores["sit_ups_counts"] = request.form.get("sit_ups_counts", "")
            valores["broad_jump_cm"] = request.form.get("broad_jump_cm", "")
            valores["class"] = request.form.get("class", "")

            # Conversión de datos numéricos
            age = convertir_float(valores["age"], "Edad")
            height_cm = convertir_float(valores["height_cm"], "Altura")
            weight_kg = convertir_float(valores["weight_kg"], "Peso")
            diastolic = convertir_float(valores["diastolic"], "Presión diastólica")
            systolic = convertir_float(valores["systolic"], "Presión sistólica")
            gripForce = convertir_float(valores["gripForce"], "Fuerza de agarre")
            sit_and_bend_forward_cm = convertir_float(
                valores["sit_and_bend_forward_cm"],
                "Flexión hacia adelante"
            )
            sit_ups_counts = convertir_float(
                valores["sit_ups_counts"],
                "Cantidad de abdominales"
            )
            broad_jump_cm = convertir_float(
                valores["broad_jump_cm"],
                "Salto largo"
            )

            # Datos categóricos
            gender = valores["gender"]
            class_value = valores["class"]

            # Validaciones de variables categóricas
            if gender not in ["M", "F"]:
                raise ValueError("Selecciona un género válido.")

            if class_value not in ["A", "B", "C", "D"]:
                raise ValueError("Selecciona una clase de rendimiento válida.")

            # Validaciones de rangos
            if age < 10 or age > 100:
                raise ValueError("La edad debe estar entre 10 y 100 años.")

            if height_cm < 100 or height_cm > 230:
                raise ValueError("La altura debe ingresarse en centímetros. Ejemplo: 175, no 1.75.")

            if weight_kg < 20 or weight_kg > 250:
                raise ValueError("El peso debe estar entre 20 y 250 kg.")

            if diastolic < 40 or diastolic > 140:
                raise ValueError("La presión diastólica debe estar entre 40 y 140. Ejemplo: 80.")

            if systolic < 70 or systolic > 240:
                raise ValueError("La presión sistólica debe estar entre 70 y 240. Ejemplo: 120.")

            if systolic <= diastolic:
                raise ValueError(
                    "La presión sistólica debe ser mayor que la diastólica. "
                    "Ejemplo correcto: sistólica 120 y diastólica 80."
                )

            if gripForce < 0 or gripForce > 100:
                raise ValueError("La fuerza de agarre debe estar entre 0 y 100.")

            if sit_and_bend_forward_cm < -30 or sit_and_bend_forward_cm > 100:
                raise ValueError("La flexión hacia adelante debe estar entre -30 y 100 cm.")

            if sit_ups_counts < 0 or sit_ups_counts > 100:
                raise ValueError("La cantidad de abdominales debe estar entre 0 y 100.")

            if broad_jump_cm < 0 or broad_jump_cm > 350:
                raise ValueError("El salto largo debe estar entre 0 y 350 cm.")

            # Variables derivadas utilizadas en la libreta
            height_m = height_cm / 100
            BMI = weight_kg / (height_m ** 2)
            pulse_pressure = systolic - diastolic
            grip_per_weight = gripForce / weight_kg
            jump_per_height = broad_jump_cm / height_cm
            situps_per_weight = sit_ups_counts / weight_kg

            # Diccionario con variables originales y derivadas
            datos = {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "diastolic": diastolic,
                "systolic": systolic,
                "gripForce": gripForce,
                "sit and bend forward_cm": sit_and_bend_forward_cm,
                "sit-ups counts": sit_ups_counts,
                "broad jump_cm": broad_jump_cm,
                "class": class_value,
                "BMI": BMI,
                "pulse_pressure": pulse_pressure,
                "grip_per_weight": grip_per_weight,
                "jump_per_height": jump_per_height,
                "situps_per_weight": situps_per_weight
            }

            # Construimos una fila con las columnas exactas que espera el modelo
            fila = {}

            for columna in columnas_modelo:
                fila[columna] = datos.get(columna, 0)

            entrada = pd.DataFrame([fila], columns=columnas_modelo)

            # Predicción
            prediccion = modelo.predict(entrada)[0]
            resultado = round(float(prediccion), 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        resultado=resultado,
        error=error,
        valores=valores
    )


if __name__ == "__main__":
    app.run(debug=True)