import os
import sys
import pandas as pd
import joblib

# Intenta importar swat para CAS. Si falla, asumimos entorno local.
try:
    import swat
    swat_available = True
except ImportError:
    swat_available = False

def load_data():
    """Carga los datos de HMEQ desde CAS (si disponible) o CSV local."""
    # Prioridad: CAS en SAS Studio
    if swat_available:
        try:
            # Conexión a CAS (usa env vars o .authinfo para credenciales)
            conn = swat.CAS()  # Conexión a CAS por defecto
            # Carga la tabla CAS a un DataFrame pandas
            castbl = conn.CASTable("HMEQ", caslib="Public")
            df = castbl.to_frame()
            print("Datos cargados desde CAS Public.HMEQ. Filas:", len(df))
            conn.close()  # Cierra la sesión CAS
            return df
        except Exception as e:
            print("Aviso: No se pudo conectar a CAS. Se usará CSV local. Detalle:", e)
    # Modo local: leer CSV
    # Determina ruta del CSV desde argumento o nombre por defecto
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if not csv_path:
        csv_path = "hmeq.csv"
    # Verifica existencia
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV de datos: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Datos cargados desde CSV '{csv_path}'. Filas: {len(df)}")
    return df

def prepare_data(df):
    """Separa la matriz de características X y el vector objetivo y, asegurando tipos correctos."""
    # Asegura que la columna BAD exista
    if 'BAD' not in df.columns:
        raise KeyError("La columna 'BAD' (objetivo) no está presente en los datos.")
    # Separar características y objetivo
    X = df.drop('BAD', axis=1)
    y = df['BAD']
    # Convierte y a entero (por si está como float u objeto)
    y = y.astype(int)
    # Asegura tipo adecuado para categoricas (object) y numericas (float)
    # (Esto suele ser automático según lectura, pero hacemos explícito:)
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].astype(float)  # como float
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str)    # como string (categorías)
    return X, y

def build_preprocessor(X):
    """Construye el ColumnTransformer para preprocesamiento (imputación, encoding)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Identifica columnas numéricas vs categóricas
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    # Definir transformadores
    numeric_transformer = SimpleImputer(strategy='mean')  # imputación por media
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # imputación por moda
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # one-hot encoding
    ])
    # Crear ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

def train_and_select_model(X, y, preprocessor):
    """Entrena varios modelos y selecciona el mejor según AUC. Devuelve el mejor pipeline y métricas."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    results = {}  # diccionario para almacenar AUC promedio de cada modelo
    # Usaremos StratifiedKFold para reproducibilidad (aunque cross_val_score lo haría automáticamente)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        # Construir pipeline de preprocesamiento + modelo
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('preproc', preprocessor),
            ('clf', model)
        ])
        # Evaluar con validación cruzada (AUC)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        mean_auc = scores.mean()
        std_auc = scores.std()
        results[name] = mean_auc
        print(f"{name}: AUC = {mean_auc:.4f} (±{std_auc:.4f})")
    # Seleccionar mejor modelo por AUC promedio
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_auc = results[best_model_name]
    print(f"Mejor modelo seleccionado: {best_model_name} con AUC = {best_auc:.4f}")
    # Entrenar pipeline final con el mejor modelo usando todos los datos
    from sklearn.pipeline import Pipeline
    best_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', best_model)
    ])
    best_pipeline.fit(X, y)
    return best_pipeline, best_model_name, results

def save_artifacts(pipeline, best_model_name, results):
    """Guarda pipeline entrenado, metadata, scripts de scoring y requerimientos."""
    # 1. Modelo serializado
    joblib.dump(pipeline, "pipeline.pkl")
    # 2. Metadata JSON
    import json
    metadata = {
        "best_model": best_model_name,
        "metrics_used": "AUC",
        "metrics": {m: round(v, 4) for m, v in results.items()},
        "n_features": pipeline.named_steps['preproc'].transformers_[0][2].__len__() + 
                      pipeline.named_steps['preproc'].transformers_[1][2].__len__(),
        "features_numeric": pipeline.named_steps['preproc'].transformers_[0][2],
        "features_categorical": pipeline.named_steps['preproc'].transformers_[1][2],
        "model_params": pipeline.named_steps['clf'].get_params()
    }
    # Agregar timestamp
    from datetime import datetime
    metadata["train_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    # 3. Score.py script
    score_script = f'''
import sys
import pandas as pd
import joblib

# Cargar el modelo entrenado
pipeline = joblib.load("pipeline.pkl")

def score_dataframe(df):
    """Devuelve las puntuaciones de probabilidad de default para un DataFrame de entradas."""
    preds = pipeline.predict_proba(df)[:, 1]  # probabilidad de clase 1 (default)
    return preds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python score.py <archivo_csv_entrada> [archivo_csv_salida]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    data = pd.read_csv(input_path)
    # Predecir probabilidades
    data["Score"] = score_dataframe(data)
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Resultados guardados en {{output_path}}")
    else:
        # Muestra las primeras filas por consola si no se especifica salida
        print(data.head())
'''
    with open("score.py", "w") as f:
        f.write(score_script.strip() + "\n")
    # 4. Requirements.txt
    reqs = ["pandas", "numpy", "scikit-learn", "joblib"]
    # Incluye swat si estaba instalado (posible SAS)
    if swat_available:
        reqs.append("swat")
    with open("requirements.txt", "w") as f:
        for pkg in reqs:
            f.write(pkg + "\n")
    # 5. Metrics en CSV y JSON
    # CSV
    with open("metrics.csv", "w") as f:
        f.write("Model,AUC\n")
        for model, auc in results.items():
            f.write(f"{model},{auc:.4f}\n")
        f.write(f"BestModel,{best_model_name}\n")
    # JSON (podría reutilizar 'results', pero hacemos formato separado si se desea)
    metrics = {
        "model_auc": {m: round(v,4) for m,v in results.items()},
        "best_model": best_model_name,
        "best_model_auc": round(results[best_model_name], 4)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

# Punto de entrada principal
if __name__ == "__main__":
    df = load_data()
    X, y = prepare_data(df)
    preprocessor = build_preprocessor(X)
    pipeline, best_model_name, results = train_and_select_model(X, y, preprocessor)
    save_artifacts(pipeline, best_model_name, results)
    print("Entrenamiento completado. Archivos de salida generados.")
