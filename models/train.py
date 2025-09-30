import os
import sys
import json
import pandas as pd
import joblib

# Intenta importar swat para CAS. Si falla, asumimos entorno local.
try:
    import swat
    swat_available = True
except ImportError:
    swat_available = False


def _onehot_dense_kwargs():
    """
    Devuelve kwargs compatibles con la versión instalada de scikit-learn
    para forzar salida densa en OneHotEncoder.
    - >=1.2:  sparse_output=False
    - <=1.1:  sparse=False
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
        # Probar si acepta 'sparse_output'
        _ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        # Versión más vieja
        return {"sparse": False, "handle_unknown": "ignore"}


def load_data():
    """Carga HMEQ desde CAS (si disponible) o CSV local."""
    if swat_available:
        try:
            host = os.getenv("CAS_HOST", "localhost")
            port = int(os.getenv("CAS_PORT", "5570"))
            protocol = os.getenv("CAS_PROTOCOL", "http")  # "http" o "cas"
            user = os.getenv("CAS_USERNAME")
            pwd = os.getenv("CAS_PASSWORD")

            if user and pwd:
                conn = swat.CAS(host, port, user, pwd, protocol=protocol)
            else:
                conn = swat.CAS(host, port, protocol=protocol)

            castbl = conn.CASTable("HMEQ", caslib="Public")
            df = castbl.to_frame()
            print("Datos cargados desde CAS Public.HMEQ. Filas:", len(df), flush=True)
            conn.close()
            return df
        except Exception as e:
            print("Aviso: No se pudo conectar a CAS. Se usará CSV local. Detalle:", e, flush=True)

    # CSV local: path por argumento o hmeq.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "hmeq.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV de datos: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Datos cargados desde CSV '{csv_path}'. Filas: {len(df)}", flush=True)
    return df


def prepare_data(df):
    """Separa X,y; normaliza 'BAD' a binario int; tipifica columnas."""
    cols_upper = {c: c.upper() for c in df.columns}
    df = df.rename(columns=cols_upper)

    if "BAD" not in df.columns:
        raise KeyError("La columna 'BAD' (objetivo) no está presente en los datos.")

    # Asegurar binario 0/1
    y = df["BAD"].copy()
    # Si no es 0/1, mapear >0 a 1
    if not set(pd.Series(y).dropna().unique()).issubset({0, 1}):
        y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
    else:
        y = y.astype(int)

    X = df.drop(columns=["BAD"]).copy()

    # Tipos explícitos
    for col in X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    return X, y


def build_preprocessor(X):
    """ColumnTransformer (imputación + OHE denso)."""
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_features = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**_onehot_dense_kwargs()))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor


def train_and_select_model(X, y, preprocessor):
    """Entrena varios modelos y selecciona el mejor por AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipe = Pipeline([
            ("preproc", preprocessor),
            ("clf", model)
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        mean_auc, std_auc = scores.mean(), scores.std()
        results[name] = float(mean_auc)
        print(f"{name}: AUC = {mean_auc:.4f} (±{std_auc:.4f})", flush=True)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"Mejor modelo seleccionado: {best_model_name} con AUC = {results[best_model_name]:.4f}", flush=True)

    best_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", best_model)
    ])
    best_pipeline.fit(X, y)
    return best_pipeline, best_model_name, results


def save_artifacts(pipeline, best_model_name, results):
    """Guarda pipeline, metadatos, score.py, requirements, y métricas."""
    # Modelo
    joblib.dump(pipeline, "pipeline.pkl")

    # Features después del fit
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        preproc = pipeline.named_steps["preproc"]
        feature_names = []
        try:
            feature_names = preproc.get_feature_names_out().tolist()
        except Exception:
            # Fallback (sin nombres expandidos)
            feature_names = []
    except Exception:
        feature_names = []

    # Metadata
    metadata = {
        "best_model": best_model_name,
        "metrics_used": "AUC",
        "metrics": {m: round(v, 4) for m, v in results.items()},
        "n_features_after_preprocess": len(feature_names) if feature_names else None,
        "feature_names": feature_names if feature_names else None,
        "model_params": pipeline.named_steps["clf"].get_params()
    }
    from datetime import datetime
    metadata["train_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # Score script (usa el pipeline entrenado)
    score_script = r'''import sys
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")

def score_dataframe(df: pd.DataFrame):
    """Devuelve probas de clase 1."""
    return pipeline.predict_proba(df)[:, 1]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python score.py <archivo_csv_entrada> [archivo_csv_salida]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    data = pd.read_csv(input_path)
    data["Score"] = score_dataframe(data)
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Resultados guardados en {output_path}")
    else:
        print(data.head())
'''
    with open("score.py", "w", encoding="utf-8") as f:
        f.write(score_script)

    # requirements.txt (añade swat si está disponible)
    reqs = ["pandas", "numpy", "scikit-learn", "joblib"]
    if swat_available:
        reqs.append("swat")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reqs) + "\n")

    # Métricas en CSV y JSON
    with open("metrics.csv", "w", encoding="utf-8") as f:
        f.write("Model,AUC\n")
        for model, auc in results.items():
            f.write(f"{model},{auc:.4f}\n")
        f.write(f"BestModel,{best_model_name}\n")

    metrics = {
        "model_auc": {m: round(v, 4) for m, v in results.items()},
        "best_model": best_model_name,
        "best_model_auc": round(results[best_model_name], 4)
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_data(df)
    preprocessor = build_preprocessor(X)
    pipeline, best_model_name, results = train_and_select_model(X, y, preprocessor)
    save_artifacts(pipeline, best_model_name, results)
    print("Entrenamiento completado. Archivos de salida generados.", flush=True)
