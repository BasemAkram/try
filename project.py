import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score , silhouette_score
st.set_page_config(layout="wide")
st.title("Machine Learning GUI Project")

# ---- Initialize Session State ----
for key in ["last_uploaded", "df_original", "df", "df_sample", "model_type", "target_column",
            "df_before_preview", "df_before_encoding", "df_before_imputation",
            "df_before_transformation", "df_before_selection", "df_before_model"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---- Step 1: Upload Data ----
st.subheader("Step 1: Upload Your Data")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ---- Step 1.5: Select Model Type ----
st.subheader("Step 1.5: Select Model Type")
st.session_state.model_type = st.selectbox(
    "Choose the type of model you want to work with",
    ["Classification", "Regression", "Clustering"]
)

# ---- Process Uploaded File ----
if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state.df_original = df.copy()
        st.session_state.df = df.copy()
        st.session_state.df_sample = df.copy()  # ✅ إنشاء df_sample مباشرة
        st.session_state.last_uploaded = uploaded_file.name

    df_sample = st.session_state.df_sample

    # ---- Target Column Selection ----
    if st.session_state.model_type in ["Classification", "Regression"]:
        st.session_state.target_column = st.selectbox(
            "Select Target Column",
            df_sample.columns
        )

    st.subheader("Step 2: Preview Data & Sample Selection")
    row_limit = st.slider(
        "Select number of rows to use in the analysis",
        5, len(df_sample),
        min(20, len(df_sample))
    )

    # ---- Preview Data ----
    st.dataframe(df_sample.head(row_limit))

    # ---- Column Type Labels ----
    cat_cols = [col for col in df_sample.columns if not pd.api.types.is_numeric_dtype(df_sample[col])]
    num_cols = [col for col in df_sample.columns if pd.api.types.is_numeric_dtype(df_sample[col])]
    col_type_labels = {col: f"{col} ({'numeric' if col in num_cols else 'categorical'})" for col in df_sample.columns}

    # ---- Step 3: Preprocessing ----
    st.subheader("Step 3: Apply Preprocessing")
    with st.expander("Missing Value Imputation", expanded=True):
        impute_map = {}
        for col in df_sample.columns:
            key_name = f"imp_{col}"
            if col in num_cols:
                impute_map[col] = st.selectbox(
                    col_type_labels.get(col, col),
                    ["No imputation", "Simple Mean", "Simple Median", "Simple Most Frequent", "KNN", "Iterative"],
                    key=key_name
                )
            else:
                impute_map[col] = st.selectbox(
                    col_type_labels.get(col, col),
                    ["No imputation", "Simple Most Frequent"],
                    key=key_name
                )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Imputation", type="primary", key="apply_imp"):
                # حفظ نسخة قبل الإيمبيوتيشن
                st.session_state.df_before_imputation = df_sample.copy()
                df_copy = df_sample.copy()
                for col, imp_type in impute_map.items():
                    if imp_type != "No imputation" and col in df_copy.columns:
                        try:
                            if imp_type.startswith("Simple"):
                                strategy = {"Simple Mean": "mean", "Simple Median": "median",
                                            "Simple Most Frequent": "most_frequent"}[imp_type]
                                imputer = SimpleImputer(strategy=strategy)
                                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                            elif imp_type == "KNN":
                                imputer = KNNImputer(n_neighbors=5)
                                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                            elif imp_type == "Iterative":
                                imputer = IterativeImputer()
                                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                        except Exception as e:
                            st.error(f"Error in imputation for {col}: {e}")
                st.session_state.df_sample = df_copy
                st.success("Imputation applied!")

        with col2:
            if st.button("Undo Imputation", key="undo_imp"):
                if "df_before_imputation" in st.session_state and st.session_state.df_before_imputation is not None:
                    st.session_state.df_sample = st.session_state.df_before_imputation.copy()
                    st.success("Imputation undone!")


    # ---------- Encoding Options ----------

    with st.expander("Encoding Options", expanded=True):
        if cat_cols:
            encoding_map = {}
            for col in cat_cols:
                encoding_map[col] = st.selectbox(
                    col_type_labels.get(col, col),
                    ["No encoding", "Label Encoding", "One-Hot Encoding"],
                    key=f"enc_{col}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Encoding", type="primary", key="apply_enc"):
                    # حفظ نسخة مستقلة للـ Encoding
                    st.session_state.df_before_encoding = st.session_state.df_sample.copy()

                    df_copy = st.session_state.df_sample.copy()
                    for col, enc_type in encoding_map.items():
                        if enc_type == "Label Encoding":
                            le = LabelEncoder()
                            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                        elif enc_type == "One-Hot Encoding":
                            df_copy = pd.get_dummies(df_copy, columns=[col], drop_first=False)

                    st.session_state.df_sample = df_copy
                    st.rerun()

            with col2:
                if st.button("Undo Encoding", key="undo_enc"):
                    if "df_before_encoding" in st.session_state:
                        st.session_state.df_sample = st.session_state.df_before_encoding.copy()
                        st.rerun()
        else:
            st.info("No categorical columns available for encoding.")

    # ---------- Scaling Options ----------

    with st.expander("Scaling Options", expanded=True):
        if num_cols:  # تأكد أن هناك أعمدة رقمية
            scaling_map = {}
            for col in num_cols:
                scaling_map[col] = st.selectbox(
                    col_type_labels.get(col, col),
                    ["No scaling", "StandardScaler", "MinMaxScaler"],
                    key=f"scale_{col}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Scaling", type="primary", key="apply_scale"):
                    # حفظ نسخة قبل التطبيق
                    st.session_state.df_before_scale = st.session_state.df_sample.copy()
                    df_copy = st.session_state.df_sample.copy()
                    for col, scale_type in scaling_map.items():
                        if scale_type == "StandardScaler":
                            scaler = StandardScaler()
                            df_copy[col] = scaler.fit_transform(df_copy[[col]])
                        elif scale_type == "MinMaxScaler":
                            scaler = MinMaxScaler()
                            df_copy[col] = scaler.fit_transform(df_copy[[col]])
                    st.session_state.df_sample = df_copy
                    st.rerun()

            with col2:
                if st.button("Undo Scaling", key="undo_scale"):
                    if "df_before_scale" in st.session_state:
                        st.session_state.df_sample = st.session_state.df_before_scale.copy()
                        st.rerun()
        else:
            st.info("No numeric columns available for scaling.")

    # ---------- Outlier Handling ----------
    with st.expander("Outlier Handling", expanded=True):
        if num_cols:  # تأكد أن هناك أعمدة رقمية
            outlier_map = {}
            for col in num_cols:
                outlier_map[col] = st.selectbox(
                    col_type_labels.get(col, col),
                    ["No Handling", "Winsorization"],
                    key=f"hand_{col}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Outlier Handling", type="primary", key="apply_outlier"):
                    # حفظ نسخة قبل التطبيق
                    st.session_state.df_before_outlier = st.session_state.df_sample.copy()
                    df_copy = st.session_state.df_sample.copy()

                    for col, outlier_type in outlier_map.items():
                        if outlier_type == "Winsorization":
                            try:
                                df_copy[col] = pd.Series(
                                    winsorize(df_copy[col], limits=[0.05, 0.05]),
                                    index=df_copy.index
                                )
                            except Exception as e:
                                st.error(f"Error in outlier handling for {col}: {e}")

                    st.session_state.df_sample = df_copy
                    st.rerun()

            with col2:
                if st.button("Undo Outlier Handling", key="undo_outlier"):
                    if "df_before_outlier" in st.session_state:
                        st.session_state.df_sample = st.session_state.df_before_outlier.copy()
                        st.rerun()
        else:
            st.info("No numeric columns available for outlier handling.")

    # ---------- Feature Transformation ----------

    with st.expander("Feature Transformation", expanded=True):
        if num_cols:
            transform_map = {}
            for col in num_cols:
                transform_map[col] = st.selectbox(
                    f"Transformation for {col}",
                    ["None", "Log (Natural)", "Log (Base 10)", "Log (Base 2)", "Inverse Log (log(1+x))",
                     "Power (Yeo-Johnson)", "Power (Box-Cox)", "Polynomial (Degree 2)"],
                    key=f"trans_{col}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Transformations", type="primary", key="apply_transform"):
                    # نسخة احتياطية خاصة بالتحويلات
                    st.session_state.df_before_transformation = st.session_state.df_sample.copy()
                    df_copy = st.session_state.df_sample.copy()

                    for col, t_type in transform_map.items():
                        try:
                            if t_type == "None":
                                continue
                            if t_type in ["Log (Natural)", "Log (Base 10)", "Log (Base 2)", "Power (Box-Cox)"] and (
                                    df_copy[col] <= 0).any():
                                st.warning(f"Skipping {col}: contains non-positive values.")
                                continue

                            if t_type == "Log (Natural)":
                                df_copy[col] = np.log(df_copy[col])
                            elif t_type == "Log (Base 10)":
                                df_copy[col] = np.log10(df_copy[col])
                            elif t_type == "Log (Base 2)":
                                df_copy[col] = np.log2(df_copy[col])
                            elif t_type == "Inverse Log (log(1+x))":
                                df_copy[col] = np.log1p(df_copy[col])
                            elif t_type == "Power (Yeo-Johnson)":
                                pt = PowerTransformer(method='yeo-johnson')
                                df_copy[col] = pt.fit_transform(df_copy[[col]]).flatten()
                            elif t_type == "Power (Box-Cox)":
                                pt = PowerTransformer(method='box-cox')
                                df_copy[col] = pt.fit_transform(df_copy[[col]]).flatten()
                            elif t_type == "Polynomial (Degree 2)":
                                pf = PolynomialFeatures(degree=2, include_bias=False)
                                poly_df = pf.fit_transform(df_copy[[col]])
                                df_copy[col + "_squared"] = poly_df[:, 1]

                        except Exception as e:
                            st.error(f"Error in transformation for {col}: {e}")

                    st.session_state.df_sample = df_copy
                    st.success("Transformations applied!")
                    st.rerun()

            with col2:
                if st.button("Undo Transformations", key="undo_transform"):
                    if "df_before_transformation" in st.session_state:
                        st.session_state.df_sample = st.session_state.df_before_transformation.copy()
                        st.success("Transformations undone!")
                        st.rerun()
        else:
            st.info("No numeric columns available for transformation.")

    # ---- Feature Selection & Dimensionality Reduction ----
    with st.expander("Feature Selection & Dimensionality Reduction", expanded=True):
        method = st.selectbox("Choose a method", ["None", "RFE (RandomForest)", "PCA"])

        if method == "RFE (RandomForest)":
            if st.session_state.model_type == "Clustering":
                st.warning("RFE is not applicable for Clustering tasks.")
            else:
                target_column = st.session_state.target_column
                all_features = [c for c in st.session_state.df_sample.columns if c != target_column]
                X_all = st.session_state.df_sample[all_features]
                X_num = X_all.select_dtypes(include=["number"])

                if X_num.shape[1] == 0:
                    st.error("RFE requires numeric features. Please encode categorical features first.")
                else:
                    n_features_to_select = st.number_input(
                        "Number of features to select",
                        min_value=1,
                        max_value=int(X_num.shape[1]),
                        value=min(5, int(X_num.shape[1])),
                        step=1
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Run RFE", key="run_rfe"):
                            try:
                                st.session_state.df_before_rfe = st.session_state.df_sample.copy()
                                y = st.session_state.df_sample[target_column]
                                xy = pd.concat([X_num, y], axis=1).dropna()
                                X_clean = xy[X_num.columns]
                                y_clean = xy[target_column]

                                estimator = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1) \
                                    if st.session_state.model_type == "Classification" else \
                                    RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)

                                selector = RFE(estimator=estimator, n_features_to_select=int(n_features_to_select),
                                               step=0.1)
                                selector.fit(X_clean, y_clean)

                                selected_features = X_num.columns[selector.support_].tolist()
                                st.success(f"Selected Features: {selected_features}")

                                st.session_state.df_sample = st.session_state.df_sample[
                                    selected_features + [target_column]]

                            except Exception as e:
                                st.error(f"Error during RFE: {e}")

                    with col2:
                        if st.button("Undo RFE", key="undo_rfe"):
                            if "df_before_rfe" in st.session_state:
                                st.session_state.df_sample = st.session_state.df_before_rfe.copy()
                                st.success("Undo RFE applied. Dataset restored.")
        elif method == "PCA":
            # تحديد الأعمدة الرقمية فقط
            features_for_pca = [c for c in st.session_state.df_sample.columns if c != st.session_state.target_column] \
                if st.session_state.model_type in ["Classification",
                                                   "Regression"] else st.session_state.df_sample.columns
            df_for_pca = st.session_state.df_sample[features_for_pca].select_dtypes(include=["number"])

            if df_for_pca.shape[1] < 2:
                st.error("PCA requires at least 2 numeric features.")
            else:
                n_components = st.number_input(
                    "Number of PCA Components",
                    min_value=1,
                    max_value=df_for_pca.shape[1],
                    value=min(2, df_for_pca.shape[1]),
                    step=1
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run PCA", key="run_pca"):
                        try:
                            st.session_state.df_before_pca = st.session_state.df_sample.copy()  # حفظ نسخة قبل التطبيق
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(df_for_pca)

                            pca_df = pd.DataFrame(
                                pca_result,
                                columns=[f"PCA_{i + 1}" for i in range(n_components)]
                            )

                            # إضافة العمود المستهدف لو موجود
                            if st.session_state.model_type in ["Classification", "Regression"]:
                                pca_df[st.session_state.target_column] = st.session_state.df_sample[
                                    st.session_state.target_column]

                            st.session_state.df_sample = pca_df
                            st.success(
                                f"PCA completed with {n_components} components. Dataset replaced with PCA result.")

                        except Exception as e:
                            st.error(f"Error during PCA: {e}")

                with col2:
                    if st.button("Undo PCA", key="undo_pca"):
                        if "df_before_pca" in st.session_state:
                            st.session_state.df_sample = st.session_state.df_before_pca.copy()
                            st.success("Undo PCA applied. Dataset restored.")

    # ----- Models -----
    with st.expander("Models", expanded=True):
        model_options = {
            "Classification": ["None", "RandomForest", "DecisionTree", "LogisticRegression"],
            "Regression": ["None", "RandomForest", "DecisionTree", "LinearRegression"],
            "Clustering": ["None", "KMeans", "AgglomerativeClustering", "DBSCAN"]
        }

        method = st.selectbox("Choose a model", model_options.get(st.session_state.model_type, ["None"]))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Model"):
                try:
                    # حفظ نسخة قبل تطبيق الموديل فقط
                    st.session_state.df_before_model = st.session_state.df_sample.copy()

                    # تحديد X و y حسب نوع الموديل
                    if st.session_state.model_type != "Clustering":
                        if "target_column" not in st.session_state or st.session_state.target_column not in st.session_state.df_sample.columns:
                            st.warning("Target column not selected or missing in dataset.")
                            st.stop()
                        target_column = st.session_state.target_column
                        X = st.session_state.df_sample.drop(columns=[target_column])
                        y = st.session_state.df_sample[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    else:
                        X = st.session_state.df_sample
                        y = None

                    # إنشاء الموديل
                    model = None
                    if method == "RandomForest":
                        model = RandomForestClassifier(
                            random_state=42) if st.session_state.model_type == "Classification" \
                            else RandomForestRegressor(random_state=42)
                    elif method == "DecisionTree":
                        model = DecisionTreeClassifier(
                            random_state=42) if st.session_state.model_type == "Classification" \
                            else DecisionTreeRegressor(random_state=42)
                    elif method == "LogisticRegression" and st.session_state.model_type == "Classification":
                        model = LogisticRegression(max_iter=1000)
                    elif method == "LinearRegression" and st.session_state.model_type == "Regression":
                        model = LinearRegression()
                    elif method == "KMeans":
                        model = KMeans(n_clusters=3, random_state=42)
                    elif method == "AgglomerativeClustering":
                        model = AgglomerativeClustering(n_clusters=3)
                    elif method == "DBSCAN":
                        model = DBSCAN()
                    else:
                        st.warning("Please select a compatible model for the chosen task.")

                    if model:
                        st.subheader("Model Evaluation")

                        if st.session_state.model_type in ["Classification", "Regression"]:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            if st.session_state.model_type == "Classification":
                                acc = accuracy_score(y_test, y_pred)
                                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Accuracy", f"{acc:.4f}")
                                c2.metric("Precision", f"{prec:.4f}")
                                c3.metric("Recall", f"{rec:.4f}")
                                c4.metric("F1-Score", f"{f1:.4f}")
                            else:
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)
                                c1, c2, c3 = st.columns(3)
                                c1.metric("MSE", f"{mse:.4f}")
                                c2.metric("RMSE", f"{rmse:.4f}")
                                c3.metric("R² Score", f"{r2:.4f}")

                        else:  # Clustering
                            if X.shape[1] < 2:
                                st.warning("Need at least 2 features for clustering plot.")
                            labels = model.fit_predict(X)
                            n_clusters = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
                            st.write(f"Number of clusters found: {n_clusters}")

                            try:
                                sil_score = silhouette_score(X, labels)
                                st.write(f"Silhouette Score: {sil_score:.4f}")
                            except Exception as e:
                                st.warning(f"Cannot compute Silhouette Score: {e}")

                            if X.shape[1] >= 2:
                                plt.figure(figsize=(8, 6))
                                sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette="Set2", s=60)
                                plt.xlabel(X.columns[0])
                                plt.ylabel(X.columns[1])
                                plt.title("Clustering Scatter Plot")
                                st.pyplot(plt.gcf())

                except Exception as e:
                    st.error(f"Error applying model: {e}")

        with col2:
            if st.button("Undo Model"):
                if "df_before_model" in st.session_state:
                    st.session_state.df_sample = st.session_state.df_before_model.copy()
                    st.success("Reverted to dataset before applying model.")

    # ---- Reset Data ----
    if st.button("Reset to Original Data", type="secondary"):
        if st.session_state.df_original is not None:
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.df_sample = st.session_state.df.copy()

            # إزالة أي نسخ احتياطية للـ Undo لكل المراحل
            for key in ["df_before_last_sample", "df_before_enc", "df_before_scale",
                        "df_before_outlier", "df_before_transform", "df_before_feature_selection",
                        "df_before_model"]:
                if key in st.session_state:
                    del st.session_state[key]

            st.success("Dataset has been reset to the original data.")
            st.rerun()
        else:
            st.warning("Original data not found.")

    # ---- Step 4: Visualization ----
    st.subheader("Step 4: Data Visualization")

    df_viz = st.session_state.df_sample if "df_sample" in st.session_state else st.session_state.df

    chart_type = st.selectbox(
        "Choose chart type",
        ["Line Chart", "Bar Chart", "Box Plot", "Scatter Plot", "Pie Chart", "Normal Distribution"]
    )

    numeric_cols = [col for col in df_viz.columns if pd.api.types.is_numeric_dtype(df_viz[col])]
    categorical_cols = [col for col in df_viz.columns if not pd.api.types.is_numeric_dtype(df_viz[col])]

    # اختيار الأعمدة
    selected_columns = []
    if chart_type in ["Line Chart", "Normal Distribution", "Box Plot"]:
        selected_columns = st.multiselect("Select numeric columns for plotting", options=numeric_cols)
    elif chart_type in ["Bar Chart", "Pie Chart"]:
        if categorical_cols:
            category_col = st.selectbox("Select category column", options=categorical_cols)
            numeric_col = st.selectbox("Select numeric column (optional)", options=[None] + numeric_cols)
            if category_col:
                selected_columns = [category_col]
                if numeric_col:
                    selected_columns.append(numeric_col)
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis (numeric)", options=numeric_cols)
            y_col = st.selectbox("Select Y-axis (numeric)", options=[col for col in numeric_cols if col != x_col])
            hue_col = st.selectbox("Optional: Select hue column", options=[None] + categorical_cols)
            selected_columns = [x_col, y_col] + ([hue_col] if hue_col else [])

    # رسم المخططات
    if chart_type == "Pie Chart":
        if selected_columns:
            data = df_viz[selected_columns[0]].value_counts()
            fig, ax = plt.subplots()
            ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.warning("Select a category column for Pie Chart.")

    elif chart_type == "Box Plot":
        if selected_columns:
            fig, ax = plt.subplots()
            sns.boxplot(data=df_viz[selected_columns], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select at least one numeric column.")

    elif chart_type == "Line Chart":
        if len(selected_columns) >= 2:
            x_col = selected_columns[0]
            y_cols = selected_columns[1:]
            fig, ax = plt.subplots()
            df_viz.plot(x=x_col, y=y_cols, kind="line", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select X-axis then Y-axis columns.")

    elif chart_type == "Bar Chart":
        if selected_columns:
            category_col = selected_columns[0]
            numeric_col = selected_columns[1] if len(selected_columns) > 1 else None
            fig, ax = plt.subplots()
            if numeric_col:
                data = df_viz.groupby(category_col)[numeric_col].sum()
                data.plot(kind="bar", ax=ax)
            else:
                data = df_viz[category_col].value_counts()
                data.plot(kind="bar", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select category column first.")

    elif chart_type == "Scatter Plot":
        if len(selected_columns) >= 2:
            fig, ax = plt.subplots()
            hue_col = selected_columns[2] if len(selected_columns) == 3 else None
            sns.scatterplot(x=selected_columns[0], y=selected_columns[1], hue=hue_col, data=df_viz, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select X and Y numeric columns for Scatter Plot.")

    elif chart_type == "Normal Distribution":
        if selected_columns:
            fig, ax = plt.subplots()
            for col in selected_columns:
                sns.histplot(df_viz[col].dropna(), kde=True, bins=30, ax=ax, label=col)
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Select at least one numeric column for Normal Distribution.")


