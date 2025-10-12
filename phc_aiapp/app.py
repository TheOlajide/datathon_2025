import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from deep_translator import GoogleTranslator
import redshift_connector

# -----------------------------
# Database Connection
# -----------------------------
REDSHIFT_HOST = "dfa-datafest.962626097808.eu-north-1.redshift-serverless.amazonaws.com"
REDSHIFT_PORT = 5439
REDSHIFT_DB = "datafest"
REDSHIFT_USER = "admin"
REDSHIFT_PASSWORD = "Newpassword0703"

TABLE_FACILITIES = "staging.staging_nigeria_phc"
TABLE_WORKERS = "staging.staging_health_workers"
TABLE_PATIENTS = "staging.staging_patients_dataset"
TABLE_DISEASES = "staging.staging_disease_report"
TABLE_INVENTORY = "staging.staging_inventory_dataset"

# -----------------------------
# Translation
# -----------------------------
LANG_MAP = {"English": "en", "Igbo": "ig", "Yoruba": "yo", "Hausa": "ha"}

def translate_text(text, lang):
    if lang == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target=LANG_MAP.get(lang, "en")).translate(text)
    except Exception:
        return text

# -----------------------------
# Load Data with Caching
# -----------------------------
@st.cache_data(ttl=3600)
def load_table(table_name, columns, limit=100000):
    conn = redshift_connector.connect(
        host=REDSHIFT_HOST,
        port=REDSHIFT_PORT,
        database=REDSHIFT_DB,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASSWORD
    )
    cursor = conn.cursor()
    query = f"SELECT {columns} FROM {table_name} LIMIT {limit};"
    cursor.execute(query)
    col_names = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(data, columns=[c.lower() for c in col_names])
    return df

# -----------------------------
# Load All Tables
# -----------------------------
facilities_df = load_table(
    TABLE_FACILITIES,
    "facility_id, facility_name, state, lga, ownership, type, latitude, longitude, operational_status, number_of_beds, average_daily_patients, health_workers"
)
# Keep only Functional or Partially Functional
facilities_df = facilities_df[
    facilities_df["operational_status"].str.lower().isin(["functional", "partially functional"])
].copy()

workers_df = load_table(
    TABLE_WORKERS,
    "worker_id, facility_id, name, role, qualification, years_experience, gender, specialization, shift, availability_status"
)

patients_df = load_table(
    TABLE_PATIENTS,
    "patient_id, facility_id, gender, age, visit_date, diagnosis, treatment, outcome"
)
patients_df["visit_date"] = pd.to_datetime(patients_df["visit_date"], errors="coerce")

diseases_df = load_table(
    TABLE_DISEASES,
    "report_id, facility_id, month, disease, cases_reported, deaths"
)
diseases_df["month_dt"] = pd.to_datetime(diseases_df["month"], errors="coerce")
diseases_df["cases_reported"] = pd.to_numeric(diseases_df["cases_reported"], errors="coerce").fillna(0)

inventory_df = load_table(
    TABLE_INVENTORY,
    "item_id, facility_id, item_name, stock_level, reorder_level, last_restock_date"
)
inventory_df["stock_level"] = pd.to_numeric(inventory_df["stock_level"], errors="coerce").fillna(0)
inventory_df["reorder_level"] = pd.to_numeric(inventory_df["reorder_level"], errors="coerce").fillna(0)
inventory_df["last_restock_date"] = pd.to_datetime(inventory_df["last_restock_date"], errors="coerce")

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="PHC AI Dashboard", layout="wide")
st.sidebar.title("🏥 PHC Dashboard")
tab_selection = st.sidebar.radio("Navigation", [
    "Facility Dashboard", "PHC Chatbot", "Disease Forecasting", "Inventory Overview", "Operational Status"
])

# -----------------------------
# 1. Facility Dashboard
# -----------------------------
if tab_selection == "Facility Dashboard":
    st.header("Facility Overview")
    facility_name = st.selectbox("Select a PHC facility", facilities_df["facility_name"].unique())
    facility = facilities_df[facilities_df["facility_name"] == facility_name]
    facility_id = facility["facility_id"].values[0]

    st.subheader(f"📍 {facility_name}")
    st.metric("👩‍⚕️ Number of Health Workers", len(workers_df[workers_df["facility_id"] == facility_id]))
    st.metric("🏥 Number of Beds", int(facility["number_of_beds"].values[0]))
    st.metric("👨‍👩‍👧‍👦 Average Daily Patients", int(facility["average_daily_patients"].values[0]))

    visits_df = patients_df[patients_df["facility_id"] == facility_id]
    if not visits_df.empty:
        visits_chart = visits_df.groupby("visit_date").size().reset_index(name="visits")
        st.plotly_chart(px.line(visits_chart, x="visit_date", y="visits",
                                title="📅 Patient Visits Over Time"))

# -----------------------------
# 2. PHC Chatbot
# -----------------------------
elif tab_selection == "PHC Chatbot":
    st.header("🤖 PHC AI Chatbot Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    lang = st.selectbox("Select Language", list(LANG_MAP.keys()), index=0)
    user_input = st.chat_input("Ask about facilities, stock, workers, patients, or diseases...")

    if user_input:
        st.chat_message("user").write(user_input)
        query = user_input.lower()
        response = "I’m not sure how to answer that yet."

        # Facility match (optional)
        facility_match = [name for name in facilities_df["facility_name"] if name.lower() in query]
        f_id = None
        if facility_match:
            facility_name = facility_match[0]
            f_id = facilities_df[facilities_df["facility_name"] == facility_name]["facility_id"].values[0]

        # Stock queries
        if any(word in query for word in ["stock", "inventory", "medicine", "item"]):
            filtered_inventory = inventory_df.copy()
            if f_id:
                filtered_inventory = filtered_inventory[filtered_inventory["facility_id"] == f_id]
            item_match = [item for item in filtered_inventory["item_name"] if item.lower() in query]
            if item_match:
                item_name = item_match[0]
                records = filtered_inventory[filtered_inventory["item_name"] == item_name]
                response_lines = []
                for _, r in records.iterrows():
                    color = "green" if r["stock_level"] >= r["reorder_level"] else "red"
                    facility_name_r = facilities_df[facilities_df["facility_id"] == r["facility_id"]]["facility_name"].values[0]
                    response_lines.append(
                        f"**{item_name}** in **{facility_name_r}**: Stock: <span style='color:{color}'>{r['stock_level']}</span>, Reorder: {r['reorder_level']}"
                    )
                response = "\n".join(response_lines)
            else:
                response = "No matching stock records found."

        # Operational queries
        elif any(word in query for word in ["operational", "in operation", "running", "active"]):
            filtered_facilities = facilities_df.copy()
            for state in facilities_df["state"].unique():
                if state.lower() in query:
                    filtered_facilities = filtered_facilities[filtered_facilities["state"].str.lower() == state.lower()]
            for lga in facilities_df["lga"].unique():
                if lga.lower() in query:
                    filtered_facilities = filtered_facilities[filtered_facilities["lga"].str.lower() == lga.lower()]
            if f_id:
                filtered_facilities = filtered_facilities[filtered_facilities["facility_id"] == f_id]
            if not filtered_facilities.empty:
                response = "Operational Facilities:\n" + ", ".join(filtered_facilities["facility_name"].tolist())
            else:
                response = "No operational facilities found."

        # Health worker queries
        elif any(word in query for word in ["worker", "doctor", "nurse", "staff"]):
            filtered_workers = workers_df.copy()
            if f_id:
                filtered_workers = filtered_workers[filtered_workers["facility_id"] == f_id]
            role_match = [role for role in filtered_workers["role"].unique() if role.lower() in query]
            if role_match:
                filtered_workers = filtered_workers[filtered_workers["role"].str.lower() == role_match[0].lower()]
            if not filtered_workers.empty:
                response = ", ".join(filtered_workers["name"].tolist())
            else:
                response = "No workers found for the query."

        # Patient queries
        elif any(word in query for word in ["patient", "visit", "attendance"]):
            filtered_patients = patients_df.copy()
            if f_id:
                filtered_patients = filtered_patients[filtered_patients["facility_id"] == f_id]
            response = f"Total patient visits: {len(filtered_patients)}"

        # Disease queries
        elif any(word in query for word in ["disease", "cases", "malaria", "cholera"]):
            filtered_diseases = diseases_df.copy()
            if f_id:
                filtered_diseases = filtered_diseases[filtered_diseases["facility_id"] == f_id]
            disease_match = [d for d in filtered_diseases["disease"].unique() if d.lower() in query]
            if disease_match:
                disease_name = disease_match[0]
                total_cases = filtered_diseases[filtered_diseases["disease"] == disease_name]["cases_reported"].sum()
                response = f"Total **{disease_name}** cases: {total_cases}"
            else:
                response = "No matching disease records found."

        response_translated = translate_text(response, lang)
        st.chat_message("assistant").markdown(response_translated, unsafe_allow_html=True)

# -----------------------------
# 3. Disease Forecasting
# -----------------------------
elif tab_selection == "Disease Forecasting":
    st.header("📈 Disease Case Forecasting")
    disease = st.selectbox("Select Disease", diseases_df["disease"].unique())
    df = diseases_df[diseases_df["disease"] == disease].groupby("month_dt")["cases_reported"].sum().reset_index()
    if len(df) > 2:
        df.columns = ["ds", "y"]
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=6, freq="M")
        forecast = model.predict(future)
        st.plotly_chart(px.line(forecast, x="ds", y="yhat", title=f"Forecast for {disease} Cases"))
    else:
        st.warning("Not enough data to forecast for this disease.")

# -----------------------------
# 4. Inventory Overview
# -----------------------------
elif tab_selection == "Inventory Overview":
    st.header("📦 Inventory Overview")
    facility_select = st.selectbox("Select Facility", facilities_df["facility_name"].unique())
    f_id = facilities_df[facilities_df["facility_name"] == facility_select]["facility_id"].values[0]
    inv_df = inventory_df[inventory_df["facility_id"] == f_id].copy()
    if inv_df.empty:
        st.info(f"No inventory records found for {facility_select}.")
    else:
        inv_df["low_stock"] = inv_df["stock_level"] <= inv_df["reorder_level"]
        st.dataframe(inv_df.style.applymap(lambda v: 'color:red;' if isinstance(v,bool) and v else '', subset=["low_stock"]))

# -----------------------------
# 5. Operational Status
# -----------------------------
elif tab_selection == "Operational Status":
    st.header("🏥 Facility Operational Status")
    status_df = facilities_df[["facility_name", "state", "lga", "operational_status"]].copy()
    status_df["status_color"] = status_df["operational_status"].apply(lambda x: 'green' if x.lower() == "functional" else 'orange')
    st.dataframe(status_df.style.applymap(lambda c: f'color:{c};', subset=["status_color"]))
