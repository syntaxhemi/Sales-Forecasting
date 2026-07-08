import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# --- Page Configuration ---
st.set_page_config(page_title="Superstore Analytics", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Overview Dashboard", "Forecast Explorer", "Anomaly Report", "Demand Segments"])

# --- Data Loading & Preprocessing ---
@st.cache_data
# @st.cache_data
def load_data():
    # Load data
    df = pd.read_csv("train.csv")
    
    # 1. 'coerce' forces completely broken dates to become NaT (Not a Time) instead of crashing
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    
    # 2. Drop the rows where the date was completely unreadable
    df = df.dropna(subset=['Order Date'])
    
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    
    # Fix Postal Code
    df.loc[(df['City'] == 'Burlington') & (df['State'] == 'Vermont'), 'Postal Code'] = 5401
    df['Postal Code'] = df['Postal Code'].fillna(0).astype(int).astype(str)
    df['Postal Code'] = df['Postal Code'].apply(lambda x: x.zfill(5))
    
    return df
@st.cache_data
def load_vg_data():
    # Load supplementary video game data for multi-source anomaly detection
    vg_df = pd.read_csv("vgsales.csv")
    vg_df = vg_df.dropna(subset=['Year'])
    vg_df['Year'] = vg_df['Year'].astype(int)
    return vg_df

df = load_data()


# PAGE 1: OVERVIEW DASHBOARD

if page == "Overview Dashboard":
    st.title("Sales Overview Dashboard")
    
    col1, col2 = st.columns(2)
    
    # Total Sales by Year
    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales = df.groupby('Year')['Sales'].sum()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        yearly_sales.plot(kind='bar', color='#1f77b4', ax=ax1)
        ax1.set_ylabel("Total Sales ($)")
        plt.xticks(rotation=0)
        st.pyplot(fig1)
        
    # Monthly Sales Trend
    with col2:
        st.subheader("Monthly Sales Trend")
        monthly_trend = df.set_index('Order Date').resample('MS')['Sales'].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(monthly_trend.index, monthly_trend.values, color='#ff7f0e', marker='.')
        ax2.set_ylabel("Total Sales ($)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
    st.markdown("---")
    st.subheader("Interactive Filters: Sales by Region & Category")
    
    filter_col1, filter_col2 = st.columns(2)
    selected_region = filter_col1.selectbox("Select Region", df['Region'].unique())
    selected_category = filter_col2.selectbox("Select Category", df['Category'].unique())
    
    filtered_df = df[(df['Region'] == selected_region) & (df['Category'] == selected_category)]
    st.metric(label=f"Total Sales for {selected_category} in {selected_region}", 
              value=f"${filtered_df['Sales'].sum():,.2f}")
    
    if not filtered_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        filtered_trend = filtered_df.set_index('Order Date').resample('MS')['Sales'].sum()
        ax3.plot(filtered_trend.index, filtered_trend.values, color='green')
        ax3.set_title(f"{selected_category} Sales Trend in {selected_region}")
        st.pyplot(fig3)

# PAGE 2: FORECAST EXPLORER (XGBoost)
elif page == "Forecast Explorer":
    st.title("Forecast Explorer (XGBoost)")
    
    # Controls
    col1, col2 = st.columns(2)
    forecast_target = col1.selectbox("Select Segment to Forecast", 
                                     ["Furniture", "Technology", "Office Supplies", "West Region", "East Region"])
    horizon = col2.slider("Forecast Horizon (Months Ahead)", min_value=1, max_value=3, value=3)
    
    # Filter data based on selection
    if "Region" in forecast_target:
        target_val = forecast_target.split(" ")[0]
        segment_df = df[df['Region'] == target_val]
    else:
        segment_df = df[df['Category'] == forecast_target]
        
    # XGBoost Logic (Task 3 & 4)
    monthly_sales = segment_df.set_index('Order Date').resample('MS')['Sales'].sum().to_frame(name='Sales')
    
    xgb_df = monthly_sales.copy()
    xgb_df['Lag_1'] = xgb_df['Sales'].shift(1)
    xgb_df['Lag_2'] = xgb_df['Sales'].shift(2)
    xgb_df['Lag_3'] = xgb_df['Sales'].shift(3)
    xgb_df['Rolling_Mean'] = xgb_df['Sales'].rolling(3).mean()
    xgb_df['Month'] = xgb_df.index.month
    xgb_df.dropna(inplace=True)
    
    features = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean', 'Month']
    
    # Train/Test Split for Metrics
    train = xgb_df.iloc[:-12]
    test = xgb_df.iloc[-12:]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(train[features], train['Sales'])
    test_preds = model.predict(test[features])
    
    mae = mean_absolute_error(test['Sales'], test_preds)
    rmse = np.sqrt(mean_squared_error(test['Sales'], test_preds))
    
    # Train on FULL data for future forecast
    model.fit(xgb_df[features], xgb_df['Sales'])
    
    # Recursive Prediction
    last_known = xgb_df.iloc[-1:]
    future_dates = pd.date_range(start=xgb_df.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    future_preds = []
    curr_lag1, curr_lag2, curr_lag3 = last_known['Sales'].values[0], last_known['Lag_1'].values[0], last_known['Lag_2'].values[0]
    
    for date in future_dates:
        rolling_val = np.mean([curr_lag1, curr_lag2, curr_lag3])
        pred_features = pd.DataFrame([[curr_lag1, curr_lag2, curr_lag3, rolling_val, date.month]], columns=features)
        pred = model.predict(pred_features)[0]
        future_preds.append(pred)
        curr_lag3, curr_lag2, curr_lag1 = curr_lag2, curr_lag1, pred
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_sales.index[-24:], monthly_sales['Sales'].iloc[-24:], label="Historical Sales", color='#1f77b4', marker='o')
    ax.plot(future_dates, future_preds, label=f"{horizon}-Month Forecast", color='green', linestyle='--', marker='o')
    ax.set_title(f"XGBoost Sales Forecast: {forecast_target}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Metrics display
    st.subheader("Model Evaluation on Holdout Set")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    mcol2.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")

# PAGE 3: MULTI-SOURCE ANOMALY REPORT

elif page == "Anomaly Report":
    st.title("Macro-Anomaly Report")
    st.markdown("Using Isolation Forest on supplementary global video game data to detect multi-variate market outliers.")
    
    vg_df = load_vg_data()
    
    # Isolation Forest Logic
    features = ['NA_Sales', 'EU_Sales', 'JP_Sales']
    iso_vg = IsolationForest(contamination=0.01, random_state=42)
    vg_df['Anomaly_Label'] = iso_vg.fit_predict(vg_df[features])
    
    anomalous_games = vg_df[vg_df['Anomaly_Label'] == -1].sort_values(by='Global_Sales', ascending=False)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(vg_df['NA_Sales'], vg_df['EU_Sales'], color='#1f77b4', alpha=0.3, label='Normal Sales Behavior')
    ax.scatter(anomalous_games['NA_Sales'], anomalous_games['EU_Sales'], color='red', s=50, label='Anomalies (Outliers)', zorder=5)
    
    ax.set_title('Global Market Anomalies (North America vs Europe)')
    ax.set_xlabel('North America Sales (Millions)')
    ax.set_ylabel('Europe Sales (Millions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Data Table
    st.subheader("Top 1% Extreme Outlier Titles")
    display_df = anomalous_games[['Name', 'Platform', 'Year', 'Global_Sales']].copy()
    display_df['Global_Sales'] = display_df['Global_Sales'].apply(lambda x: f"{x:,.2f}M")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

# PAGE 4: DEMAND SEGMENTS

elif page == "Demand Segments":
    st.title("Product Demand Segments")
    
    # Aggregate Sub-Category Features
    total_sales = df.groupby('Sub-Category')['Sales'].sum().rename('Total_Sales')
    aov = df.groupby('Sub-Category')['Sales'].mean().rename('AOV')
    
    cluster_df = pd.concat([total_sales, aov], axis=1).fillna(0)
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(cluster_df)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_df['Cluster'] = kmeans.fit_predict(scaled_df)
    
    # PCA for 2D Plot
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(scaled_df)
    cluster_df['PCA1'] = pca_comp[:, 0]
    cluster_df['PCA2'] = pca_comp[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(cluster_df['PCA1'], cluster_df['PCA2'], c=cluster_df['Cluster'], cmap='viridis', s=100)
    
    # for i, txt in enumerate(cluster_df.index):
    #     ax.annotate(txt, (cluster_df['PCA1'][i]+0.1, cluster_df['PCA2'][i]+0.1), fontsize=9
   
    for i, txt in enumerate(cluster_df.index):
        ax.annotate(txt, (cluster_df['PCA1'].iloc[i] + 0.1, cluster_df['PCA2'].iloc[i] + 0.1), fontsize=9)
        
    ax.set_title("K-Means Product Segmentation (PCA Reduced)")
    st.pyplot(fig)
    
    st.subheader("Sub-Category Assignments")
    display_df = cluster_df[['Total_Sales', 'AOV', 'Cluster']].sort_values('Cluster')
    display_df['Total_Sales'] = display_df['Total_Sales'].apply(lambda x: f"${x:,.2f}")
    display_df['AOV'] = display_df['AOV'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True)
