# prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def app():
    st.title("Prediction Dashboard with Machine Learning")

    if 'dataframe' not in st.session_state:
        st.warning("Please load the data through the 'Google Sheet Link' menu first.")
        return
    
    df = st.session_state['dataframe']
    
    # Display the data
    st.subheader("Loaded Data")
    st.write(df.head())

    # Data Preparation for Machine Learning
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()

    # Splitting the data into training and test sets
    X = monthly_sales.index.values.reshape(-1, 1)
    y = monthly_sales['Sales'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Displaying the results
    st.subheader("Linear Regression Model Results")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='green', label='Test Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Prediction')
    plt.legend()
    st.pyplot(plt)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    # Future Predictions
    st.subheader("Future Sales Prediction")
    future_months = pd.date_range(start=df['Order Date'].max(), periods=12, freq='M').to_period('M').astype(str)
    future_indices = range(len(monthly_sales), len(monthly_sales) + len(future_months))
    future_sales = model.predict([[i] for i in future_indices])
    future_predictions = pd.DataFrame({'Order Month': future_months, 'Predicted Sales': future_sales})
    st.write(future_predictions)
    plt.figure(figsize=(10, 6))
    plt.plot(future_months, future_sales, color='purple', linewidth=2, label='Future Predictions')
    plt.xlabel('Months')
    plt.ylabel('Predicted Sales')
    plt.title('Future Monthly Sales Prediction')
    plt.legend()
    st.pyplot(plt)

    # High Sales Products
    st.header("High Sales Products")
    high_sales_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(high_sales_products)
    
    # Profit Margins
    st.header("Profit Margins")
    profit_margins = df.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(profit_margins)
    
    # High-Value Customers
    st.header("High-Value Customers")
    high_value_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(high_value_customers)
    
    # Discount Impact on Profit
    st.header("Discount Impact on Profit")
    discount_profit = df.groupby('Discount')['Profit'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(discount_profit['Discount'], discount_profit['Profit'], marker='o')
    plt.title('Impact of Discount on Profit')
    plt.xlabel('Discount')
    plt.ylabel('Profit')
    st.pyplot(plt)
    
    # Shipping Efficiency
    st.header("Shipping Efficiency")
    shipping_efficiency = df.groupby('Ship Mode')['Profit'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(shipping_efficiency['Ship Mode'], shipping_efficiency['Profit'], color='skyblue')
    plt.title('Shipping Mode Efficiency')
    plt.xlabel('Ship Mode')
    plt.ylabel('Profit')
    st.pyplot(plt)
if __name__ == '__main__':
    app()
