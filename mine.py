from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load model
model = tf.keras.models.load_model("final_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



st.markdown("""
    <style>
        .stApp {
            background-color: #e6f2ff;
            padding: 20px;
            font-family: 'Arial', sans-serif;
            color: #333;

                   }
        h1, h2, h3 {
            text-align: center;
        }

        /* Black hover effect on input fields */
        input:hover, select:hover, textarea:hover, .stNumberInput input:hover {
            border: 2px solid black !important;
            box-shadow: none !important;
        }

        /* Optional: Improve padding & spacing in input fields */
        input, select, textarea, .stNumberInput input {
            padding: 8px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "form_data" not in st.session_state:
    st.session_state.form_data = {}

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1




# === STEP 0: Landing Page ===
if st.session_state.step == 0:
    #st.image("https://sdmntprsouthcentralus.oaiusercontent.com/files/00000000-9864-61f7-8be3-2b160185129a/raw?se=2025-05-04T09%3A48%3A19Z&sp=r&sv=2024-08-04&sr=b&scid=8ef889e2-f5ed-524b-9b8c-ce1f3d18dee7&skoid=cbbaa726-4a2e-4147-932c-56e6e553f073&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-03T17%3A08%3A28Z&ske=2025-05-04T17%3A08%3A28Z&sks=b&skv=2024-08-04&sig=U%2B%2BltjPcV5NUuMasotGc%2BoXj77tFFeDDGvJUinDSO%2BU%3D",width=400)
    st.title("💡 Welcome to the Loan Risk Predictor")
    st.markdown("""
    This app helps you understand your **loan risk** category based on various financial indicators.

    📊 **Benefits:**
    - Predict your **loan risk** level (Good, Standard, Poor)
    - Get insights into your **financial behavior**
    - Make better decisions for credit and debt management

    🛠️ Powered by a trained machine learning model built with TensorFlow.
    """)
    
    if st.button("🚀 Get Started"):
        next_step()

# === STEP 1 ===
elif st.session_state.step == 1:
    st.title("📋 Loan Risk Classification Form")
    st.progress(st.session_state.step / 3)
    st.header("Step 1: Basic Information")

    with st.form("step1_form"):
        col1, col2 = st.columns(2)
        with col1:
            ID = st.text_input("ID", value=st.session_state.form_data.get("ID", ""))
            Customer_ID = st.text_input("Customer ID", value=st.session_state.form_data.get("Customer_ID", ""))
            Month = st.text_input("Month", value=st.session_state.form_data.get("Month", ""))
            Name = st.text_input("Name", value=st.session_state.form_data.get("Name", ""))
            Age = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.form_data.get("Age", 30)))
            SSN = st.text_input("SSN", value=st.session_state.form_data.get("SSN", ""))
        with col2:
            occupations = ["Accountant", "Engineer", "Doctor", "Teacher", "Lawyer", "Manager","Scientist"]
            Occupation = st.selectbox("Occupation", occupations,
                                      index=occupations.index(st.session_state.form_data.get("Occupation", "Accountant")))
            Annual_Income = st.number_input("Annual Income", value=float(st.session_state.form_data.get("Annual_Income", 500000)))
            Monthly_Inhand_Salary = st.number_input("Monthly In-hand Salary", value=float(st.session_state.form_data.get("Monthly_Inhand_Salary", 1000)))
            Num_Bank_Accounts = st.number_input("Number of Bank Accounts", min_value=0, step=1, value=int(st.session_state.form_data.get("Num_Bank_Accounts", 2)))
            Num_Credit_Card = st.number_input("Number of Credit Cards", min_value=0, step=1, value=int(st.session_state.form_data.get("Num_Credit_Card", 2)))
            Interest_Rate = st.number_input("Interest Rate (%)", min_value=0.0, value=float(st.session_state.form_data.get("Interest_Rate", 10)))

        col_back, col_next = st.columns(2)
        with col_back:
            back_clicked = st.form_submit_button("Back")
        with col_next:
            next_clicked = st.form_submit_button("Next")
        if back_clicked:
            prev_step()
        if next_clicked:
            errors = []
            if not ID.strip(): errors.append("ID is required.")
            if not Customer_ID.strip(): errors.append("Customer ID is required.")
            if not Month.strip(): errors.append("Month is required.")
            if not Name.strip(): errors.append("Name is required.")
            if not SSN.strip(): errors.append("SSN is required.")
            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.session_state.form_data.update({
                    "ID": ID, "Customer_ID": Customer_ID, "Month": Month, "Name": Name, "Age": Age,
                    "SSN": SSN, "Occupation": Occupation, "Annual_Income": Annual_Income,
                    "Monthly_Inhand_Salary": Monthly_Inhand_Salary, "Num_Bank_Accounts": Num_Bank_Accounts,
                    "Num_Credit_Card": Num_Credit_Card, "Interest_Rate": Interest_Rate
                })
                next_step()

    

# === STEP 2 ===
elif st.session_state.step == 2:
    st.title("📋 Credit Score Classification Form")
    st.progress(st.session_state.step / 4)
    st.header("Step 2: Loan & Payment Info")
    with st.form("step2_form"):
        col1, col2 = st.columns(2)
        with col1:
            Num_of_Loan = st.number_input("Number of Loans", min_value=0, step=1, value=int(st.session_state.form_data.get("Num_of_Loan", 1)))
            Type_of_Loan = st.text_input("Type of Loan (optional)", value=st.session_state.form_data.get("Type_of_Loan", "Home Loan"))
            Delay_from_due_date = st.number_input("Delay from Due Date (days)", step=1, value=int(st.session_state.form_data.get("Delay_from_due_date", 0)))
            Num_of_Delayed_Payment = st.number_input("Number of Delayed Payments", min_value=0, step=1, value=int(st.session_state.form_data.get("Num_of_Delayed_Payment", 0)))
        with col2:
            Changed_Credit_Limit = st.number_input("Changed Credit Limit", value=float(st.session_state.form_data.get("Changed_Credit_Limit", 10000)))
            Num_Credit_Inquiries = st.number_input("Number of Credit Inquiries", min_value=0, step=1, value=int(st.session_state.form_data.get("Num_Credit_Inquiries", 1)))
            Credit_Mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"],
                                      index=["Bad", "Standard", "Good"].index(st.session_state.form_data.get("Credit_Mix", "Standard")))
            Outstanding_Debt = st.number_input("Outstanding Debt", value=float(st.session_state.form_data.get("Outstanding_Debt", 50000)))

        col1b, col2b = st.columns(2)
        back = col1b.form_submit_button("Back")
        next_ = col2b.form_submit_button("Next")
        if back:
            prev_step()
        elif next_:
            st.session_state.form_data.update({
                "Num_of_Loan": Num_of_Loan, "Type_of_Loan": Type_of_Loan,
                "Delay_from_due_date": Delay_from_due_date,
                "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
                "Changed_Credit_Limit": Changed_Credit_Limit,
                "Num_Credit_Inquiries": Num_Credit_Inquiries,
                "Credit_Mix": Credit_Mix,
                "Outstanding_Debt": Outstanding_Debt
            })
            next_step()

# === STEP 3 ===
elif st.session_state.step == 3:
    st.title("📋 Credit Score Classification Form")
    st.progress(st.session_state.step / 4)
    st.header("Step 3: Financial Behavior")
    with st.form("step3_form"):
        col1, col2 = st.columns(2)
        with col1:
            Credit_Utilization_Ratio = st.number_input("Credit Utilization Ratio", value=float(st.session_state.form_data.get("Credit_Utilization_Ratio", 0.0)))
            Credit_History_Age = st.text_input("Credit History Age (e.g., 3 Years 4 Months)", value=st.session_state.form_data.get("Credit_History_Age", "2 Years 6 Months"))
            Payment_of_Min_Amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"],
                                                 index=["Yes", "No"].index(st.session_state.form_data.get("Payment_of_Min_Amount", "No")))
        with col2:
            Total_EMI_per_month = st.number_input("Total EMI per Month", value=float(st.session_state.form_data.get("Total_EMI_per_month", 0.0)))
            Amount_invested_monthly = st.number_input("Amount Invested Monthly", value=float(st.session_state.form_data.get("Amount_invested_monthly", 0)))
            Payment_Behaviour = st.selectbox("Payment Behaviour", [
                "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
                "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments", "High_spent_Medium_value_payments"
            ], index=[
                "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
                "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments", "High_spent_Medium_value_payments"
            ].index(st.session_state.form_data.get("Payment_Behaviour", "High_spent_Large_value_payments")))
            Monthly_Balance = st.number_input("Monthly Balance", value=float(st.session_state.form_data.get("Monthly_Balance", 10000)))

        col1b, col2b = st.columns(2)
        back = col1b.form_submit_button("Back")
        next_ = col2b.form_submit_button("Next")
        if back:
            prev_step()
        elif next_:
            if not Credit_History_Age.strip():
                st.error("Credit History Age is required.")
            else:
                st.session_state.form_data.update({
                    "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
                    "Credit_History_Age": Credit_History_Age,
                    "Payment_of_Min_Amount": Payment_of_Min_Amount,
                    "Total_EMI_per_month": Total_EMI_per_month,
                    "Amount_invested_monthly": Amount_invested_monthly,
                    "Payment_Behaviour": Payment_Behaviour,
                    "Monthly_Balance": Monthly_Balance
                })
                next_step()

# === STEP 4 ===
elif st.session_state.step == 4:
    st.title("📋 Credit Score Classification Form")
    st.progress(1.0)
    st.header("Step 4: Review & Submit")
    
    Credit_Score = st.selectbox("Credit Score (for review)", ["Good", "Standard", "Poor"])
    st.session_state.form_data["Credit_Score"] = Credit_Score

    st.write("### Your Data:")
    st.json(st.session_state.form_data)

    col1, col2, col3 = st.columns(3)
    with col1:
        back = st.button("⬅️ Previous")
    with col2:
        submit = st.button("✅ Submit")
    with col3:
        visualize = st.button("📊 Visualize")

    if back:
        prev_step()
    if submit:
        st.success("✅ Form Submitted! Predicting...")
        st.balloons()
         # Convert input to DataFrame
        df_input = pd.DataFrame([st.session_state.form_data])

        # Convert credit history age to months
        def parse_credit_history_age(age_str):
            match = re.match(r"(\d+)\s*Years?\s*(\d*)\s*Months?", age_str)
            if match:
                years = int(match.group(1))
                months = int(match.group(2)) if match.group(2) else 0
                return years * 12 + months
            return 0

        df_input["Credit_History_Age"] = df_input["Credit_History_Age"].apply(parse_credit_history_age)

        # Label Encoding
        categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
        for col in categorical_cols:
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col])

        # Drop unused columns
        drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Credit_Score', 'Type_of_Loan']
        df_input.drop(columns=drop_cols, inplace=True, errors='ignore')
        df_input.fillna(0, inplace=True)

        st.write("✅ Final input shape:", df_input.shape)

        try:
            prediction = model.predict(df_input)
            classes = ["Good", "Standard", "Poor"]
            predicted_class = classes[np.argmax(prediction)]  # Get the predicted class
            probabilities_percent = {cls: f"{prob*100:.2f}%" for cls, prob in zip(classes, prediction[0])}

            # Store the prediction and predicted class in session state
            st.session_state.predicted_class = predicted_class
            st.session_state.prediction_probs = probabilities_percent

            st.success(f"🎯 Predicted Loan Risk Category: **{predicted_class}**")
            st.write("🔢 Class Probabilities (%):", probabilities_percent)
        except Exception as e:
            st.error(f"⚠️ Prediction Failed: {e}")
    if visualize:
        # Set step to 5 for visualizations
        st.session_state.step = 5

       




# Step 5: Prediction and Visualization
elif st.session_state.step == 5:
    st.header("Step 5: Prediction and Insights")
    
    # Define the predict_credit_score function
    def predict_credit_score(form_data):
        # Convert input to DataFrame
        df_input = pd.DataFrame([form_data])

        # Convert credit history age to months
        def parse_credit_history_age(age_str):
            match = re.match(r"(\d+)\s*Years?\s*(\d*)\s*Months?", age_str)
            if match:
                years = int(match.group(1))
                months = int(match.group(2)) if match.group(2) else 0
                return years * 12 + months
            return 0

        df_input["Credit_History_Age"] = df_input["Credit_History_Age"].apply(parse_credit_history_age)

        # Label Encoding
        categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
        for col in categorical_cols:
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col])

        # Drop unused columns
        drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Credit_Score', 'Type_of_Loan']
        df_input.drop(columns=drop_cols, inplace=True, errors='ignore')
        df_input.fillna(0, inplace=True)

        # Predict using the model
        prediction = model.predict(df_input)
        classes = ["Good", "Standard", "Poor"]
        predicted_class = classes[np.argmax(prediction)]  # Get the predicted class
        return predicted_class

    # Call the function
    prediction = predict_credit_score(st.session_state.form_data)
    st.success(f"Predicted Credit Score: {prediction}")

    
    
    # Replace 'train' with your dataset
    # Visualization: Average Monthly Inhand Salary by Credit Score
    
    try:
        # Example dataset for visualization (replace with actual dataset)
        train = pd.read_csv("clean_train.csv") 
       
       
        
        st.subheader("Applicant Data (Statistical Summary)")
        default_data1 = train['Monthly_Inhand_Salary'].describe()
        st.table(pd.DataFrame(default_data1.items(), columns=["Field", "Value"]))
        # Visualization: Average Monthly Inhand Salary by Credit Score
        st.subheader("Average Monthly Inhand Salary by Credit Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        barplot = sns.barplot(x='Credit_Score', y='Monthly_Inhand_Salary', data=train, ci=None, palette='Greens_r', ax=ax)
        
        # Add annotations to the bar plot
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9), textcoords='offset points')
            
        
        ax.set_title('Average Monthly Inhand Salary by Credit Score')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Average Monthly Inhand Salary')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error generating visualization: {e}")
            # Visualization: Credit Mix Distribution by Credit Score
    st.subheader("Credit Mix Distribution by Credit Score")
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        train = pd.read_csv("clean_train.csv") 
        sns.countplot(x='Credit_Mix', hue='Credit_Score', data=train, palette='Greens_r', ax=ax2)
        
        ax2.set_title('Credit Mix Distribution by Credit Score')
        ax2.set_xlabel('Credit Mix')
        ax2.set_ylabel('Count')
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"⚠️ Error generating Credit Mix visualization: {e}")
            # Visualization: Average Outstanding Debt by Credit Score
    st.subheader("Average Outstanding Debt by Credit Score")
    try:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        train = pd.read_csv("clean_train.csv") 
        sns.barplot(x='Credit_Score', y='Outstanding_Debt', data=train, ci=None, palette='Greens_r', ax=ax3)

        # Annotate bars with values
        for p in ax3.patches:
            ax3.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 9), textcoords='offset points')

        ax3.set_title('Average Outstanding Debt by Credit Score')
        ax3.set_xlabel('Credit Score')
        ax3.set_ylabel('Average Outstanding Debt')
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"⚠️ Error generating Outstanding Debt visualization: {e}")
    # Visualization: Average Delay from Due Date by Credit Score
    st.subheader("Average Delay from Due Date by Credit Score")
    try:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        train = pd.read_csv("clean_train.csv") 
        sns.barplot(x='Credit_Score', y='Delay_from_due_date', data=train, ci=None, palette='Greens_r', ax=ax4)

        # Annotate bars with values
        for p in ax4.patches:
            ax4.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 9), textcoords='offset points')

        ax4.set_title('Average Delay from Due Date by Credit Score')
        ax4.set_xlabel('Credit Score')
        ax4.set_ylabel('Average Delay from Due Date')
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"⚠️ Error generating Delay from Due Date visualization: {e}")
            # Visualization: Average Delay from Due Date by Credit Score
    st.subheader("Average Delay from Due Date by Credit Score")
    try:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Credit_Score', y='Delay_from_due_date', data=train, ci=None, palette='Greens_r', ax=ax4)

        # Annotate bars with values
        for p in ax4.patches:
            ax4.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 9), textcoords='offset points')

        ax4.set_title('Average Delay from Due Date by Credit Score')
        ax4.set_xlabel('Credit Score')
        ax4.set_ylabel('Average Delay from Due Date')
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"⚠️ Error generating Delay from Due Date visualization: {e}")
        # Restart button
    st.button("Restart", on_click=lambda: st.session_state.update(step=1, form_data={}))

        
   




    






