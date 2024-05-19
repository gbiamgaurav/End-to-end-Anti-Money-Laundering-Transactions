import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the model and preprocessor
model = joblib.load("final_model/model.joblib")
preprocessor = joblib.load("final_model/preprocessor.joblib")

def main():
    st.title('Money Laundering Detection System')

    st.markdown("Please enter the customer details to check Money Laundering")


    from_bank = st.slider('From Bank', 1, 356233, step=100, value=None)
    Account = st.slider('Account', 1, 1648, step=1, value=None)
    to_bank = st.slider('To Bank', 1, 415, step=1, value=None)
    Account_1 = st.slider('Account.1', 1, 1648, step=1, value=None)
    Amount_received = st.slider('Amount Received', 1, 119057635622, step=1, value=None)

    Receiving_Currency = st.selectbox('Job', ['US Dollar','Euro', 'Swiss Franc', 'Yuan',
                                             'Shekel', 'UK Pound', 'Rupee', 'Yen',
                                            'Ruble', 'Bitcoin', 'Australian Dollar',
                                            'Canadian Dollar', 'Mexican Peso', 'Saudi Riyal',
                                            'Brazil Real'])

    Payment_format = st.selectbox('Payment Format', ['Cheque','Credit Card','ACH',
                                                     'Cash','Reinvestment', 'Wire', 'Bitcoin']) 

    with st.form("Term Deposit Prediction"):
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        try:
            if any(value is None for value in [from_bank, Account, to_bank, Account_1, Amount_received, Receiving_Currency, 
                                               Payment_format]):
                st.warning("Please select values for all input parameters.")
            else:
                input_data = pd.DataFrame({
                    'From Bank': [from_bank],
                    'To Bank': [to_bank],
                    'Account': [Account],
                    'Account.1': [Account_1],
                    'Amount Received': [Amount_received],  # Corrected column name
                    'Receiving Currency': [Receiving_Currency],
                    'Payment Format': [Payment_format],  # Corrected column name
                })

                # Preprocess the input data
                X_transformed = preprocessor.transform(input_data)

                # Make the prediction
                prediction = model.predict(X_transformed)[0]

                result_text = "No Money Laundering" if prediction == 0 else "Money Laundering"
                st.write(f"Predicted Result: {result_text}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()