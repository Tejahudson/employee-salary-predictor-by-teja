import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used to load the pre-trained model
import time # For simulating loading time

# --- Configuration ---
# Set page title and favicon
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💰", layout="centered")

# --- Load the Model ---
# IMPORTANT: Ensure 'salary_prediction_model.pkl' is in the same directory as this app.py
try:
    # Load the trained pipeline (preprocessor + model)
    model_pipeline = joblib.load('salary_prediction_model.pkl')
    st.success("Machine Learning model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'salary_prediction_model.pkl' not found. Please ensure your trained model is saved and accessible in the same directory.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# --- Currency Conversion Rates (Examples) ---
# This is a simplified approach. For a production app, you'd use a real-time API.
# Rates are relative to USD.
USD_TO_INR_RATE = 83.5
USD_TO_EUR_RATE = 0.92  # 1 USD = 0.92 EUR
USD_TO_GBP_RATE = 0.79  # 1 USD = 0.79 GBP
USD_TO_CAD_RATE = 1.37  # 1 USD = 1.37 CAD
USD_TO_AUD_RATE = 1.50  # 1 USD = 1.50 AUD
USD_TO_JPY_RATE = 157.0 # 1 USD = 157.0 JPY
USD_TO_CHF_RATE = 0.89  # 1 USD = 0.89 CHF

# Mapping of country codes to their primary currency symbol and rate (relative to USD)
COUNTRY_CURRENCIES = {
    "US": {"symbol": "$", "name": "USD", "rate": 1.0},
    "GB": {"symbol": "£", "name": "GBP", "rate": USD_TO_GBP_RATE},
    "CA": {"symbol": "C$", "name": "CAD", "rate": USD_TO_CAD_RATE},
    "DE": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "IN": {"symbol": "₹", "name": "INR", "rate": USD_TO_INR_RATE},
    "FR": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "ES": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "AU": {"symbol": "A$", "name": "AUD", "rate": USD_TO_AUD_RATE},
    "BR": {"symbol": "R$", "name": "BRL", "rate": 5.40}, # Example rate
    "NL": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "JP": {"symbol": "¥", "name": "JPY", "rate": USD_TO_JPY_RATE},
    "CH": {"symbol": "CHF", "name": "CHF", "rate": USD_TO_CHF_RATE},
    "IT": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "PL": {"symbol": "zł", "name": "PLN", "rate": 4.05}, # Example rate
    "PT": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "MX": {"symbol": "$", "name": "MXN", "rate": 18.0}, # Example rate
    "DK": {"symbol": "kr", "name": "DKK", "rate": 6.90}, # Example rate
    "GR": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "TR": {"symbol": "₺", "name": "TRY", "rate": 32.5}, # Example rate
    "AT": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "BE": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "IE": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "LU": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "NG": {"symbol": "₦", "name": "NGN", "rate": 1500.0}, # Example rate
    "PK": {"symbol": "₨", "name": "PKR", "rate": 278.0}, # Example rate
    "RU": {"symbol": "₽", "name": "RUB", "rate": 87.0}, # Example rate
    "SG": {"symbol": "S$", "name": "SGD", "rate": 1.35}, # Example rate
    "UA": {"symbol": "₴", "name": "UAH", "rate": 40.0}, # Example rate
    "AE": {"symbol": "د.إ", "name": "AED", "rate": 3.67}, # Example rate
    "CL": {"symbol": "CLP", "name": "CLP", "rate": 930.0}, # Example rate
    "CO": {"symbol": "$", "name": "COP", "rate": 4000.0}, # Example rate
    "CY": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "CZ": {"symbol": "Kč", "name": "CZK", "rate": 23.0}, # Example rate
    "EE": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "FI": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "GH": {"symbol": "₵", "name": "GHS", "rate": 15.0}, # Example rate
    "HR": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "HU": {"symbol": "Ft", "name": "HUF", "rate": 360.0}, # Example rate
    "IR": {"symbol": "﷼", "name": "IRR", "rate": 42000.0}, # Example rate
    "MT": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "NZ": {"symbol": "NZ$", "name": "NZD", "rate": 1.63}, # Example rate
    "PH": {"symbol": "₱", "name": "PHP", "rate": 58.0}, # Example rate
    "PR": {"symbol": "$", "name": "USD", "rate": 1.0}, # Puerto Rico uses USD
    "RO": {"symbol": "lei", "name": "RON", "rate": 4.60}, # Example rate
    "SI": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "SK": {"symbol": "€", "name": "EUR", "rate": USD_TO_EUR_RATE},
    "TH": {"symbol": "฿", "name": "THB", "rate": 36.0}, # Example rate
    "VN": {"symbol": "₫", "name": "VND", "rate": 25400.0}, # Example rate
    # Add more countries and their currency symbols/rates as needed
}


# --- Web Application Layout & Styling ---

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        background-color: #1a1a2e; /* Dark background for a sleek look */
        color: #e0e0e0; /* Light text color */
    }

    .main {
        background-color: #2a2a4a; /* Slightly lighter dark background for content */
        padding: 40px;
        border-radius: 20px; /* Even more rounded corners */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); /* Deeper shadow */
        max-width: 950px; /* Wider for better presentation */
        margin: 40px auto;
        border: 1px solid #3a3a5e; /* Subtle border for definition */
    }

    h1 {
        color: #00bcd4; /* Cyan for main title */
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
        font-weight: 800; /* Extra bold title */
        font-size: 3em; /* Larger title */
        text-shadow: 2px 2px 5px rgba(0, 188, 212, 0.3); /* Text shadow for pop */
    }

    .stMarkdown h2 {
        color: #8aff8a; /* Light green for subheadings */
        margin-top: 40px;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        border-bottom: 2px solid #4a4a7a; /* Darker underline */
        padding-bottom: 10px;
    }

    .stButton>button {
        background-color: #ff4081; /* Accent pink/red button */
        color: white;
        padding: 15px 30px;
        border-radius: 12px; /* Super rounded button corners */
        border: none;
        font-size: 20px; /* Larger font for button */
        font-weight: 700;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        width: 100%;
        margin-top: 30px;
        box-shadow: 0 6px 15px rgba(255, 64, 129, 0.4); /* Vibrant shadow */
    }

    .stButton>button:hover {
        background-color: #e0326e; /* Darker pink/red on hover */
        transform: translateY(-5px); /* More pronounced lift effect */
        box-shadow: 0 8px 20px rgba(255, 64, 129, 0.6);
    }

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        background-color: #3a3a5e; /* Dark input background */
        color: #e0e0e0; /* Light text in inputs */
        border-radius: 10px;
        border: 1px solid #5a5a8a; /* Slightly lighter border */
        padding: 12px 18px;
        font-family: 'Inter', sans-serif;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    /* Specific styling for selectbox dropdown text */
    .stSelectbox>div>div {
        background-color: #3a3a5e; /* Dark input background */
        color: #e0e0e0; /* Light text in inputs */
        border-radius: 10px;
        border: 1px solid #5a5a8a; /* Slightly lighter border */
        padding: 12px 18px;
        font-family: 'Inter', sans-serif;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    /* Ensure selected text in dropdown is visible */
    .stSelectbox>div>div>div>div>span {
        color: #e0e0e0 !important; /* Force light color for selected text */
    }

    .stSelectbox>div>div:focus-visible,
    .stTextInput>div>div>input:focus-visible,
    .stNumberInput>div>div>input:focus-visible {
        border-color: #00bcd4; /* Highlight focus with cyan */
        box-shadow: 0 0 0 0.2rem rgba(0, 188, 212, 0.25);
    }

    .stSlider .stSlider-thumb {
        background-color: #ff4081; /* Slider thumb color */
        border: 2px solid #e0e0e0;
    }
    .stSlider .stSlider-track {
        background-color: #5a5a8a; /* Slider track color */
    }
    .stSlider .stSlider-track-fill {
        background-color: #00bcd4; /* Slider fill color */
    }

    .stAlert {
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
    }

    .prediction-output {
        background-color: #4a4a7a; /* Darker background for output */
        border: 2px solid #8aff8a; /* Bright green border for output */
        border-radius: 15px;
        padding: 30px;
        margin-top: 40px;
        text-align: center;
        font-size: 2.5em;
        font-weight: 800;
        color: #8aff8a; /* Bright green text */
        box-shadow: 0 8px 25px rgba(138, 255, 138, 0.3); /* Green glow effect */
        animation: pulse 1.5s infinite alternate; /* Pulsing animation */
    }

    .prediction-output .currency-text {
        font-size: 0.6em; /* Smaller text for currency type */
        font-weight: 500;
        color: #e0e0e0;
        display: block;
        margin-top: 5px;
    }

    .prediction-output .input-summary {
        font-size: 0.4em; /* Smaller font for input summary */
        color: #cccccc;
        margin-top: 20px;
        line-height: 1.4;
    }

    .stInfo {
        background-color: #3a3a5e;
        border-left: 5px solid #00bcd4;
        color: #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
    }

    /* Loading Spinner */
    .stSpinner > div > div {
        border-top-color: #ff4081 !important;
        border-left-color: #ff4081 !important;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 8px 25px rgba(138, 255, 138, 0.3); }
        100% { transform: scale(1.02); box-shadow: 0 10px 30px rgba(138, 255, 138, 0.5); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {
            padding: 25px;
            margin: 20px auto;
        }
        h1 {
            font-size: 2.2em;
        }
        .prediction-output {
            font-size: 2em;
            padding: 20px;
        }
        .prediction-output .input-summary {
            font-size: 0.35em;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💰 Employee Salary Prediction")

st.write(
    """
    Welcome to the Employee Salary Predictor!
    Enter the employee's details below to get an estimated salary.
    This tool uses a machine learning model to provide an estimated salary based on key attributes.
    """
)

# --- Input Fields ---
st.header("Employee Information")

# Using columns for a more organized input layout
col1, col2 = st.columns(2)

with col1:
    employee_name = st.text_input(
        "Employee Name",
        placeholder="e.g., Jane Doe",
        help="Enter the employee's full name."
    )
    # New feature: Experience Level
    experience_level_options = ["Entry-level", "Mid-level", "Senior", "Executive"]
    experience_level = st.selectbox(
        "Experience Level",
        experience_level_options,
        help="Select the employee's experience level."
    )

    # New feature: Company Location
    # This list should ideally be populated from unique values in your dataset's 'company_location' column
    # For now, providing a sample of common countries from ds_salaries.csv
    company_location_options = sorted([ # Sorted for better UX
        "US", "GB", "CA", "DE", "IN", "FR", "ES", "AU", "BR", "NL", "JP", "CH",
        "IT", "PL", "PT", "MX", "DK", "GR", "TR", "AT", "BE", "IE", "LU", "NG",
        "PK", "RU", "SG", "UA", "AE", "CL", "CO", "CY", "CZ", "EE", "FI", "GH",
        "HR", "HU", "IR", "MT", "NZ", "PH", "PR", "RO", "SI", "SK", "TH", "VN"
    ])
    company_location = st.selectbox(
        "Company Location (Country Code)",
        company_location_options,
        index=company_location_options.index("IN") if "IN" in company_location_options else 0, # Default to India if available
        help="Select the company's country location (e.g., US, IN)."
    )


with col2:
    # New feature: Work Year
    work_year = st.number_input(
        "Work Year",
        min_value=2020, # Based on ds_salaries data
        max_value=2025, # Current year or relevant future year
        value=2024,
        step=1,
        help="Enter the year the salary data pertains to."
    )

    # New feature: Remote Ratio
    remote_ratio = st.slider(
        "Remote Work Ratio (%)",
        min_value=0,
        max_value=100,
        value=0, # Default to no remote work
        step=5,
        help="Percentage of remote work (0 for no remote, 100 for fully remote)."
    )

    job_title_options = sorted([ # Sorted for better UX
        "Data Scientist", "Machine Learning Engineer", "Data Engineer",
        "Analytics Engineer", "Data Analyst", "Research Scientist",
        "AI Engineer", "Big Data Engineer", "BI Developer",
        "Computer Vision Engineer", "Data Architect", "Data Science Consultant",
        "Deep Learning Engineer", "ETL Developer", "Financial Data Analyst",
        "Head of Data", "Lead Data Analyst", "Lead Data Scientist",
        "ML Engineer", "Principal Data Scientist", "Research Engineer",
        "Software Engineer", "Other" # Keep 'Other' for flexibility
    ])
    job_title = st.selectbox(
        "Job Title",
        job_title_options,
        help="Select the employee's job title."
    )

# --- Temporary Debugging/Confirmation Section (Remove this later if satisfied) ---
st.markdown("---")
st.subheader("Current Selections (For Confirmation):")
st.write(f"**Employee Name:** {employee_name if employee_name else 'Not entered'}")
st.write(f"**Experience Level:** {experience_level}")
st.write(f"**Company Location:** {company_location}")
st.write(f"**Work Year:** {work_year}")
st.write(f"**Remote Ratio:** {remote_ratio}%")
st.write(f"**Job Title:** {job_title}")
st.markdown("---")
# --- End Temporary Section ---

# --- Prediction Button ---
if st.button("Predict Salary"):
    # Validate required fields (optional, but good practice)
    if not employee_name:
        st.warning("Please enter the Employee Name.")
    else:
        # Create a DataFrame with the input data, ensuring column names match the training features
        # and the order is consistent with how the model expects them.
        # Features for ds_salaries.csv: ['experience_level', 'job_title', 'company_location', 'remote_ratio', 'work_year']
        input_data = pd.DataFrame([[experience_level, job_title, company_location, remote_ratio, work_year]],
                                  columns=['experience_level', 'job_title', 'company_location', 'remote_ratio', 'work_year'])

        try:
            with st.spinner('Calculating salary...'):
                time.sleep(1) # Simulate a short delay for prediction
                predicted_salary_usd = model_pipeline.predict(input_data)[0]

            # Get local currency info
            local_currency_info = COUNTRY_CURRENCIES.get(company_location, {"symbol": "$", "name": "USD", "rate": 1.0})
            predicted_salary_local = predicted_salary_usd * local_currency_info["rate"]

            # Format output string
            output_string = f'<div class="prediction-output">Predicted Salary for {employee_name}:'
            output_string += f'<br>${predicted_salary_usd:,.2f}<span class="currency-text"> (USD)</span>'
            output_string += f'<br>₹{predicted_salary_usd * USD_TO_INR_RATE:,.2f}<span class="currency-text"> (INR)</span>' # Always show INR
            if local_currency_info["name"] != "USD" and local_currency_info["name"] != "INR": # Avoid duplicating USD/INR if they are local
                output_string += f'<br>{local_currency_info["symbol"]}{predicted_salary_local:,.2f}<span class="currency-text"> ({local_currency_info["name"]})</span>'

            # Add input summary to the output
            output_string += f'''
            <div class="input-summary">
                <p>Based on your inputs:</p>
                <ul>
                    <li>Experience Level: <b>{experience_level}</b></li>
                    <li>Job Title: <b>{job_title}</b></li>
                    <li>Company Location: <b>{company_location}</b></li>
                    <li>Work Year: <b>{work_year}</b></li>
                    <li>Remote Ratio: <b>{remote_ratio}%</b></li>
                </ul>
            </div>
            '''
            output_string += '</div>' # Close prediction-output div

            st.markdown(output_string, unsafe_allow_html=True)
            st.balloons() # A little celebration!

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model is correctly trained for these features.")


st.markdown("---")
st.info("This application uses a Machine Learning model to predict salaries. Predictions are estimates and should be used for informational purposes only. **The INR and other local currency conversions are direct exchanges from the USD prediction, not independent predictions of local market salaries. Real-time market rates and local economic factors can cause significant variations.**")

